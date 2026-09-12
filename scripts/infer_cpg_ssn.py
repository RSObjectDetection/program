#!/usr/bin/env python3
"""Run image-level OK/NG inference with a trained CPG-SSN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare_okng_dataset import fixed_starts
from run_cpg_ssn import (
    CPGSuperSimpleNet,
    TileDataset,
    build_prototypes,
    prototype_batch,
    read_rows,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict whole-image OK/NG labels using CPG-SSN."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Single image to predict")
    source.add_argument("--input-dir", type=Path, help="Directory of images to predict")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument(
        "--prototypes",
        type=Path,
        help="Saved prototypes.pt. If omitted, prototypes are rebuilt from --manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Training tiles.csv used to rebuild prototypes when --prototypes is absent.",
    )
    parser.add_argument(
        "--template",
        help="Template ID used in prototype keys; required if the manifest has multiple templates.",
    )
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--prototype-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.7863940596580505)
    parser.add_argument("--aggregation", choices=("max", "top2", "top3"), default="top2")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def model_config(backbone: str, epochs: int = 12) -> dict:
    """Configuration used by the S08 deployment candidate."""
    return {
        "backbone": backbone,
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": False,
        "noise_std": 0.015,
        "perlin_thr": 0.6,
        "epochs": epochs,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": False,
    }


def collect_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        return [args.image.resolve()]
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    images = sorted(
        path.resolve()
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found under {args.input_dir}")
    return images


def load_or_build_prototypes(model, args, device):
    manifest_rows = None
    if args.prototypes:
        payload = torch.load(args.prototypes, map_location="cpu")
        prototypes = payload["prototypes"]
        default = payload["default"]
        templates = payload.get("templates", [])
    else:
        if args.manifest is None:
            raise ValueError("Pass --prototypes or --manifest to provide normal prototypes")
        manifest_rows = read_rows(args.manifest, shots=0)
        prototypes, default = build_prototypes(
            model,
            manifest_rows["train"],
            args.image_size,
            args.batch,
            args.workers,
            device,
            args.prototype_size,
        )
        templates = sorted({row["template"] for row in manifest_rows["train"]})
    if not prototypes:
        raise ValueError("The prototype bank is empty")
    if not templates:
        templates = sorted({key.split(":", 1)[0] for key in prototypes})
    return prototypes, default, templates


def resolve_template(requested: str | None, templates: list[str]) -> str:
    if requested:
        if templates and requested not in templates:
            raise ValueError(
                f"Unknown template {requested!r}; available templates: {', '.join(templates)}"
            )
        return requested
    if len(templates) == 1:
        return templates[0]
    raise ValueError(
        "Multiple templates are available. Pass --template with one of: "
        + ", ".join(templates)
    )


def image_rows(path: Path, template: str, tile_size: int, overlap: int) -> list[dict]:
    from PIL import Image

    with Image.open(path) as opened:
        width, height = opened.size
    rows = []
    for y0 in fixed_starts(height, tile_size, overlap):
        for x0 in fixed_starts(width, tile_size, overlap):
            x1, y1 = min(x0 + tile_size, width), min(y0 + tile_size, height)
            rows.append(
                {
                    "image": str(path),
                    "label": 0,
                    "source_id": path.stem,
                    "source_class": "unknown",
                    "template": template,
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "prototype_key": f"{template}:{x0}:{y0}:{x1-x0}:{y1-y0}",
                }
            )
    return rows


def aggregate(scores: list[float], mode: str) -> float:
    ordered = sorted(scores, reverse=True)
    count = {"max": 1, "top2": 2, "top3": 3}[mode]
    return float(np.mean(ordered[: min(count, len(ordered))]))


@torch.no_grad()
def predict_image(model, path, template, prototypes, default, args, device) -> dict:
    rows = image_rows(path, template, args.tile_size, args.overlap)
    loader = DataLoader(
        TileDataset(rows, args.image_size, training=False),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    scores = []
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        proto = prototype_batch(batch["prototype_key"], prototypes, default, device)
        _, logits = model(batch["image"].to(device, non_blocking=True), prototypes=proto)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    image_score = aggregate(scores, args.aggregation)
    top_indices = np.argsort(np.asarray(scores))[::-1][: min(3, len(scores))]
    return {
        "image": str(path),
        "template": template,
        "score": image_score,
        "prediction": "NG" if image_score >= args.threshold else "OK",
        "threshold": args.threshold,
        "aggregation": args.aggregation,
        "tile_count": len(scores),
        "elapsed_ms": elapsed_ms,
        "top_tile_scores": json.dumps([scores[index] for index in top_indices]),
    }


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0 or args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("Require tile-size > 0 and 0 <= overlap < tile-size")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    sys.path.insert(0, str(args.official_repo.resolve()))
    from model.supersimplenet import SuperSimpleNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CPGSuperSimpleNet(
        SuperSimpleNet,
        args.image_size,
        model_config(args.backbone),
        use_prototype=True,
    ).to(device)
    model.load_model(args.checkpoint)
    model.eval()
    prototypes, default, templates = load_or_build_prototypes(model, args, device)
    template = resolve_template(args.template, templates)
    results = [
        predict_image(model, path, template, prototypes, default, args, device)
        for path in collect_images(args)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
