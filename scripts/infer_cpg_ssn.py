#!/usr/bin/env python3
"""Run image-level OK/NG inference with a trained CPG-SSN checkpoint."""

from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

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
    parser.add_argument(
        "--threshold",
        type=float,
        help="Decision threshold. Defaults to validation threshold in checkpoint-dir/experiment_summary.json.",
    )
    parser.add_argument("--aggregation", choices=("max", "top2", "top3"), default="top2")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=4,
        help="Threads used by the decode-once PIL preprocessing path.",
    )
    parser.add_argument(
        "--legacy-loader",
        action="store_true",
        help="Use the old per-tile JPEG loader for before/after benchmarking.",
    )
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


def resolve_threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return args.threshold
    summary_path = args.checkpoint.parent / "experiment_summary.json"
    if not summary_path.is_file():
        raise ValueError(
            "Pass --threshold or place experiment_summary.json next to the checkpoint"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary["validation"][args.aggregation]["threshold"])


def tile_rows(
    path: Path,
    template: str,
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> list[dict]:
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


def image_rows(path: Path, template: str, tile_size: int, overlap: int) -> list[dict]:

    with Image.open(path) as opened:
        width, height = opened.size
    return tile_rows(path, template, width, height, tile_size, overlap)


def aggregate(scores: list[float], mode: str) -> float:
    ordered = sorted(scores, reverse=True)
    count = {"max": 1, "top2": 2, "top3": 3}[mode]
    return float(np.mean(ordered[: min(count, len(ordered))]))


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def preprocess_tile(image: Image.Image, row: dict, image_size: int) -> torch.Tensor:
    crop = image.crop(
        (row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"])
    )
    crop = TF.resize(crop, [image_size, image_size], antialias=True)
    return TF.normalize(
        TF.to_tensor(crop),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )


@torch.inference_mode()
def warmup_model(model, default, args, device: str) -> float:
    if device != "cuda":
        return 0.0
    started = time.perf_counter()
    batch_size = max(1, args.batch)
    dummy_images = torch.zeros(
        (batch_size, 3, args.image_size, args.image_size), device=device
    )
    dummy_prototypes = default.unsqueeze(0).expand(batch_size, *default.shape)
    model(dummy_images, prototypes=dummy_prototypes)
    synchronize(device)
    del dummy_images, dummy_prototypes
    return (time.perf_counter() - started) * 1000


def prediction_result(path, template, scores, args, timings, pipeline) -> dict:
    aggregation_started = time.perf_counter()
    image_score = aggregate(scores, args.aggregation)
    aggregation_ms = (time.perf_counter() - aggregation_started) * 1000
    top_indices = np.argsort(np.asarray(scores))[::-1][: min(3, len(scores))]
    timings["pipeline_total_ms"] = timings["elapsed_ms"] + aggregation_ms
    return {
        "image": str(path),
        "template": template,
        "score": image_score,
        "prediction": "NG" if image_score >= args.threshold else "OK",
        "threshold": args.threshold,
        "aggregation": args.aggregation,
        "pipeline": pipeline,
        "tile_count": len(scores),
        **timings,
        "aggregation_ms": aggregation_ms,
        "top_tile_scores": json.dumps([scores[index] for index in top_indices]),
    }


@torch.inference_mode()
def predict_image_legacy(model, path, template, prototypes, default, args, device) -> dict:
    """Original training-oriented loader, retained only for reproducible comparison."""
    rows = image_rows(path, template, args.tile_size, args.overlap)
    loader = DataLoader(
        TileDataset(rows, args.image_size, training=False),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    scores = []
    synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        proto = prototype_batch(batch["prototype_key"], prototypes, default, device)
        _, logits = model(batch["image"].to(device, non_blocking=True), prototypes=proto)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    timings = {
        "elapsed_ms": elapsed_ms,
        "decode_ms": None,
        "grid_ms": None,
        "preprocess_ms": None,
        "image_h2d_ms": None,
        "prototype_select_ms": None,
        "model_forward_ms": None,
        "score_d2h_ms": None,
        "legacy_loader_and_inference_ms": elapsed_ms,
    }
    return prediction_result(path, template, scores, args, timings, "legacy")


@torch.inference_mode()
def predict_image(model, path, template, prototypes, default, args, device) -> dict:
    """Decode once, crop in memory, and batch all tiles without DataLoader workers."""
    total_started = time.perf_counter()

    stage_started = time.perf_counter()
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
    decode_ms = (time.perf_counter() - stage_started) * 1000

    stage_started = time.perf_counter()
    rows = tile_rows(
        path,
        template,
        image.width,
        image.height,
        args.tile_size,
        args.overlap,
    )
    grid_ms = (time.perf_counter() - stage_started) * 1000

    stage_started = time.perf_counter()
    transform = lambda row: preprocess_tile(image, row, args.image_size)
    if args.preprocess_workers > 1:
        with ThreadPoolExecutor(max_workers=args.preprocess_workers) as pool:
            tiles = list(pool.map(transform, rows))
    else:
        tiles = [transform(row) for row in rows]
    preprocess_ms = (time.perf_counter() - stage_started) * 1000

    scores = []
    image_h2d_ms = 0.0
    prototype_select_ms = 0.0
    model_forward_ms = 0.0
    score_d2h_ms = 0.0
    for offset in range(0, len(rows), args.batch):
        chunk = rows[offset : offset + args.batch]
        stage_started = time.perf_counter()
        cpu_batch = torch.stack(tiles[offset : offset + args.batch])
        if device == "cuda":
            cpu_batch = cpu_batch.pin_memory()
        preprocess_ms += (time.perf_counter() - stage_started) * 1000

        synchronize(device)
        stage_started = time.perf_counter()
        image_batch = cpu_batch.to(device, non_blocking=device == "cuda")
        synchronize(device)
        image_h2d_ms += (time.perf_counter() - stage_started) * 1000

        synchronize(device)
        stage_started = time.perf_counter()
        proto = torch.stack(
            [prototypes.get(row["prototype_key"], default) for row in chunk]
        )
        synchronize(device)
        prototype_select_ms += (time.perf_counter() - stage_started) * 1000

        synchronize(device)
        stage_started = time.perf_counter()
        _, logits = model(image_batch, prototypes=proto)
        synchronize(device)
        model_forward_ms += (time.perf_counter() - stage_started) * 1000

        stage_started = time.perf_counter()
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        synchronize(device)
        score_d2h_ms += (time.perf_counter() - stage_started) * 1000

    elapsed_ms = (time.perf_counter() - total_started) * 1000
    timings = {
        "elapsed_ms": elapsed_ms,
        "decode_ms": decode_ms,
        "grid_ms": grid_ms,
        "preprocess_ms": preprocess_ms,
        "image_h2d_ms": image_h2d_ms,
        "prototype_select_ms": prototype_select_ms,
        "model_forward_ms": model_forward_ms,
        "score_d2h_ms": score_d2h_ms,
        "legacy_loader_and_inference_ms": None,
    }
    return prediction_result(path, template, scores, args, timings, "decode_once")


def main() -> None:
    main_started = time.perf_counter()
    args = parse_args()
    cli_parse_ms = (time.perf_counter() - main_started) * 1000
    if args.tile_size <= 0 or args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("Require tile-size > 0 and 0 <= overlap < tile-size")
    if args.batch <= 0 or args.workers < 0 or args.preprocess_workers <= 0:
        raise ValueError("Require batch > 0, workers >= 0, and preprocess-workers > 0")
    args.threshold = resolve_threshold(args)
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    stage_started = time.perf_counter()
    sys.path.insert(0, str(args.official_repo.resolve()))
    from model.supersimplenet import SuperSimpleNet
    official_import_ms = (time.perf_counter() - stage_started) * 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage_started = time.perf_counter()
    model = CPGSuperSimpleNet(
        SuperSimpleNet,
        args.image_size,
        model_config(args.backbone),
        use_prototype=True,
    ).to(device)
    synchronize(device)
    model_init_ms = (time.perf_counter() - stage_started) * 1000

    stage_started = time.perf_counter()
    model.load_model(args.checkpoint)
    model.eval()
    synchronize(device)
    checkpoint_load_ms = (time.perf_counter() - stage_started) * 1000

    stage_started = time.perf_counter()
    prototypes, default, templates = load_or_build_prototypes(model, args, device)
    prototype_load_ms = (time.perf_counter() - stage_started) * 1000

    stage_started = time.perf_counter()
    prototypes = {key: value.to(device) for key, value in prototypes.items()}
    default = default.to(device)
    synchronize(device)
    prototype_to_device_ms = (time.perf_counter() - stage_started) * 1000

    cuda_warmup_ms = warmup_model(model, default, args, device)

    template = resolve_template(args.template, templates)
    images = collect_images(args)
    startup_total_ms = (time.perf_counter() - PROCESS_STARTED) * 1000
    startup_timings = {
        "python_import_ms": (main_started - PROCESS_STARTED) * 1000,
        "cli_parse_ms": cli_parse_ms,
        "official_model_import_ms": official_import_ms,
        "model_init_ms": model_init_ms,
        "checkpoint_load_ms": checkpoint_load_ms,
        "prototype_load_ms": prototype_load_ms,
        "prototype_to_device_ms": prototype_to_device_ms,
        "cuda_warmup_ms": cuda_warmup_ms,
        "startup_total_ms": startup_total_ms,
    }
    predictor = predict_image_legacy if args.legacy_loader else predict_image
    results = [
        {**startup_timings, **predictor(model, path, template, prototypes, default, args, device)}
        for path in images
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_started = time.perf_counter()
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    output_write_ms = (time.perf_counter() - output_started) * 1000
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(
        f"Saved {len(results)} predictions to {args.output} "
        f"(output_write_ms={output_write_ms:.3f})"
    )


if __name__ == "__main__":
    main()
