#!/usr/bin/env python3
"""Run one Ultralytics experiment and save machine-readable metadata."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        help="Optionally transfer compatible weights into a custom model YAML.",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/experiments")
    )
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N model layers (useful for extreme few-shot runs).",
    )
    return parser.parse_args()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def resolve_test_image(data_yaml: Path) -> Path:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(config["path"])
    test_source = Path(config["test"])
    test_dir = test_source if test_source.is_absolute() else dataset_root / test_source
    candidates = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        candidates.extend(test_dir.glob(suffix))
    if not candidates:
        raise FileNotFoundError(f"No test images found under {test_dir}")
    return sorted(candidates)[0]


def measure_latency(model: YOLO, image_path: Path, imgsz: int, device: str) -> dict:
    for _ in range(10):
        model.predict(
            source=str(image_path), imgsz=imgsz, device=device, verbose=False, save=False
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(100):
        started = time.perf_counter()
        model.predict(
            source=str(image_path), imgsz=imgsz, device=device, verbose=False, save=False
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return {
        "mean_ms": sum(timings) / len(timings),
        "p50_ms": timings[len(timings) // 2],
        "p95_ms": timings[int(len(timings) * 0.95) - 1],
        "fps_from_mean": 1000.0 / (sum(timings) / len(timings)),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    run_dir = (args.project / args.name).resolve()
    metadata = {
        "name": args.name,
        "data": str(args.data.resolve()),
        "model": str(args.model.resolve()),
        "pretrained_weights": (
            str(args.pretrained_weights.resolve()) if args.pretrained_weights else None
        ),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "eval_split": args.eval_split,
        "freeze": args.freeze,
        "augmentation": {
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "fliplr": args.fliplr,
            "flipud": args.flipud,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "multi_scale": args.multi_scale,
        },
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_revision": git_revision(),
        "started_at_unix": time.time(),
    }

    model = YOLO(str(args.model))
    if args.pretrained_weights:
        model.load(str(args.pretrained_weights))
    model.train(
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        cache=args.cache,
        close_mosaic=args.close_mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        mixup=args.mixup,
        multi_scale=args.multi_scale,
        freeze=args.freeze,
        deterministic=True,
        pretrained=True,
        plots=True,
        exist_ok=False,
    )
    best_model = YOLO(str(run_dir / "weights" / "best.pt"))
    validation = best_model.val(
        data=str(args.data),
        split=args.eval_split,
        imgsz=args.imgsz,
        batch=1,
        device=args.device,
    )
    latency = measure_latency(
        best_model, resolve_test_image(args.data.resolve()), args.imgsz, args.device
    )

    metadata["completed_at_unix"] = time.time()
    metadata["test_metrics"] = {
        "map50_95": float(validation.box.map),
        "map50": float(validation.box.map50),
        "map75": float(validation.box.map75),
        "per_class_map50_95": [float(value) for value in validation.box.maps],
        "fitness": float(validation.fitness),
    }
    metadata["validation_results_dict"] = {
        key: float(value) for key, value in validation.results_dict.items()
    }
    metadata["latency"] = latency
    metadata["parameter_count"] = sum(
        parameter.numel() for parameter in best_model.model.parameters()
    )
    metadata["best_model_size_mb"] = (run_dir / "weights" / "best.pt").stat().st_size / (
        1024**2
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(metadata, indent=2)
    (run_dir / "experiment_summary.json").write_text(summary_text, encoding="utf-8")
    artifact_dir = args.artifact_dir.resolve() / args.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "experiment_summary.json").write_text(
        summary_text, encoding="utf-8"
    )
    for filename in (
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "labels.jpg",
    ):
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, artifact_dir / filename)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo_config").resolve()))
    main()
