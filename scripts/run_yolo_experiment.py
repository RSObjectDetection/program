#!/usr/bin/env python3
"""Run one Ultralytics experiment and save machine-readable metadata."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--close-mosaic", type=int, default=10)
    return parser.parse_args()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    run_dir = (args.project / args.name).resolve()
    metadata = {
        "name": args.name,
        "data": str(args.data.resolve()),
        "model": str(args.model.resolve()),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_revision": git_revision(),
        "started_at_unix": time.time(),
    }

    model = YOLO(str(args.model))
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
        deterministic=True,
        pretrained=True,
        plots=True,
        exist_ok=False,
    )
    best_model = YOLO(str(run_dir / "weights" / "best.pt"))
    validation = best_model.val(
        data=str(args.data), split="test", imgsz=args.imgsz, batch=1, device=args.device
    )
    speed = best_model.benchmark(
        data=str(args.data), imgsz=args.imgsz, device=args.device, format="onnx", half=False
    )

    metadata["completed_at_unix"] = time.time()
    metadata["test_metrics"] = {
        "map50_95": float(validation.box.map),
        "map50": float(validation.box.map50),
        "map75": float(validation.box.map75),
        "per_class_map50_95": [float(value) for value in validation.box.maps],
        "fitness": float(validation.fitness),
    }
    metadata["benchmark"] = str(speed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo_config").resolve()))
    main()

