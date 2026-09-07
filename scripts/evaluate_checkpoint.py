#!/usr/bin/env python3
"""Evaluate a frozen checkpoint once on the held-out test split."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from ultralytics import YOLO

from run_yolo_experiment import measure_latency, resolve_test_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_test"))
    parser.add_argument("--project", type=Path, default=Path("runs/test"))
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.checkpoint))
    result = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        batch=1,
        device=args.device,
        project=str(args.project),
        name=args.name,
        plots=True,
    )
    summary = {
        "name": args.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "split": "test",
        "imgsz": args.imgsz,
        "hostname": platform.node(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "metrics": {
            "precision": float(result.results_dict["metrics/precision(B)"]),
            "recall": float(result.results_dict["metrics/recall(B)"]),
            "map50": float(result.box.map50),
            "map50_95": float(result.box.map),
            "map75": float(result.box.map75),
            "per_class_map50_95": [float(value) for value in result.box.maps],
        },
        "latency": measure_latency(
            model, resolve_test_image(args.data.resolve()), args.imgsz, args.device
        ),
        "parameter_count": sum(p.numel() for p in model.model.parameters()),
        "model_size_mb": args.checkpoint.stat().st_size / (1024**2),
    }
    output = args.output_dir / args.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
