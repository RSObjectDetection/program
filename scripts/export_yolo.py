#!/usr/bin/env python3
"""Export a selected Ultralytics checkpoint and record reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "engine"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = Path(
        YOLO(str(args.checkpoint)).export(
            format=args.format,
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            simplify=args.simplify,
            half=args.half,
            device=args.device,
        )
    )
    destination = args.output_dir / exported.name
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)
    metadata = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "artifact": str(destination.resolve()),
        "format": args.format,
        "image_size": args.imgsz,
        "batch": 1,
        "dynamic": False,
        "half": args.half,
        "simplified": args.simplify,
        "size_mb": destination.stat().st_size / (1024 * 1024),
        "sha256": sha256(destination),
    }
    (args.output_dir / f"{args.format}_export.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
