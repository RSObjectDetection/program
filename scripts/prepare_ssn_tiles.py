#!/usr/bin/env python3
"""Create tiled images and coarse box masks for SuperSimpleNet experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def starts(length: int, tile: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, tile))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def read_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = line.split()
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        boxes.append(
            (
                int(cls),
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            )
        )
    return boxes


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    summary = {}
    tile = args.tile_size

    for split in ("train", "val", "test"):
        image_out = args.output_root / "images" / split
        mask_out = args.output_root / "masks" / split
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        source_dir = args.yolo_root / "images" / split
        for image_path in sorted(source_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            label_path = args.yolo_root / "labels" / split / f"{image_path.stem}.txt"
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                boxes = read_boxes(label_path, width, height)
                for y0 in starts(height, tile):
                    for x0 in starts(width, tile):
                        x1, y1 = min(x0 + tile, width), min(y0 + tile, height)
                        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
                        draw = ImageDraw.Draw(mask)
                        classes = set()
                        for cls, bx0, by0, bx1, by1 in boxes:
                            ix0, iy0 = max(x0, bx0), max(y0, by0)
                            ix1, iy1 = min(x1, bx1), min(y1, by1)
                            if ix1 <= ix0 or iy1 <= iy0:
                                continue
                            draw.rectangle(
                                (ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0),
                                fill=255,
                            )
                            classes.add(cls)
                        label = int(bool(classes))
                        name = f"{image_path.stem}__x{x0}_y{y0}"
                        tile_image_path = image_out / f"{name}.jpg"
                        tile_mask_path = mask_out / f"{name}.png"
                        image.crop((x0, y0, x1, y1)).save(
                            tile_image_path, quality=args.jpeg_quality
                        )
                        mask.save(tile_mask_path)
                        counts["tiles"] += 1
                        counts["anomalous" if label else "normal"] += 1
                        manifest.append(
                            {
                                "split": split,
                                "image": str(tile_image_path.resolve()),
                                "mask": str(tile_mask_path.resolve()),
                                "label": label,
                                "classes": "|".join(map(str, sorted(classes))),
                                "source": str(image_path.resolve()),
                                "x": x0,
                                "y": y0,
                            }
                        )
        summary[split] = dict(counts)

    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
