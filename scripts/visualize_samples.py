#!/usr/bin/env python3
"""Create a compact visual QA grid with Pascal VOC boxes."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumb-width", type=int, default=640)
    return parser.parse_args()


def annotate(image_path: Path, xml_path: Path, target_width: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    scale = target_width / image.width
    target_height = round(image.height * scale)
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name = obj.findtext("name", "unknown")
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        coordinates = [
            float(bbox.findtext("xmin", "0")) * scale,
            float(bbox.findtext("ymin", "0")) * scale,
            float(bbox.findtext("xmax", "0")) * scale,
            float(bbox.findtext("ymax", "0")) * scale,
        ]
        draw.rectangle(coordinates, outline=(255, 60, 40), width=4)
        label_x, label_y = coordinates[0], max(0, coordinates[1] - 18)
        draw.rectangle(
            (label_x, label_y, label_x + len(name) * 8 + 6, label_y + 18),
            fill=(255, 60, 40),
        )
        draw.text((label_x + 3, label_y + 2), name, fill="white", font=ImageFont.load_default())
    return image


def main() -> None:
    args = parse_args()
    rows = []
    annotation_root = args.dataset_root / "Annotations"
    image_root = args.dataset_root / "images"
    for class_dir in sorted(annotation_root.iterdir()):
        if not class_dir.is_dir():
            continue
        xml_path = sorted(class_dir.glob("*.xml"))[0]
        image_path = image_root / class_dir.name / f"{xml_path.stem}.jpg"
        rows.append(annotate(image_path, xml_path, args.thumb_width))

    gap = 12
    canvas_width = args.thumb_width * 2 + gap * 3
    row_heights = []
    for index in range(0, len(rows), 2):
        row_heights.append(max(image.height for image in rows[index : index + 2]))
    canvas_height = sum(row_heights) + gap * (len(row_heights) + 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    y = gap
    for row_index, row_height in enumerate(row_heights):
        for column, image in enumerate(rows[row_index * 2 : row_index * 2 + 2]):
            canvas.paste(image, (gap + column * (args.thumb_width + gap), y))
        y += row_height + gap
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()
