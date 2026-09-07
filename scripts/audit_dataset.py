#!/usr/bin/env python3
"""Audit the supplied Pascal VOC PCB defect dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def main() -> None:
    args = parse_args()
    image_root = args.dataset_root / "images"
    annotation_root = args.dataset_root / "Annotations"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_root.rglob("*.jpg"))
    xml_paths = sorted(annotation_root.rglob("*.xml"))
    image_by_stem = {path.stem.lower(): path for path in image_paths}
    xml_by_stem = {path.stem.lower(): path for path in xml_paths}

    issues: list[str] = []
    image_sizes: Counter[str] = Counter()
    folder_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    class_boxes: Counter[str] = Counter()
    class_images: defaultdict[str, set[str]] = defaultdict(set)
    box_area_ratios: defaultdict[str, list[float]] = defaultdict(list)
    box_widths: defaultdict[str, list[float]] = defaultdict(list)
    box_heights: defaultdict[str, list[float]] = defaultdict(list)

    for image_path in image_paths:
        folder_counts[image_path.parent.name] += 1
        group_counts[image_path.stem.split("_", 1)[0]] += 1
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            image_sizes[f"{width}x{height}"] += 1
        except Exception as exc:  # audit should continue and report every bad file
            issues.append(f"Unreadable image {image_path}: {exc}")

    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
            width = int(root.findtext("size/width", "0"))
            height = int(root.findtext("size/height", "0"))
            filename = root.findtext("filename", "")
            if not filename or Path(filename).stem.lower() != xml_path.stem.lower():
                issues.append(f"Filename mismatch in {xml_path}: {filename!r}")
            for obj in root.findall("object"):
                name = (obj.findtext("name") or "unknown").strip().lower()
                bbox = obj.find("bndbox")
                if bbox is None:
                    issues.append(f"Missing bbox in {xml_path}")
                    continue
                xmin = float(bbox.findtext("xmin", "0"))
                ymin = float(bbox.findtext("ymin", "0"))
                xmax = float(bbox.findtext("xmax", "0"))
                ymax = float(bbox.findtext("ymax", "0"))
                if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                    issues.append(
                        f"Out-of-bounds bbox in {xml_path}: "
                        f"{xmin},{ymin},{xmax},{ymax} for {width}x{height}"
                    )
                box_w = max(0.0, xmax - xmin)
                box_h = max(0.0, ymax - ymin)
                class_boxes[name] += 1
                class_images[name].add(xml_path.stem)
                if width and height:
                    box_area_ratios[name].append((box_w * box_h) / (width * height))
                    box_widths[name].append(box_w / width)
                    box_heights[name].append(box_h / height)
        except Exception as exc:
            issues.append(f"Unreadable XML {xml_path}: {exc}")

    missing_xml = sorted(set(image_by_stem) - set(xml_by_stem))
    missing_images = sorted(set(xml_by_stem) - set(image_by_stem))
    issues.extend(f"Missing XML for {stem}" for stem in missing_xml)
    issues.extend(f"Missing image for {stem}" for stem in missing_images)

    class_stats = {}
    for name in sorted(class_boxes):
        areas = box_area_ratios[name]
        class_stats[name] = {
            "images": len(class_images[name]),
            "boxes": class_boxes[name],
            "boxes_per_image": class_boxes[name] / max(1, len(class_images[name])),
            "area_ratio_median": statistics.median(areas),
            "area_ratio_p10": percentile(areas, 0.10),
            "area_ratio_p90": percentile(areas, 0.90),
            "width_ratio_median": statistics.median(box_widths[name]),
            "height_ratio_median": statistics.median(box_heights[name]),
        }

    report = {
        "dataset_root": str(args.dataset_root),
        "images": len(image_paths),
        "annotations": len(xml_paths),
        "image_sizes": dict(image_sizes),
        "folder_counts": dict(sorted(folder_counts.items())),
        "template_group_counts": dict(sorted(group_counts.items())),
        "classes": class_stats,
        "issues": issues,
    }
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Dataset audit",
        "",
        f"- Images: {len(image_paths)}",
        f"- XML annotations: {len(xml_paths)}",
        f"- Distinct image sizes: {len(image_sizes)}",
        f"- Validation issues: {len(issues)}",
        "",
        "## Template groups",
        "",
        "| Group | Images |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(group_counts.items()))
    lines.extend(
        [
            "",
            "## Defect classes",
            "",
            "| Class | Images | Boxes | Boxes/image | Median box area | P10 box area |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in class_stats.items():
        lines.append(
            f"| {name} | {stats['images']} | {stats['boxes']} | "
            f"{stats['boxes_per_image']:.2f} | {stats['area_ratio_median']:.4%} | "
            f"{stats['area_ratio_p10']:.4%} |"
        )
    lines.extend(["", "## Image sizes", "", "| Size | Images |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in image_sizes.most_common())
    if issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in issues)
    (args.output_dir / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

