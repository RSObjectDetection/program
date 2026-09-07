#!/usr/bin/env python3
"""Convert VOC annotations to reproducible YOLO datasets."""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml


CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["group_holdout", "random_stratified"], required=True
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--shots-per-class", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def record_for_xml(xml_path: Path, dataset_root: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
    image_path = dataset_root / "images" / xml_path.parent.name / filename
    labels = []
    classes = set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {name!r} in {xml_path}")
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height
        labels.append(f"{CLASS_TO_ID[name]} {x_center:.8f} {y_center:.8f} {box_w:.8f} {box_h:.8f}")
        classes.add(name)
    return {
        "stem": xml_path.stem,
        "group": xml_path.stem.split("_", 1)[0],
        "folder": xml_path.parent.name,
        "image_path": image_path,
        "labels": labels,
        "classes": classes,
    }


def assign_group_holdout(records: list[dict]) -> dict[str, list[dict]]:
    assignments = {
        "train": {"01", "04", "05", "06", "07", "08"},
        "val": {"09", "10"},
        "test": {"11", "12"},
    }
    result = {name: [] for name in assignments}
    for record in records:
        destination = next(
            (name for name, groups in assignments.items() if record["group"] in groups),
            None,
        )
        if destination is None:
            raise ValueError(f"Unassigned template group: {record['group']}")
        result[destination].append(record)
    return result


def assign_random_stratified(records: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_folder: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_folder[record["folder"]].append(record)
    result = {"train": [], "val": [], "test": []}
    for folder_records in by_folder.values():
        rng.shuffle(folder_records)
        count = len(folder_records)
        train_end = max(1, round(count * 0.70))
        val_end = min(count - 1, train_end + max(1, round(count * 0.15)))
        result["train"].extend(folder_records[:train_end])
        result["val"].extend(folder_records[train_end:val_end])
        result["test"].extend(folder_records[val_end:])
    return result


def apply_few_shot(records: list[dict], shots: int, seed: int) -> list[dict]:
    if shots <= 0:
        return records
    rng = random.Random(seed)
    by_folder: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_folder[record["folder"]].append(record)
    selected = []
    for folder in sorted(by_folder):
        candidates = sorted(by_folder[folder], key=lambda item: item["stem"])
        rng.shuffle(candidates)
        selected.extend(candidates[:shots])
    return selected


def main() -> None:
    args = parse_args()
    name = args.split
    if args.shots_per_class:
        name += f"_{args.shots_per_class}shot"
    output_dir = args.output_root / name
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} exists; pass --force to rebuild")
        shutil.rmtree(output_dir)

    xml_paths = sorted((args.dataset_root / "Annotations").rglob("*.xml"))
    records = [record_for_xml(path, args.dataset_root) for path in xml_paths]
    if args.split == "group_holdout":
        splits = assign_group_holdout(records)
    else:
        splits = assign_random_stratified(records, args.seed)
    splits["train"] = apply_few_shot(
        splits["train"], args.shots_per_class, args.seed
    )

    manifest_lines = ["split,image,template_group,source_folder,classes"]
    summary = {}
    for split_name, split_records in splits.items():
        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        class_images = Counter()
        class_boxes = Counter()
        for record in sorted(split_records, key=lambda item: item["stem"]):
            destination_image = image_dir / record["image_path"].name
            destination_image.symlink_to(record["image_path"])
            (label_dir / f"{record['stem']}.txt").write_text(
                "\n".join(record["labels"]) + "\n", encoding="utf-8"
            )
            for class_name in record["classes"]:
                class_images[class_name] += 1
            for label in record["labels"]:
                class_boxes[CLASSES[int(label.split()[0])]] += 1
            manifest_lines.append(
                f"{split_name},{record['image_path']},{record['group']},"
                f"{record['folder']},{'|'.join(sorted(record['classes']))}"
            )
        summary[split_name] = {
            "images": len(split_records),
            "class_images": dict(class_images),
            "class_boxes": dict(class_boxes),
            "template_groups": sorted({item["group"] for item in split_records}),
        }

    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASSES)},
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "manifest.csv").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "split_summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()

