#!/usr/bin/env python3
"""Build a paired, leakage-safe OK/NG dataset from the HRIPCB-style archive.

Six defect-class images sharing ``template`` and ``index`` are near-identical
renderings of the same board.  A channel-wise median removes their sparse,
class-specific defects and creates one seamless candidate OK image per group.
The six source images and their generated OK image always stay in one split.

Tiles are represented by coordinates in a CSV instead of being materialized on
disk.  Training keeps only NG tiles containing an annotated defect centre;
validation and test retain the complete fixed grid for honest image-level
aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = (
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--train-groups", type=int, default=80)
    parser.add_argument("--val-groups", type=int, default=15)
    parser.add_argument("--test-groups", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_record(xml_path: Path, dataset_root: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
    image_path = dataset_root / "images" / xml_path.parent.name / filename
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    boxes = []
    names = set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        names.add(name)
        boxes.append(
            (
                float(bbox.findtext("xmin", "0")),
                float(bbox.findtext("ymin", "0")),
                float(bbox.findtext("xmax", "0")),
                float(bbox.findtext("ymax", "0")),
            )
        )
    if len(names) != 1:
        raise ValueError(f"Expected one defect type in {xml_path}, got {names}")
    defect_class = next(iter(names))
    template = xml_path.stem.split("_", 1)[0]
    sample_index = xml_path.stem.rsplit("_", 1)[-1]
    return {
        "stem": xml_path.stem,
        "template": template,
        "sample_index": sample_index,
        "pair_key": f"{template}_{sample_index}",
        "class": defect_class,
        "image": image_path,
        "width": width,
        "height": height,
        "boxes": boxes,
    }


def fixed_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile-size")
    values = list(range(0, length - tile + 1, step))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def contains_box_center(boxes: list[tuple], x0: int, y0: int, x1: int, y1: int) -> bool:
    for bx0, by0, bx1, by1 in boxes:
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if x0 <= cx < x1 and y0 <= cy < y1:
            return True
    return False


def split_pair_keys(
    keys: list[str], train_count: int, val_count: int, test_count: int, seed: int
) -> tuple[dict[str, str], dict[str, int]]:
    requested = train_count + val_count + test_count
    if requested > len(keys):
        raise ValueError(f"Requested {requested} groups, only {len(keys)} complete groups exist")
    rng = random.Random(seed)
    ordered = sorted(keys)
    rng.shuffle(ordered)
    selected = ordered[:requested]
    train = selected[:train_count]
    val = selected[train_count : train_count + val_count]
    test = selected[train_count + val_count :]
    assignment = {key: "train" for key in train}
    assignment.update({key: "val" for key in val})
    assignment.update({key: "test" for key in test})
    train_rank = {key: index + 1 for index, key in enumerate(train)}
    return assignment, train_rank


def robust_median_image(records: list[dict]) -> tuple[Image.Image, dict]:
    arrays = []
    shape = None
    for record in records:
        with Image.open(record["image"]) as opened:
            array = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        if shape is None:
            shape = array.shape
        if array.shape != shape:
            raise ValueError(f"Shape mismatch in pair group {record['pair_key']}")
        arrays.append(array)
    stack = np.stack(arrays, axis=0)
    ordered = np.sort(stack, axis=0)
    middle = (
        ordered[(len(arrays) - 1) // 2].astype(np.uint16)
        + ordered[len(arrays) // 2].astype(np.uint16)
    ) // 2
    median = middle.astype(np.uint8)
    abs_delta = np.abs(stack.astype(np.int16) - median[None].astype(np.int16))
    metrics = {
        "members": len(records),
        "mean_absolute_delta": float(abs_delta.mean()),
        "p99_absolute_delta": float(np.percentile(abs_delta, 99)),
        "max_absolute_delta": int(abs_delta.max()),
    }
    return Image.fromarray(median, mode="RGB"), metrics


def add_tile_rows(
    rows: list[dict], record: dict, split: str, image_path: Path, label: int,
    source_id: str, source_class: str, tile: int, overlap: int, train_rank: int,
    synthetic_ok: int,
) -> None:
    width, height = record["width"], record["height"]
    for y0 in fixed_starts(height, tile, overlap):
        for x0 in fixed_starts(width, tile, overlap):
            x1, y1 = min(x0 + tile, width), min(y0 + tile, height)
            has_defect = contains_box_center(record["boxes"], x0, y0, x1, y1)
            if split == "train" and label == 1 and not has_defect:
                continue
            rows.append(
                {
                    "split": split,
                    "image": str(image_path.resolve()),
                    "label": label,
                    "source_id": source_id,
                    "source_class": source_class,
                    "pair_key": record["pair_key"],
                    "template": record["template"],
                    "train_rank": train_rank,
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "prototype_key": f"{record['template']}:{x0}:{y0}:{x1-x0}:{y1-y0}",
                    "contains_defect": int(has_defect),
                    "synthetic_ok": synthetic_ok,
                }
            )


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(f"{args.output_root} is not empty; pass --force to rebuild")
    args.output_root.mkdir(parents=True, exist_ok=True)
    normal_root = args.output_root / "normal_images"
    normal_root.mkdir(parents=True, exist_ok=True)

    records = [
        parse_record(path, args.dataset_root)
        for path in sorted((args.dataset_root / "Annotations").rglob("*.xml"))
    ]
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["pair_key"]].append(record)
    complete = {
        key: sorted(items, key=lambda item: item["class"])
        for key, items in grouped.items()
        if len(items) == len(CLASSES) and {item["class"] for item in items} == set(CLASSES)
    }
    excluded = sorted(set(grouped) - set(complete))
    assignments, train_ranks = split_pair_keys(
        list(complete), args.train_groups, args.val_groups, args.test_groups, args.seed
    )

    tile_rows: list[dict] = []
    image_rows: list[dict] = []
    quality = {}
    for pair_key in sorted(assignments):
        items = complete[pair_key]
        split = assignments[pair_key]
        representative = items[0]
        normal_image, normal_metrics = robust_median_image(items)
        split_normal_dir = normal_root / split
        split_normal_dir.mkdir(parents=True, exist_ok=True)
        normal_path = split_normal_dir / f"{pair_key}_ok.jpg"
        normal_image.save(normal_path, quality=args.jpeg_quality, subsampling=0)
        quality[pair_key] = {"split": split, **normal_metrics}
        rank = train_ranks.get(pair_key, 0)
        ok_source = f"ok:{pair_key}"
        image_rows.append(
            {
                "split": split,
                "image": str(normal_path.resolve()),
                "label": 0,
                "source_id": ok_source,
                "source_class": "ok",
                "pair_key": pair_key,
                "template": representative["template"],
                "train_rank": rank,
                "synthetic_ok": 1,
            }
        )
        add_tile_rows(
            tile_rows, representative, split, normal_path, 0, ok_source, "ok",
            args.tile_size, args.overlap, rank, 1,
        )
        for record in items:
            ng_source = f"ng:{record['stem']}"
            image_rows.append(
                {
                    "split": split,
                    "image": str(record["image"].resolve()),
                    "label": 1,
                    "source_id": ng_source,
                    "source_class": record["class"],
                    "pair_key": pair_key,
                    "template": record["template"],
                    "train_rank": rank,
                    "synthetic_ok": 0,
                }
            )
            add_tile_rows(
                tile_rows, record, split, record["image"], 1, ng_source,
                record["class"], args.tile_size, args.overlap, rank, 0,
            )

    for filename, rows in (("tiles.csv", tile_rows), ("images.csv", image_rows)):
        with (args.output_root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    tile_counts = Counter((row["split"], int(row["label"])) for row in tile_rows)
    image_counts = Counter((row["split"], int(row["label"])) for row in image_rows)
    summary = {
        "complete_pair_groups": len(complete),
        "used_pair_groups": len(assignments),
        "excluded_incomplete_pair_groups": excluded,
        "split_pair_groups": dict(Counter(assignments.values())),
        "image_counts": {f"{s}_{y}": n for (s, y), n in sorted(image_counts.items())},
        "tile_counts": {f"{s}_{y}": n for (s, y), n in sorted(tile_counts.items())},
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "seed": args.seed,
        "normal_generation": "channel-wise median across six aligned defect classes",
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "normal_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

