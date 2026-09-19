#!/usr/bin/env python3
"""Run cumulative class and cumulative data CPG-SSN experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
PERCENTAGES = list(range(10, 101, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=("horizontal", "vertical"),
        default=("horizontal", "vertical"),
    )
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--incremental-lr-scale", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def audit_manifest(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    image_rows = {}
    for row in rows:
        image_rows[row["source_id"]] = row

    pair_splits: defaultdict[str, set[str]] = defaultdict(set)
    class_counts: defaultdict[str, Counter] = defaultdict(Counter)
    rank_pairs: defaultdict[int, set[str]] = defaultdict(set)
    for row in image_rows.values():
        pair_splits[row["pair_key"]].add(row["split"])
        class_counts[row["split"]][row["source_class"]] += 1
        if row["split"] == "train":
            rank_pairs[int(row["train_rank"])].add(row["pair_key"])

    leaking_pairs = {
        key: sorted(splits) for key, splits in pair_splits.items() if len(splits) != 1
    }
    invalid_ranks = {
        str(rank): sorted(pairs) for rank, pairs in rank_pairs.items() if len(pairs) != 1
    }
    expected_train_ranks = list(range(1, 81))
    actual_train_ranks = sorted(rank_pairs)
    expected_classes = {"ok", *CLASSES}
    class_errors = {
        split: dict(counts)
        for split, counts in class_counts.items()
        if set(counts) != expected_classes or len(set(counts.values())) != 1
    }
    audit = {
        "manifest": str(path.resolve()),
        "images": len(image_rows),
        "pair_groups": len(pair_splits),
        "image_counts_by_split_and_class": {
            split: dict(sorted(counts.items())) for split, counts in sorted(class_counts.items())
        },
        "pair_split_leaks": leaking_pairs,
        "invalid_train_rank_mappings": invalid_ranks,
        "train_ranks": actual_train_ranks,
        "valid": not leaking_pairs
        and not invalid_ranks
        and not class_errors
        and actual_train_ranks == expected_train_ranks,
    }
    if not audit["valid"]:
        audit["class_distribution_errors"] = class_errors
        raise ValueError(f"Dataset audit failed: {json.dumps(audit, ensure_ascii=False)}")
    return audit


def stage_command(
    args: argparse.Namespace,
    name: str,
    output_dir: Path,
    classes: list[str],
    shots: int,
    initial_weights: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_cpg_ssn.py")),
        "--manifest",
        str(args.manifest),
        "--official-repo",
        str(args.official_repo),
        "--output-dir",
        str(output_dir),
        "--name",
        name,
        "--backbone",
        "resnet18",
        "--image-size",
        "256",
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--shots-per-class",
        str(shots),
        "--max-train-samples",
        str(args.max_train_samples),
        "--eval-every",
        "3",
        "--patience",
        "4",
        "--benchmark-iters",
        str(args.benchmark_iters),
        "--prototype",
        "--glass",
        "--include-classes",
        *classes,
    ]
    if initial_weights is not None:
        command.extend(
            [
                "--initial-weights",
                str(initial_weights),
                "--learning-rate-scale",
                str(args.incremental_lr_scale),
            ]
        )
    return command


def run_stage(
    args: argparse.Namespace,
    track: str,
    stage: str,
    classes: list[str],
    shots: int,
    previous_weights: Path | None,
) -> Path:
    output_dir = args.output_root / track / f"seed_{args.seed}" / stage
    summary_path = output_dir / "experiment_summary.json"
    if summary_path.is_file() and not args.force:
        print(f"SKIP completed {track}/{stage}", flush=True)
        return output_dir / "weights.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = stage_command(
        args,
        f"{track}_{stage}_seed{args.seed}",
        output_dir,
        classes,
        shots,
        previous_weights,
    )
    (output_dir / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    print(f"START {track}/{stage}: classes={classes}, shots={shots}", flush=True)
    started = time.time()
    with (output_dir / "run.log").open("a", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metric = summary["test"]["top2"]
    print(
        f"DONE {track}/{stage} in {(time.time() - started) / 60:.1f} min: "
        f"F1={metric['f1']:.4f}, recall={metric['recall_ng']:.4f}, "
        f"specificity={metric['specificity_ok']:.4f}, FP={metric['confusion']['fp']}, "
        f"FN={metric['confusion']['fn']}",
        flush=True,
    )
    return output_dir / "weights.pt"


def collect_results(output_root: Path, seed: int) -> None:
    records = []
    for track in ("horizontal", "vertical"):
        seed_root = output_root / track / f"seed_{seed}"
        if not seed_root.is_dir():
            continue
        for summary_path in sorted(seed_root.glob("*/experiment_summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metric = summary["test"]["top2"]
            record = {
                "track": track,
                "stage": summary_path.parent.name,
                "seed": seed,
                "shots_per_class": summary["shots_per_class"],
                "classes": ";".join(summary["include_classes"]),
                "threshold": metric["threshold"],
                "auroc": metric["auroc"],
                "auprc": metric["auprc"],
                "f1": metric["f1"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "recall_ng": metric["recall_ng"],
                "specificity_ok": metric["specificity_ok"],
                **metric["confusion"],
            }
            for class_name in CLASSES:
                record[f"recall_{class_name}"] = metric["per_defect_recall"].get(
                    class_name, ""
                )
            records.append(record)
    if not records:
        return
    result_path = output_root / f"iterative_summary_seed_{seed}.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    horizontal = [row for row in records if row["track"] == "horizontal"]
    maxima = {class_name: 0.0 for class_name in CLASSES}
    forgetting = []
    for row in sorted(horizontal, key=lambda item: item["stage"]):
        output = {"stage": row["stage"], "seed": seed}
        values = []
        for class_name in CLASSES:
            value = row[f"recall_{class_name}"]
            if value == "":
                output[f"forgetting_{class_name}"] = ""
                continue
            value = float(value)
            output[f"forgetting_{class_name}"] = max(0.0, maxima[class_name] - value)
            maxima[class_name] = max(maxima[class_name], value)
            values.append(output[f"forgetting_{class_name}"])
        output["mean_forgetting"] = sum(values) / len(values)
        output["max_forgetting"] = max(values)
        forgetting.append(output)
    if forgetting:
        path = output_root / f"forgetting_seed_{seed}.csv"
        fields = ["stage", "seed", *[f"forgetting_{name}" for name in CLASSES], "mean_forgetting", "max_forgetting"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(forgetting)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_manifest(args.manifest)
    (args.output_root / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plan = {
        "seed": args.seed,
        "class_order": CLASSES,
        "percentages": PERCENTAGES,
        "train_groups": 80,
        "tracks": args.tracks,
        "epochs_per_stage": args.epochs,
        "max_train_samples_per_epoch": args.max_train_samples,
        "incremental_learning_rate_scale": args.incremental_lr_scale,
        "normal_policy": "training-split synthetic OK only; validation and test are never used",
    }
    (args.output_root / f"plan_seed_{args.seed}.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if "horizontal" in args.tracks:
        previous = None
        for index, class_name in enumerate(CLASSES, start=1):
            stage = f"H{index:02d}_{class_name}"
            previous = run_stage(
                args, "horizontal", stage, CLASSES[:index], 0, previous
            )
            collect_results(args.output_root, args.seed)

    if "vertical" in args.tracks:
        previous = None
        for percentage in PERCENTAGES:
            shots = 80 * percentage // 100
            stage = f"V{percentage:03d}"
            previous = run_stage(
                args, "vertical", stage, CLASSES, shots, previous
            )
            collect_results(args.output_root, args.seed)

    collect_results(args.output_root, args.seed)
    print("All requested iterative stages completed", flush=True)


if __name__ == "__main__":
    main()
