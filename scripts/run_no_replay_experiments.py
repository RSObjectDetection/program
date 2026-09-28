#!/usr/bin/env python3
"""Run strict no-replay class- and data-incremental CPG-SSN experiments.

Historical NG images are never loaded by a later stage. Every stage reuses the
same train-split OK reference set, starts from the preceding stage's trainable
weights, and evaluates on a fixed held-out split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from run_iterative_experiments import audit_manifest


CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
PERCENTAGES = list(range(10, 101, 10))
TRAIN_IMAGES_PER_CLASS = 80
CHUNK_SIZE = 8


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
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write and validate the no-replay data plan without launching training.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def image_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return list({row["source_id"]: row for row in rows}.values())


def digest_source_ids(rows: list[dict]) -> str:
    payload = "\n".join(sorted(row["source_id"] for row in rows)).encode()
    return hashlib.sha256(payload).hexdigest()


def build_no_replay_audit(path: Path) -> dict:
    rows = image_rows(path)
    train = [row for row in rows if row["split"] == "train"]
    ok_rows = [row for row in train if row["source_class"] == "ok"]
    horizontal = []
    horizontal_seen: set[str] = set()
    for index, class_name in enumerate(CLASSES, start=1):
        stage_rows = [row for row in train if row["source_class"] == class_name]
        source_ids = {row["source_id"] for row in stage_rows}
        overlap = sorted(horizontal_seen & source_ids)
        if len(stage_rows) != TRAIN_IMAGES_PER_CLASS or overlap:
            raise ValueError(
                f"Invalid horizontal stage H{index:02d}: "
                f"images={len(stage_rows)}, historical_overlap={overlap[:3]}"
            )
        horizontal_seen.update(source_ids)
        horizontal.append(
            {
                "stage": f"H{index:02d}_{class_name}",
                "train_classes": [class_name],
                "eval_classes": CLASSES[:index],
                "new_ng_images": len(stage_rows),
                "new_ng_source_digest": digest_source_ids(stage_rows),
                "historical_ng_overlap": 0,
            }
        )

    vertical = []
    vertical_seen: set[str] = set()
    for index, percentage in enumerate(PERCENTAGES, start=1):
        start = (index - 1) * CHUNK_SIZE + 1
        end = index * CHUNK_SIZE
        stage_rows = [
            row
            for row in train
            if row["source_class"] in CLASSES
            and start <= int(row["train_rank"]) <= end
        ]
        counts = Counter(row["source_class"] for row in stage_rows)
        source_ids = {row["source_id"] for row in stage_rows}
        overlap = sorted(vertical_seen & source_ids)
        if any(counts[name] != CHUNK_SIZE for name in CLASSES) or overlap:
            raise ValueError(
                f"Invalid vertical stage V{percentage:03d}: "
                f"counts={dict(counts)}, historical_overlap={overlap[:3]}"
            )
        vertical_seen.update(source_ids)
        vertical.append(
            {
                "stage": f"V{percentage:03d}",
                "train_rank_start": start,
                "train_rank_end": end,
                "new_ng_images_per_class": CHUNK_SIZE,
                "new_ng_images": len(stage_rows),
                "historical_ng_images_per_class": end,
                "new_ng_source_digest": digest_source_ids(stage_rows),
                "historical_ng_overlap": 0,
            }
        )

    if len(ok_rows) != TRAIN_IMAGES_PER_CLASS:
        raise ValueError(f"Expected 80 fixed train OK images, found {len(ok_rows)}")
    return {
        "protocol": "strict_no_replay_ng",
        "fixed_ok_images": len(ok_rows),
        "fixed_ok_source_digest": digest_source_ids(ok_rows),
        "horizontal": horizontal,
        "vertical": vertical,
        "valid": True,
    }


def stage_command(
    args: argparse.Namespace,
    name: str,
    output_dir: Path,
    train_classes: list[str],
    eval_classes: list[str],
    rank_start: int,
    rank_end: int,
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
        "--train-rank-start",
        str(rank_start),
        "--train-rank-end",
        str(rank_end),
        "--fixed-ok-train",
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
        "--train-classes",
        *train_classes,
        "--eval-classes",
        *eval_classes,
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
    train_classes: list[str],
    eval_classes: list[str],
    rank_start: int,
    rank_end: int,
    previous_weights: Path | None,
    stage_metadata: dict,
) -> Path:
    output_dir = args.output_root / track / f"seed_{args.seed}" / stage
    summary_path = output_dir / "experiment_summary.json"
    if summary_path.is_file() and not args.force:
        print(f"SKIP completed {track}/{stage}", flush=True)
        weights = output_dir / "weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Completed stage is missing weights: {weights}")
        return weights
    output_dir.mkdir(parents=True, exist_ok=True)
    command = stage_command(
        args,
        f"no_replay_{track}_{stage}_seed{args.seed}",
        output_dir,
        train_classes,
        eval_classes,
        rank_start,
        rank_end,
        previous_weights,
    )
    (output_dir / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    print(
        f"START {track}/{stage}: train={train_classes}, eval={eval_classes}, "
        f"ranks={rank_start}-{rank_end}, previous={previous_weights}",
        flush=True,
    )
    started = time.time()
    with (output_dir / "run.log").open("a", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "continual_protocol": "strict_no_replay_ng",
            "stage_metadata": stage_metadata,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metric = summary["test"]["top2"]
    print(
        f"DONE {track}/{stage} in {(time.time() - started) / 60:.1f} min: "
        f"F1={metric['f1']:.4f}, recall={metric['recall_ng']:.4f}, "
        f"specificity={metric['specificity_ok']:.4f}, "
        f"FP={metric['confusion']['fp']}, FN={metric['confusion']['fn']}",
        flush=True,
    )
    return output_dir / "weights.pt"


def prediction_records(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metrics_at_threshold(records: list[dict], threshold: float) -> dict:
    tn = fp = fn = tp = 0
    class_hits: dict[str, list[int]] = {}
    for row in records:
        label = int(row["label"])
        prediction = int(float(row["score"]) >= threshold)
        if label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1
        else:
            tp += 1
        if label == 1:
            class_hits.setdefault(row["source_class"], []).append(prediction)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": threshold,
        "f1": f1,
        "recall_ng": recall,
        "specificity_ok": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "per_defect_recall": {
            name: sum(values) / len(values) for name, values in class_hits.items()
        },
    }


def write_forgetting(records: list[dict], path: Path, metric_key: str) -> list[dict]:
    maxima = {name: 0.0 for name in CLASSES}
    output_rows = []
    for record in sorted(records, key=lambda item: item["stage"]):
        output = {"stage": record["stage"], "seed": record["seed"]}
        values = []
        recalls = record[metric_key]["per_defect_recall"]
        for class_name in CLASSES:
            if class_name not in recalls:
                output[f"forgetting_{class_name}"] = ""
                continue
            value = float(recalls[class_name])
            forgetting = max(0.0, maxima[class_name] - value)
            output[f"forgetting_{class_name}"] = forgetting
            maxima[class_name] = max(maxima[class_name], value)
            values.append(forgetting)
        output["mean_forgetting"] = sum(values) / len(values)
        output["max_forgetting"] = max(values)
        output_rows.append(output)
    fields = [
        "stage",
        "seed",
        *[f"forgetting_{name}" for name in CLASSES],
        "mean_forgetting",
        "max_forgetting",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def collect_results(output_root: Path, seed: int) -> None:
    stage_summaries: dict[str, list[tuple[Path, dict]]] = {}
    for track in ("horizontal", "vertical"):
        seed_root = output_root / track / f"seed_{seed}"
        items = []
        if seed_root.is_dir():
            for path in sorted(seed_root.glob("*/experiment_summary.json")):
                items.append((path, json.loads(path.read_text(encoding="utf-8"))))
        stage_summaries[track] = items

    records = []
    rich_records: dict[str, list[dict]] = {"horizontal": [], "vertical": []}
    for track, items in stage_summaries.items():
        if not items:
            continue
        fixed_threshold = items[0][1]["validation"]["top2"]["threshold"]
        for summary_path, summary in items:
            adaptive = summary["test"]["top2"]
            fixed = metrics_at_threshold(
                prediction_records(summary_path.parent / "test_predictions_top2.csv"),
                fixed_threshold,
            )
            stage = summary_path.parent.name
            rich = {
                "track": track,
                "stage": stage,
                "seed": seed,
                "adaptive": adaptive,
                "fixed": fixed,
            }
            rich_records[track].append(rich)
            metadata = summary.get("stage_metadata", {})
            record = {
                "track": track,
                "stage": stage,
                "seed": seed,
                "train_classes": ";".join(summary.get("train_classes", [])),
                "eval_classes": ";".join(summary.get("eval_classes", [])),
                "train_rank_start": summary.get("train_rank_start", ""),
                "train_rank_end": summary.get("train_rank_end", ""),
                "new_ng_images": metadata.get("new_ng_images", ""),
                "historical_ng_images_per_class": metadata.get(
                    "historical_ng_images_per_class", ""
                ),
                "adaptive_threshold": adaptive["threshold"],
                "adaptive_f1": adaptive["f1"],
                "adaptive_recall_ng": adaptive["recall_ng"],
                "adaptive_specificity_ok": adaptive["specificity_ok"],
                "adaptive_fp": adaptive["confusion"]["fp"],
                "adaptive_fn": adaptive["confusion"]["fn"],
                "fixed_threshold": fixed_threshold,
                "fixed_f1": fixed["f1"],
                "fixed_recall_ng": fixed["recall_ng"],
                "fixed_specificity_ok": fixed["specificity_ok"],
                "fixed_fp": fixed["confusion"]["fp"],
                "fixed_fn": fixed["confusion"]["fn"],
            }
            for class_name in CLASSES:
                record[f"adaptive_recall_{class_name}"] = adaptive[
                    "per_defect_recall"
                ].get(class_name, "")
                record[f"fixed_recall_{class_name}"] = fixed[
                    "per_defect_recall"
                ].get(class_name, "")
            records.append(record)

    if not records:
        return
    summary_path = output_root / f"no_replay_summary_seed_{seed}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    horizontal = rich_records["horizontal"]
    if horizontal:
        adaptive_forgetting = write_forgetting(
            horizontal,
            output_root / f"forgetting_adaptive_seed_{seed}.csv",
            "adaptive",
        )
        fixed_forgetting = write_forgetting(
            horizontal,
            output_root / f"forgetting_fixed_seed_{seed}.csv",
            "fixed",
        )
        final = horizontal[-1]
        continual = {
            "seed": seed,
            "completed_horizontal_stages": len(horizontal),
            "completed_vertical_stages": len(rich_records["vertical"]),
            "horizontal_final_stage": final["stage"],
            "horizontal_final_adaptive": final["adaptive"],
            "horizontal_final_fixed": final["fixed"],
            "horizontal_final_adaptive_forgetting": adaptive_forgetting[-1],
            "horizontal_final_fixed_forgetting": fixed_forgetting[-1],
        }
        (output_root / f"continual_summary_seed_{seed}.json").write_text(
            json.dumps(continual, indent=2), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_manifest(args.manifest)
    no_replay_audit = build_no_replay_audit(args.manifest)
    (args.output_root / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_root / "no_replay_audit.json").write_text(
        json.dumps(no_replay_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plan = {
        "seed": args.seed,
        "protocol": "strict_no_replay_ng",
        "class_order": CLASSES,
        "tracks": args.tracks,
        "epochs_per_stage": args.epochs,
        "max_train_samples_per_epoch": args.max_train_samples,
        "incremental_learning_rate_scale": args.incremental_lr_scale,
        "backbone_initialization": "ImageNet-pretrained ResNet-18; frozen",
        "stage_initialization": "previous stage trainable weights; S08 is not used",
        "normal_policy": "all 80 train-split synthetic OK images are a fixed reference set",
        "horizontal_policy": "only the newly introduced defect class is used for gradients",
        "vertical_policy": "only the new disjoint 8-image chunk per defect class is used",
        "validation_policy": "seen classes only for horizontal; all classes for vertical",
        "test_policy": "seen classes only for horizontal; fixed full test for vertical",
    }
    (args.output_root / f"plan_seed_{args.seed}.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.audit_only:
        print("No-replay dataset audit completed", flush=True)
        return

    if "horizontal" in args.tracks:
        previous = None
        for index, class_name in enumerate(CLASSES, start=1):
            stage = f"H{index:02d}_{class_name}"
            metadata = no_replay_audit["horizontal"][index - 1]
            previous = run_stage(
                args,
                "horizontal",
                stage,
                [class_name],
                CLASSES[:index],
                1,
                TRAIN_IMAGES_PER_CLASS,
                previous,
                metadata,
            )
            collect_results(args.output_root, args.seed)

    if "vertical" in args.tracks:
        previous = None
        for index, percentage in enumerate(PERCENTAGES, start=1):
            stage = f"V{percentage:03d}"
            metadata = no_replay_audit["vertical"][index - 1]
            previous = run_stage(
                args,
                "vertical",
                stage,
                CLASSES,
                CLASSES,
                metadata["train_rank_start"],
                metadata["train_rank_end"],
                previous,
                metadata,
            )
            collect_results(args.output_root, args.seed)

    collect_results(args.output_root, args.seed)
    print("All requested strict no-replay stages completed", flush=True)


if __name__ == "__main__":
    main()
