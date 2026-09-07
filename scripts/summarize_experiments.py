#!/usr/bin/env python3
"""Aggregate tracked experiment summaries into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/experiments")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("artifacts/EXPERIMENT_RESULTS.csv")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("docs/EXPERIMENT_RESULTS.md")
    )
    return parser.parse_args()


def row_for(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["test_metrics"]
    validation = data["validation_results_dict"]
    row = {
        "experiment": data["name"],
        "split": data["eval_split"],
        "imgsz": data["imgsz"],
        "parameters_m": data["parameter_count"] / 1_000_000,
        "model_mb": data["best_model_size_mb"],
        "precision": validation["metrics/precision(B)"],
        "recall": validation["metrics/recall(B)"],
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "latency_mean_ms": data["latency"]["mean_ms"],
        "latency_p95_ms": data["latency"]["p95_ms"],
        "fps": data["latency"]["fps_from_mean"],
        "peak_gpu_memory_mb": data["latency"]["peak_gpu_memory_mb"],
    }
    for name, value in zip(CLASSES, metrics["per_class_map50_95"]):
        row[f"map_{name}"] = value
    return row


def main() -> None:
    args = parse_args()
    paths = sorted(args.artifact_root.glob("*/experiment_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No summaries found under {args.artifact_root}")
    rows = [row_for(path) for path in paths]

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Experiment results",
        "",
        "All values below are produced by the reproducible runner. `val` is used for model selection; the held-out `test` set must only be evaluated after configuration freeze.",
        "",
        "| Experiment | Input | mAP50 | mAP50-95 | Recall | Params (M) | Mean latency (ms) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['imgsz']} | {row['map50']:.3f} | "
            f"{row['map50_95']:.3f} | {row['recall']:.3f} | "
            f"{row['parameters_m']:.2f} | {row['latency_mean_ms']:.1f} | "
            f"{row['fps']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Per-class mAP50-95",
            "",
            "| Experiment | " + " | ".join(CLASSES) + " |",
            "|---|" + "---:|" * len(CLASSES),
        ]
    )
    for row in rows:
        values = " | ".join(f"{row[f'map_{name}']:.3f}" for name in CLASSES)
        lines.append(f"| {row['experiment']} | {values} |")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv_output} and {args.markdown_output}")


if __name__ == "__main__":
    main()
