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
    parser.add_argument(
        "--final-test",
        type=Path,
        default=Path("artifacts/final_test/yolo11n_1280_default_final_test/test_summary.json"),
    )
    parser.add_argument(
        "--anomaly-summary",
        type=Path,
        default=Path("artifacts/anomaly_baselines/supersimplenet_box_tiles_r18_256/experiment_summary.json"),
    )
    parser.add_argument(
        "--anomaly-thresholds",
        type=Path,
        default=Path("artifacts/anomaly_baselines/supersimplenet_box_tiles_r18_256/threshold_calibration.json"),
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
    if args.final_test.exists():
        final = json.loads(args.final_test.read_text(encoding="utf-8"))
        metrics = final["metrics"]
        latency = final["latency"]
        lines.extend(
            [
                "",
                "## Frozen-model test result",
                "",
                "After selecting E06a from validation results, its checkpoint was evaluated once on the untouched PCB-template test split (groups 11 and 12; 120 images and 606 boxes).",
                "",
                "| Model | Precision | Recall | mAP50 | mAP50-95 | Params (M) | Size (MB) | Mean latency (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                f"| YOLO11n, 1280 | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['map50']:.3f} | {metrics['map50_95']:.3f} | {final['parameter_count'] / 1_000_000:.2f} | {final['model_size_mb']:.2f} | {latency['mean_ms']:.1f} |",
                "",
                "Test per-class mAP50-95: "
                + ", ".join(
                    f"{name.replace('_', ' ')} {value:.3f}"
                    for name, value in zip(CLASSES, metrics["per_class_map50_95"])
                )
                + ".",
                "",
                "## Decisions",
                "",
                "- Use E06a as the current detector baseline. Compared with 1024 input, it improves validation recall from 0.851 to 0.910 with the same 2.59M-parameter model.",
                "- Reject 640 input: mAP50-95 falls from 0.473 to 0.317 for only about a 9% reduction in measured end-to-end latency.",
                "- Reject the tuned-augmentation run as the primary checkpoint: its 0.001 mAP50-95 gain is outweighed by a 0.060 recall drop.",
                "- Reject the custom P2 head in its present form. Only part of the pretrained detection head transfers, and accuracy regresses.",
                "- Do not freeze ten modules in the 5-shot setting. It reduces mAP50-95 from 0.153 to 0.118.",
                "- Ten images per class materially outperform five (0.303 versus 0.153 mAP50-95), but the full training set remains substantially better.",
                "",
                "Latency numbers include preprocessing and postprocessing in the PyTorch runner on an RTX 3090 and should not be interpreted as edge-device TensorRT latency.",
            ]
        )
    if args.anomaly_summary.exists():
        anomaly = json.loads(args.anomaly_summary.read_text(encoding="utf-8"))
        val = anomaly["validation"]
        test = anomaly["test"]
        lines.extend(
            [
                "",
                "## Adapted SuperSimpleNet result",
                "",
                "This weak-supervision baseline treats box-free tiles from anomalous images as local pseudo-normal samples and converts VOC boxes to coarse rectangular masks. It is not comparable to a standard normal-only anomaly-detection protocol.",
                "",
                "| Split | Tile AUROC | Tile AP | Pixel AUROC@64 | Pixel AP@64 | Best pixel F1@64 |",
                "|---|---:|---:|---:|---:|---:|",
                f"| Validation | {val['tile_auroc']:.3f} | {val['tile_ap']:.3f} | {val['pixel_auroc_64']:.3f} | {val['pixel_ap_64']:.3f} | {val['pixel_best_f1_64']:.3f} |",
                f"| Test | {test['tile_auroc']:.3f} | {test['tile_ap']:.3f} | {test['pixel_auroc_64']:.3f} | {test['pixel_ap_64']:.3f} | {test['pixel_best_f1_64']:.3f} |",
                "",
                f"The ResNet-18 variant has {anomaly['parameter_count'] / 1_000_000:.2f}M parameters and measured {anomaly['latency_mean_ms_per_tile']:.2f}ms per 256-pixel tile on the RTX 3090. The test heat-map threshold was not tuned on test data; the reported best-test F1 is descriptive only.",
            ]
        )
        if args.anomaly_thresholds.exists():
            calibrated = json.loads(args.anomaly_thresholds.read_text(encoding="utf-8"))
            fixed = calibrated["test_at_frozen_validation_thresholds"]
            lines.extend(
                [
                    "",
                    f"Validation-only calibration selects tile threshold {calibrated['tile_threshold']:.2f} and pixel threshold {calibrated['pixel_threshold']:.2f}. At these frozen thresholds, test tile F1 is {fixed['tile']['f1']:.3f} and test pixel F1@64 is {fixed['pixel_64']['f1']:.3f}.",
                ]
            )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv_output} and {args.markdown_output}")


if __name__ == "__main__":
    main()
