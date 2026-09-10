#!/usr/bin/env python3
"""Compare deployable OK/NG thresholds using saved image-level predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aggregation", choices=("max", "top2", "top3"), default="top2")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        np.asarray([float(row["score"]) for row in rows], dtype=np.float64),
    )


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "recall_ng": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity_ok": float(tn / (tn + fp)) if tn + fp else 0.0,
        "false_positive_rate_ok": float(fp / (tn + fp)) if tn + fp else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    ranked = [(f1_score(labels, scores >= threshold), threshold) for threshold in candidates]
    return float(max(ranked, key=lambda item: (item[0], item[1]))[1])


def main() -> None:
    args = parse_args()
    val_labels, val_scores = read_predictions(
        args.run_dir / f"validation_predictions_{args.aggregation}.csv"
    )
    test_labels, test_scores = read_predictions(
        args.run_dir / f"test_predictions_{args.aggregation}.csv"
    )
    normal_scores = val_scores[val_labels == 0]
    ng_scores = val_scores[val_labels == 1]
    thresholds = {
        "f1_optimal": best_f1_threshold(val_labels, val_scores),
        "zero_validation_fn": float(ng_scores.min()),
        "zero_validation_fp": float(np.nextafter(normal_scores.max(), np.inf)),
    }
    result = {
        "source_run": str(args.run_dir),
        "aggregation": args.aggregation,
        "warning": "Calibration uses synthetic OK; repeat with real production OK before deployment.",
        "operating_points": {
            name: {
                "validation": metrics(val_labels, val_scores, threshold),
                "test": metrics(test_labels, test_scores, threshold),
            }
            for name, threshold in thresholds.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
