#!/usr/bin/env python3
"""Collect lightweight S01-S09 artifacts while leaving model weights in /hy-tmp."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


RUNS = {
    "S01": "S01_baseline",
    "S02": "S02_prototype",
    "S03": "S03_glass",
    "S04": "S04_cpg_ssn",
    "S06": "S06_overlap64",
    "S07-30": "S07_shot30",
    "S07-50": "S07_shot50",
    "S07-100": "S07_shot100",
    "S08": "S08_resnet18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("/hy-tmp/runs"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    table = []
    for experiment, run_name in RUNS.items():
        run_dir = args.runs_root / run_name
        summary_path = run_dir / "experiment_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        aggregation = summary["selected_aggregation"]
        test = summary["test"][aggregation]
        table.append({
            "experiment": experiment,
            "run": run_name,
            "aggregation": aggregation,
            "auroc": test["auroc"],
            "auprc": test["auprc"],
            "f1": test["f1"],
            "recall_ng": test["recall_ng"],
            "specificity_ok": test["specificity_ok"],
            "fp": test["confusion"]["fp"],
            "fn": test["confusion"]["fn"],
            "latency_ms_per_tile": summary["latency_mean_ms_per_tile"],
            "parameters": summary["parameter_count"],
        })
        destination = args.output_root / run_name
        destination.mkdir(exist_ok=True)
        for source in run_dir.iterdir():
            if source.is_file() and source.name != "weights.pt":
                shutil.copy2(source, destination / source.name)

    with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if table:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
