#!/usr/bin/env python3
"""Calibrate SuperSimpleNet tile/pixel thresholds on validation and freeze for test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from run_supersimplenet_pcb import TileDataset, read_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def config() -> dict:
    return {
        "backbone": "resnet18",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": False,
        "noise_std": 0.015,
        "perlin_thr": 0.6,
        "epochs": 15,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": False,
    }


@torch.no_grad()
def predictions(model, rows: list[dict], image_size: int, batch: int, workers: int, device: str):
    loader = DataLoader(
        TileDataset(rows, image_size), batch_size=batch, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    tile_y, tile_score, pixel_y, pixel_score = [], [], [], []
    for item in loader:
        anomaly_map, anomaly_score = model(item["image"].to(device, non_blocking=True))
        tile_y.append(item["label"].numpy().astype(np.uint8))
        tile_score.append(torch.sigmoid(anomaly_score).cpu().numpy())
        reduced_map = F.interpolate(
            torch.sigmoid(anomaly_map), size=(64, 64), mode="bilinear", align_corners=False
        )
        reduced_mask = F.interpolate(
            item["mask"].unsqueeze(1), size=(64, 64), mode="nearest"
        )
        pixel_y.append(reduced_mask.numpy().reshape(-1).astype(np.uint8))
        pixel_score.append(reduced_map.cpu().numpy().reshape(-1))
    return tuple(np.concatenate(values) for values in (tile_y, tile_score, pixel_y, pixel_score))


def f1_at_threshold(target: np.ndarray, score: np.ndarray, threshold: float) -> float:
    prediction = score >= threshold
    tp = np.logical_and(prediction, target == 1).sum()
    fp = np.logical_and(prediction, target == 0).sum()
    fn = np.logical_and(~prediction, target == 1).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def select_threshold(target: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.01, 0.99, 99)
    values = [f1_at_threshold(target, score, threshold) for threshold in candidates]
    index = int(np.argmax(values))
    return float(candidates[index]), float(values[index])


def metrics(target: np.ndarray, score: np.ndarray) -> dict:
    return {
        "positive_prevalence": float(target.mean()),
        "auroc": float(roc_auc_score(target, score)),
        "average_precision": float(average_precision_score(target, score)),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.official_repo.resolve()))
    from model.supersimplenet import SuperSimpleNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperSimpleNet((args.image_size, args.image_size), config()).to(device)
    model.load_model(args.checkpoint)
    model.eval()
    rows = read_rows(args.manifest)
    val = predictions(model, rows["val"], args.image_size, args.batch, args.workers, device)
    test = predictions(model, rows["test"], args.image_size, args.batch, args.workers, device)
    tile_threshold, val_tile_f1 = select_threshold(val[0], val[1])
    pixel_threshold, val_pixel_f1 = select_threshold(val[2], val[3])
    result = {
        "threshold_source": "validation only",
        "tile_threshold": tile_threshold,
        "pixel_threshold": pixel_threshold,
        "validation": {
            "tile": {**metrics(val[0], val[1]), "f1": val_tile_f1},
            "pixel_64": {**metrics(val[2], val[3]), "f1": val_pixel_f1},
        },
        "test_at_frozen_validation_thresholds": {
            "tile": {**metrics(test[0], test[1]), "f1": f1_at_threshold(test[0], test[1], tile_threshold)},
            "pixel_64": {**metrics(test[2], test[3]), "f1": f1_at_threshold(test[2], test[3], pixel_threshold)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
