#!/usr/bin/env python3
"""Render qualitative SuperSimpleNet heatmaps for held-out PCB tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from run_supersimplenet_pcb import TileDataset, read_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--examples-per-label", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.official_repo.resolve()))
    from model.supersimplenet import SuperSimpleNet

    rows = read_rows(args.manifest)["test"]
    rng = np.random.default_rng(args.seed)
    selected = []
    for label in (0, 1):
        indices = np.array([index for index, row in enumerate(rows) if row["label"] == label])
        selected.extend(rng.choice(indices, size=args.examples_per_label, replace=False).tolist())
    dataset = TileDataset(rows, args.image_size)
    loader = DataLoader(Subset(dataset, selected), batch_size=1, shuffle=False)
    config = {
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperSimpleNet((args.image_size, args.image_size), config).to(device)
    model.load_model(args.checkpoint)
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    columns = 4
    figure, axes = plt.subplots(3, columns, figsize=(14, 10), constrained_layout=True)
    for axis, batch, row_index in zip(axes.flat, loader, selected):
        image = batch["image"].to(device)
        with torch.no_grad():
            anomaly_map, score = model(image)
        rgb = image[0].permute(1, 2, 0).cpu().numpy() * std + mean
        rgb = np.clip(rgb, 0, 1)
        heat = torch.sigmoid(anomaly_map)[0, 0].cpu().numpy()
        mask = batch["mask"][0].numpy()
        axis.imshow(rgb)
        axis.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        if mask.max() > 0:
            axis.contour(mask, levels=[0.5], colors="white", linewidths=1)
        axis.set_title(
            f"GT={rows[row_index]['label']} score={torch.sigmoid(score)[0].item():.3f}"
        )
        axis.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
