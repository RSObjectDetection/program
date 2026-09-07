#!/usr/bin/env python3
"""Run an adapted SuperSimpleNet baseline on box-mask PCB tiles.

This is a weak/coarse-mask baseline: normal tiles are sampled from annotated
images only when they do not intersect any labelled defect box.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    return parser.parse_args()


class TileDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int) -> None:
        self.rows = rows
        self.image_size = image_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row["image"]) as opened:
            image = opened.convert("RGB")
            image = TF.resize(image, [self.image_size, self.image_size])
            image = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        with Image.open(row["mask"]) as opened:
            mask = opened.convert("L")
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
            mask = (TF.to_tensor(mask).squeeze(0) > 0.5).float()
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(float(row["label"])),
        }


def read_rows(path: Path) -> dict[str, list[dict]]:
    result = {"train": [], "val": [], "test": []}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["label"] = int(row["label"])
            result[row["split"]].append(row)
    return result


def stratified_limit(rows: list[dict], maximum: int, seed: int) -> list[dict]:
    """Deterministically cap a split while retaining both tile labels."""
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    rng = np.random.default_rng(seed)
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[row["label"]].append(row)
    selected = []
    for label_rows in by_label.values():
        count = max(1, round(maximum * len(label_rows) / len(rows)))
        indices = rng.choice(len(label_rows), size=min(count, len(label_rows)), replace=False)
        selected.extend(label_rows[index] for index in indices)
    rng.shuffle(selected)
    return selected[:maximum]


def focal_loss(inputs, targets, gamma: float = 4.0):
    ce = F.binary_cross_entropy(inputs.float(), targets.float(), reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    return ce * ((1 - p_t) ** gamma)


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    model.eval()
    labels, scores = [], []
    pixel_targets, pixel_scores = [], []
    for batch in loader:
        images = batch["image"].to(device)
        anomaly_map, anomaly_score = model(images)
        labels.append(batch["label"].numpy())
        scores.append(torch.sigmoid(anomaly_score).flatten().cpu().numpy())
        reduced_map = F.interpolate(
            torch.sigmoid(anomaly_map), size=(64, 64), mode="bilinear", align_corners=False
        )
        reduced_mask = F.interpolate(
            batch["mask"].unsqueeze(1), size=(64, 64), mode="nearest"
        )
        pixel_scores.append(reduced_map.flatten().cpu().numpy())
        pixel_targets.append(reduced_mask.flatten().numpy().astype(np.uint8))
    labels_np = np.concatenate(labels)
    scores_np = np.concatenate(scores)
    pixel_targets_np = np.concatenate(pixel_targets)
    pixel_scores_np = np.concatenate(pixel_scores)
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1, best_threshold = 0.0, 0.5
    for threshold in thresholds:
        prediction = pixel_scores_np >= threshold
        tp = np.logical_and(prediction, pixel_targets_np == 1).sum()
        fp = np.logical_and(prediction, pixel_targets_np == 0).sum()
        fn = np.logical_and(~prediction, pixel_targets_np == 1).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(threshold)
    return {
        "tile_auroc": float(roc_auc_score(labels_np, scores_np)),
        "tile_ap": float(average_precision_score(labels_np, scores_np)),
        "pixel_auroc_64": float(roc_auc_score(pixel_targets_np, pixel_scores_np)),
        "pixel_ap_64": float(average_precision_score(pixel_targets_np, pixel_scores_np)),
        "pixel_best_f1_64": best_f1,
        "pixel_best_threshold": best_threshold,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sys.path.insert(0, str(args.official_repo.resolve()))
    from model.supersimplenet import SuperSimpleNet

    rows = read_rows(args.manifest)
    rows["train"] = stratified_limit(rows["train"], args.max_train_rows, args.seed)
    rows["val"] = stratified_limit(rows["val"], args.max_eval_rows, args.seed + 1)
    rows["test"] = stratified_limit(rows["test"], args.max_eval_rows, args.seed + 2)
    train_data = TileDataset(rows["train"], args.image_size)
    val_data = TileDataset(rows["val"], args.image_size)
    counts = Counter(row["label"] for row in rows["train"])
    weights = [1.0 / counts[row["label"]] for row in rows["train"]]
    positive_count = counts[1]
    sampler = WeightedRandomSampler(
        weights, num_samples=positive_count * 2, replacement=True
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    config = {
        "backbone": args.backbone,
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
        "epochs": args.epochs,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": False,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperSimpleNet((args.image_size, args.image_size), config).to(device)
    optimizer, scheduler = model.get_optimizers()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ap, stale = -1.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            mask = F.interpolate(mask.unsqueeze(1), size=(model.fh, model.fw), mode="nearest")
            label = batch["label"].to(device, non_blocking=True)
            anomaly_map, score, target_mask, target_label = model(images, mask, label)
            seg_focal = focal_loss(torch.sigmoid(anomaly_map), target_mask).mean()
            normal_scores = anomaly_map[target_mask == 0]
            bad_scores = anomaly_map[target_mask > 0]
            zero = anomaly_map.sum() * 0.0
            good_loss = (
                torch.clip(normal_scores + 0.5, min=0).mean()
                if normal_scores.numel()
                else zero
            )
            bad_loss = (
                torch.clip(-bad_scores + 0.5, min=0).mean()
                if bad_scores.numel()
                else zero
            )
            cls_loss = focal_loss(torch.sigmoid(score), target_label).mean()
            loss = seg_focal + good_loss + bad_loss + cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
        scheduler.step()
        record = {"epoch": epoch, "train_loss": running / len(train_loader)}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device)
            record.update(metrics)
            print(json.dumps(record))
            if metrics["pixel_ap_64"] > best_ap:
                best_ap, stale = metrics["pixel_ap_64"], 0
                model.save_model(args.output_dir)
                (args.output_dir / "best_validation.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
            else:
                stale += 1
                if stale >= args.patience:
                    history.append(record)
                    break
        history.append(record)

    model.load_model(args.output_dir / "weights.pt")
    final_metrics = evaluate(model, val_loader, device)
    test_loader = DataLoader(
        TileDataset(rows["test"], args.image_size),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    final_test_metrics = evaluate(model, test_loader, device)
    sample = next(iter(val_loader))["image"][:1].to(device)
    for _ in range(10):
        model(sample)
    if device == "cuda":
        torch.cuda.synchronize()
    timings = []
    for _ in range(args.benchmark_iters):
        started = time.perf_counter()
        model(sample)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    summary = {
        "name": f"supersimplenet_box_tiles_r18_{args.image_size}",
        "data_assumption": "normal tiles are box-free regions from anomalous images",
        "mask_assumption": "VOC boxes converted to coarse rectangular masks",
        "backbone": args.backbone,
        "image_size": args.image_size,
        "train_tile_counts": dict(counts),
        "validation": final_metrics,
        "test": final_test_metrics,
        "latency_mean_ms_per_tile": float(np.mean(timings)),
        "latency_p95_ms_per_tile": float(np.percentile(timings, 95)),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epochs_completed": history[-1]["epoch"],
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
