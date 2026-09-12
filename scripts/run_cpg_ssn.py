#!/usr/bin/env python3
"""Train and evaluate SuperSimpleNet/CPG-SSN for image-level OK/NG.

CPG-SSN keeps the SuperSimpleNet pipeline and adds two contained modules:
1. a component-position normal prototype residual, inspired by UniVAD;
2. projected hard feature perturbations, inspired by GLASS GAS.

The external SuperSimpleNet implementation is pinned and patched separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--shots-per-class", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--prototype", action="store_true")
    parser.add_argument("--prototype-size", type=int, default=8)
    parser.add_argument("--glass", action="store_true")
    parser.add_argument("--glass-step", type=float, default=0.001)
    parser.add_argument("--glass-steps", type=int, default=1)
    parser.add_argument("--glass-warmup", type=int, default=3)
    parser.add_argument("--benchmark-iters", type=int, default=50)
    return parser.parse_args()


def read_rows(path: Path, shots: int) -> dict[str, list[dict]]:
    splits = {"train": [], "val": [], "test": []}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["label"] = int(row["label"])
            row["train_rank"] = int(row["train_rank"])
            for key in ("x", "y", "w", "h", "contains_defect", "synthetic_ok"):
                row[key] = int(row[key])
            if row["split"] == "train" and shots > 0 and row["train_rank"] > shots:
                continue
            splits[row["split"]].append(row)
    return splits


class TileDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int, training: bool = False) -> None:
        self.rows = rows
        self.image_size = image_size
        self.training = training
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.jitter = ColorJitter(brightness=0.05, contrast=0.05, saturation=0.03)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row["image"]) as opened:
            image = opened.convert("RGB")
            image = image.crop(
                (row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"])
            )
        if self.training:
            image = self.jitter(image)
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        tensor = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        return {
            "image": tensor,
            "label": torch.tensor(float(row["label"])),
            "source_id": row["source_id"],
            "source_class": row["source_class"],
            "prototype_key": row["prototype_key"],
        }


def focal_loss(probabilities: torch.Tensor, targets: torch.Tensor, gamma: float = 4.0):
    ce = F.binary_cross_entropy(probabilities.float(), targets.float(), reduction="none")
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    return ce * ((1 - p_t) ** gamma)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PrototypeDiscriminator(nn.Module):
    """SuperSimpleNet discriminator with one normal-residual input channel."""

    def __init__(self, projection_dim: int, feature_h: int, feature_w: int, stop_grad=False):
        super().__init__()
        self.fh, self.fw = feature_h, feature_w
        self.stop_grad = stop_grad
        self.seg = nn.Sequential(
            nn.Conv2d(projection_dim, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1, bias=False),
        )
        self.residual_adapter = nn.Sequential(
            nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        )
        self.dec_head = nn.Sequential(
            nn.Conv2d(projection_dim + 2, 128, 5, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.map_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dec_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_score = nn.Linear(128 * 2 + 2, 1)
        self.apply(init_weights)

    def get_params(self):
        seg_params = self.seg.parameters()
        dec_params = (
            list(self.residual_adapter.parameters())
            + list(self.dec_head.parameters())
            + list(self.fc_score.parameters())
        )
        return seg_params, dec_params

    def forward(self, seg_features, cls_features, residual=None):
        anomaly_map = self.seg(seg_features)
        map_for_dec = anomaly_map.detach() if self.stop_grad else anomaly_map
        if residual is None:
            residual = torch.zeros_like(anomaly_map)
        residual = self.residual_adapter(residual)
        dec_out = self.dec_head(torch.cat((cls_features, map_for_dec, residual), dim=1))
        map_max = self.map_max_pool(anomaly_map)
        map_avg = self.map_avg_pool(anomaly_map)
        if self.stop_grad:
            map_max, map_avg = map_max.detach(), map_avg.detach()
        pooled = torch.cat(
            (
                self.dec_max_pool(dec_out),
                self.dec_avg_pool(dec_out),
                map_max,
                map_avg,
            ),
            dim=1,
        ).squeeze(dim=(2, 3))
        return anomaly_map, self.fc_score(pooled).squeeze(1)


class CPGSuperSimpleNet(nn.Module):
    def __init__(self, official_class, image_size: int, config: dict, use_prototype: bool):
        super().__init__()
        self.base = official_class((image_size, image_size), config)
        self.use_prototype = use_prototype
        if use_prototype:
            channels, height, width = self.base.feature_extractor.feature_dim
            self.base.discriminator = PrototypeDiscriminator(
                channels, height, width, config.get("stop_grad", False)
            )

    @property
    def fh(self):
        return self.base.fh

    @property
    def fw(self):
        return self.base.fw

    def residual_map(self, features, prototypes):
        if not self.use_prototype or prototypes is None:
            return None
        query = F.adaptive_avg_pool2d(features, prototypes.shape[-2:])
        residual = 1 - F.cosine_similarity(query, prototypes, dim=1).unsqueeze(1)
        return F.interpolate(
            residual, size=features.shape[-2:], mode="bilinear", align_corners=False
        ).clamp_(0, 2)

    def discriminate(self, seg_features, cls_features, residual):
        if self.use_prototype:
            return self.base.discriminator(seg_features, cls_features, residual)
        return self.base.discriminator(seg_features, cls_features)

    def glass_hard_mine(self, seg_features, cls_features, residual, synth_mask, steps, step_size):
        if steps <= 0 or not synth_mask.any():
            return seg_features, cls_features
        probe_seg = seg_features.detach()
        probe_cls = cls_features.detach()
        for _ in range(steps):
            probe_seg.requires_grad_(True)
            probe_cls.requires_grad_(True)
            _, probe_score = self.discriminate(probe_seg, probe_cls, residual)
            hard_loss = F.binary_cross_entropy_with_logits(
                probe_score, torch.ones_like(probe_score)
            )
            grad_seg, grad_cls = torch.autograd.grad(hard_loss, (probe_seg, probe_cls))
            grad_seg = F.normalize(grad_seg, dim=1)
            grad_cls = F.normalize(grad_cls, dim=1)
            with torch.no_grad():
                probe_seg = probe_seg + step_size * grad_seg * synth_mask
                probe_cls = probe_cls + step_size * grad_cls * synth_mask
        return (
            seg_features + (probe_seg - seg_features.detach()),
            cls_features + (probe_cls - cls_features.detach()),
        )

    def forward(
        self, images, mask=None, label=None, prototypes=None, glass_active=False,
        glass_steps=1, glass_step=0.001,
    ):
        features = self.base.feature_extractor(images)
        adapted = self.base.feature_adaptor(features)
        residual = self.residual_map(features, prototypes)
        seg_features = adapted
        cls_features = adapted if self.base.adapt_cls_feat else features
        if self.training:
            original_mask = mask
            if self.base.config["noise"]:
                noised_feat, noised_adapt, target_mask, target_label = self.base.anomaly_generator(
                    features=None if self.base.adapt_cls_feat else features,
                    adapted=adapted,
                    mask=mask,
                    labels=label,
                )
                seg_features = noised_adapt
                cls_features = noised_adapt if self.base.adapt_cls_feat else noised_feat
                if residual is not None:
                    residual = torch.cat((residual, residual), dim=0)
                if glass_active:
                    original_repeated = torch.cat((original_mask, original_mask), dim=0)
                    synth_mask = ((target_mask > 0) & (original_repeated <= 0)).float()
                    seg_features, cls_features = self.glass_hard_mine(
                        seg_features, cls_features, residual, synth_mask,
                        glass_steps, glass_step,
                    )
                mask, label = target_mask, target_label
            anomaly_map, score = self.discriminate(seg_features, cls_features, residual)
            return anomaly_map, score, mask, label
        anomaly_map, score = self.discriminate(seg_features, cls_features, residual)
        anomaly_map = self.base.anomaly_map_generator(anomaly_map)
        return anomaly_map, score

    def get_optimizers(self):
        seg_params, dec_params = self.base.discriminator.get_params()
        optimizer = torch.optim.AdamW(
            [
                {"params": self.base.feature_adaptor.parameters(), "lr": self.base.config["adapt_lr"]},
                {"params": seg_params, "lr": self.base.config["seg_lr"], "weight_decay": 1e-5},
                {"params": dec_params, "lr": self.base.config["dec_lr"], "weight_decay": 1e-5},
            ]
        )
        milestones = [
            max(1, int(self.base.config["epochs"] * 0.8)),
            max(2, int(self.base.config["epochs"] * 0.9)),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=self.base.config["gamma"]
        )
        return optimizer, scheduler

    def save_model(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("base.feature_extractor")
        }
        torch.save(state, directory / "weights.pt")

    def load_model(self, path: Path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)


@torch.no_grad()
def build_prototypes(model, rows, image_size, batch_size, workers, device, size):
    normal_rows = [row for row in rows if row["label"] == 0]
    loader = DataLoader(
        TileDataset(normal_rows, image_size, training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    sums, counts = {}, Counter()
    model.eval()
    for batch in loader:
        features = model.base.feature_extractor(batch["image"].to(device, non_blocking=True))
        pooled = F.adaptive_avg_pool2d(features, (size, size)).cpu()
        for index, key in enumerate(batch["prototype_key"]):
            sums[key] = sums.get(key, torch.zeros_like(pooled[index])) + pooled[index]
            counts[key] += 1
    prototypes = {key: value / counts[key] for key, value in sums.items()}
    default = torch.stack(list(prototypes.values())).mean(dim=0)
    return prototypes, default


def prototype_batch(keys, prototypes, default, device):
    if prototypes is None:
        return None
    return torch.stack([prototypes.get(key, default) for key in keys]).to(
        device, non_blocking=True
    )


def aggregate_sources(tile_rows, tile_scores, mode):
    grouped_scores = defaultdict(list)
    metadata = {}
    for row, score in zip(tile_rows, tile_scores):
        grouped_scores[row["source_id"]].append(float(score))
        metadata[row["source_id"]] = row
    result = []
    for source_id, values in grouped_scores.items():
        ordered = sorted(values, reverse=True)
        if mode == "max":
            image_score = ordered[0]
        elif mode == "top3":
            image_score = float(np.mean(ordered[: min(3, len(ordered))]))
        else:
            image_score = float(np.mean(ordered[: min(2, len(ordered))]))
        row = metadata[source_id]
        result.append(
            {
                "source_id": source_id,
                "label": int(row["label"]),
                "source_class": row["source_class"],
                "score": image_score,
                "tile_count": len(values),
            }
        )
    return sorted(result, key=lambda item: item["source_id"])


def choose_threshold(labels, scores):
    from sklearn.metrics import f1_score

    candidates = np.unique(np.r_[0.0, np.linspace(0.01, 0.99, 99), scores, 1.0])
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        predictions = scores >= threshold
        metric = f1_score(labels, predictions, zero_division=0)
        if metric > best[0]:
            best = (float(metric), float(threshold))
    return best[1]


def source_metrics(records, threshold=None):
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = np.asarray([record["label"] for record in records], dtype=np.uint8)
    scores = np.asarray([record["score"] for record in records], dtype=np.float64)
    if threshold is None:
        threshold = choose_threshold(labels, scores)
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    per_class = {}
    classes = sorted({record["source_class"] for record in records if record["label"] == 1})
    for class_name in classes:
        subset = [record for record in records if record["source_class"] == class_name]
        per_class[class_name] = float(
            np.mean([record["score"] >= threshold for record in subset])
        )
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision_ng": float(precision_score(labels, predictions, zero_division=0)),
        "recall_ng": float(recall_score(labels, predictions, zero_division=0)),
        "specificity_ok": float(tn / max(tn + fp, 1)),
        "false_positive_rate_ok": float(fp / max(tn + fp, 1)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_defect_recall": per_class,
        "images": int(len(labels)),
    }


@torch.no_grad()
def score_tiles(model, rows, args, prototypes, default, device):
    loader = DataLoader(
        TileDataset(rows, args.image_size, training=False),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    model.eval()
    scores = []
    for batch in loader:
        proto = prototype_batch(batch["prototype_key"], prototypes, default, device)
        _, logits = model(batch["image"].to(device, non_blocking=True), prototypes=proto)
        scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def evaluate_all_aggregations(model, rows, args, prototypes, default, device, thresholds=None):
    tile_scores = score_tiles(model, rows, args, prototypes, default, device)
    outputs = {}
    source_records = {}
    for mode in ("max", "top2", "top3"):
        records = aggregate_sources(rows, tile_scores, mode)
        threshold = None if thresholds is None else thresholds[mode]
        outputs[mode] = source_metrics(records, threshold)
        source_records[mode] = records
    return outputs, source_records


def write_predictions(path: Path, records: list[dict], threshold: float):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_id", "label", "source_class", "score", "prediction", "tile_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "prediction": int(record["score"] >= threshold)})


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(args.official_repo.resolve()))
    from model.supersimplenet import SuperSimpleNet

    rows = read_rows(args.manifest, args.shots_per_class)
    if not rows["train"] or not rows["val"] or not rows["test"]:
        raise ValueError("Manifest must contain non-empty train, val and test splits")
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
    model = CPGSuperSimpleNet(SuperSimpleNet, args.image_size, config, args.prototype).to(device)
    prototypes = default_prototype = None
    if args.prototype:
        prototypes, default_prototype = build_prototypes(
            model, rows["train"], args.image_size, args.batch, args.workers,
            device, args.prototype_size,
        )
    train_dataset = TileDataset(rows["train"], args.image_size, training=True)
    counts = Counter(row["label"] for row in rows["train"])
    weights = [1.0 / counts[row["label"]] for row in rows["train"]]
    epoch_samples = min(args.max_train_samples, max(counts.values()) * 2)
    sampler = WeightedRandomSampler(weights, num_samples=epoch_samples, replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    optimizer, scheduler = model.get_optimizers()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if prototypes is not None:
        torch.save(
            {
                "prototypes": prototypes,
                "default": default_prototype,
                "prototype_size": args.prototype_size,
                "templates": sorted({row["template"] for row in rows["train"]}),
            },
            args.output_dir / "prototypes.pt",
        )
    history = []
    best_ap, stale = float("-inf"), 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        started = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            masks = torch.zeros(len(images), 1, model.fh, model.fw, device=device)
            proto = prototype_batch(
                batch["prototype_key"], prototypes, default_prototype, device
            )
            anomaly_map, score, target_mask, target_label = model(
                images,
                masks,
                labels,
                prototypes=proto,
                glass_active=args.glass and epoch > args.glass_warmup,
                glass_steps=args.glass_steps,
                glass_step=args.glass_step,
            )
            seg_focal = focal_loss(torch.sigmoid(anomaly_map), target_mask).mean()
            normal_scores = anomaly_map[target_mask == 0]
            anomaly_scores = anomaly_map[target_mask > 0]
            zero = anomaly_map.sum() * 0
            normal_margin = (
                torch.clip(normal_scores + 0.5, min=0).mean()
                if normal_scores.numel() else zero
            )
            anomaly_margin = (
                torch.clip(-anomaly_scores + 0.5, min=0).mean()
                if anomaly_scores.numel() else zero
            )
            cls_loss = focal_loss(torch.sigmoid(score), target_label).mean()
            loss = seg_focal + normal_margin + anomaly_margin + cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "epoch_seconds": time.perf_counter() - started,
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation, _ = evaluate_all_aggregations(
                model, rows["val"], args, prototypes, default_prototype, device
            )
            record["validation_top2"] = validation["top2"]
            print(json.dumps(record, ensure_ascii=False))
            if validation["top2"]["auprc"] > best_ap:
                best_ap, stale = validation["top2"]["auprc"], 0
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
    model.to(device)
    validation, validation_records = evaluate_all_aggregations(
        model, rows["val"], args, prototypes, default_prototype, device
    )
    thresholds = {mode: values["threshold"] for mode, values in validation.items()}
    test, test_records = evaluate_all_aggregations(
        model, rows["test"], args, prototypes, default_prototype, device, thresholds
    )
    for mode in ("max", "top2", "top3"):
        write_predictions(
            args.output_dir / f"validation_predictions_{mode}.csv",
            validation_records[mode], thresholds[mode],
        )
        write_predictions(
            args.output_dir / f"test_predictions_{mode}.csv",
            test_records[mode], thresholds[mode],
        )

    sample_batch = next(iter(DataLoader(
        TileDataset(rows["val"][: args.batch], args.image_size),
        batch_size=min(args.batch, len(rows["val"])), shuffle=False,
    )))
    sample_images = sample_batch["image"].to(device)
    sample_proto = prototype_batch(
        sample_batch["prototype_key"], prototypes, default_prototype, device
    )
    model.eval()
    for _ in range(5):
        model(sample_images, prototypes=sample_proto)
    if device == "cuda":
        torch.cuda.synchronize()
    timings = []
    for _ in range(args.benchmark_iters):
        start = time.perf_counter()
        model(sample_images, prototypes=sample_proto)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000 / len(sample_images))

    summary = {
        "name": args.name,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "prototype": args.prototype,
        "prototype_count": 0 if prototypes is None else len(prototypes),
        "glass": args.glass,
        "glass_steps": args.glass_steps if args.glass else 0,
        "shots_per_class": args.shots_per_class,
        "train_tile_counts": dict(counts),
        "epoch_samples": epoch_samples,
        "epochs_completed": history[-1]["epoch"],
        "validation": validation,
        "test": test,
        "selected_aggregation": "top2",
        "latency_mean_ms_per_tile": float(np.mean(timings)),
        "latency_p95_ms_per_tile": float(np.percentile(timings, 95)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "normal_data_warning": "OK validation/test images are synthetic; real-OK calibration is pending",
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
