# Experiment plan

## 1. Problem statement

The supplied archive is a Pascal VOC object-detection dataset rather than a normal-only anomaly-detection dataset. It contains 693 anomalous PCB images and bounding boxes for six defect classes, but no known-good images. The immediate task is therefore to detect and classify known defects from few labelled images with a lightweight detector.

Two risks dominate the evaluation:

1. The files use a leading PCB template identifier such as `01`, `04`, or `12`. Random image splitting can place near-identical PCB layouts in both training and test sets and produce an unrealistically optimistic score.
2. Defects occupy a very small fraction of each 3k-by-1.5k image. Aggressive resizing can erase the signal.

## 2. Metrics

- Primary: box mAP50-95 and per-class recall on the group-held-out test set.
- Safety metric: minimum per-class recall and false positives per image.
- Efficiency: parameter count, model file size, batch-1 latency, throughput and peak GPU memory.
- Stability: mean and standard deviation across three seeds for shortlisted configurations.

## 3. Experiment matrix

| ID | Purpose | Split | Model | Resolution | Data |
|---|---|---|---|---:|---|
| E00 | Leakage diagnostic | random stratified | YOLO11n | 1024 | full |
| E01 | Honest lightweight baseline | PCB-template holdout | YOLO11n | 1024 | full |
| E02 | Resolution ablation | PCB-template holdout | YOLO11n | 640 | full |
| E03 | Capacity ablation | PCB-template holdout | YOLO11s | 1024 | full |
| E04 | Extreme few-shot | PCB-template holdout | YOLO11n | 1024 | 5 images/class |
| E04b | Frozen-backbone few-shot | PCB-template holdout | YOLO11n, freeze 10 | 1024 | 5 images/class |
| E05 | Few-shot | PCB-template holdout | YOLO11n | 1024 | 10 images/class |
| E06a | Resolution optimization | PCB-template holdout | YOLO11n | 1280 | full, default augmentation |
| E06b | Augmentation optimization | PCB-template holdout | YOLO11n | 1280 | full, reduced scale/color/Mosaic |
| E07 | Tiny-object head | PCB-template holdout | YOLO11n-P2 | 1024 | full |
| E08 | Box-derived anomaly baseline | PCB-template holdout | SuperSimpleNet, ResNet-18 | 512 tiles | pseudo-normal + coarse masks |

E00 is diagnostic only and must not be quoted as the production estimate. E01 is the baseline used for decisions.

## 4. Optimization rules

1. If E01 is much worse than E00, treat the gap as evidence of template-domain shift. Do not tune against the test templates; improve geometric and photometric augmentation and repeat on validation templates.
2. If small defects dominate false negatives, increase input resolution or train overlapping crops while retaining full-image validation.
3. If E03 gives less than a two-point mAP50-95 gain over E01, keep the nano model.
4. If a class has low recall, inspect its box-size distribution and confusion matrix before using class weights or oversampling.
5. Select thresholds using validation data only. The held-out test split remains untouched until a configuration is frozen.

## 5. Execution status and decisions

- E00 confirms a modest random-split optimism gap and is diagnostic only.
- E01 establishes the honest 1024 baseline.
- E02 shows that 640 input removes too much tiny-defect information for only a small latency gain; it is rejected.
- E06a improves recall materially with a modest latency increase and is the current production candidate.
- E04b is rejected: freezing the first ten modules reduces 5-shot mAP50-95 from 0.153 to 0.118.
- E05 reaches 0.303 mAP50-95 with 10 images per class, nearly twice the 5-shot value but still below full-data training.
- E06b is rejected as the primary model: mAP50-95 changes only from 0.491 to 0.492 while recall falls from 0.910 to 0.850.
- E07 is rejected: its P2 head transfers fewer pretrained parameters and reaches 0.445 mAP50-95, below the ordinary nano model.
- The frozen E06a checkpoint reaches 0.975 mAP50, 0.489 mAP50-95 and 0.920 recall on the untouched test split.
- E08 evaluates whether local box-free regions can substitute for normal images. Its scores must be reported with the pseudo-normal/coarse-mask qualification.

## 6. Follow-up normal-only anomaly detection

The present E08 branch is an adapted SuperSimpleNet baseline because it can consume positive coarse masks together with pseudo-normal regions. Standard EfficientAD is intentionally not reported: its student-teacher training assumes known-good images, so using the supplied anomaly-only images as normal would invalidate the objective. When matching good chips become available, train EfficientAD-S and the standard normal-only SuperSimpleNet protocol. The supervised detector should identify the six known defects, while the anomaly branch flags unknown contamination, scratches and process drift. Geometric displacement should remain a separate registration/tolerance measurement rather than being inferred only from an anomaly score.
