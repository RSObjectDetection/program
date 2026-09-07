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
| E05 | Few-shot | PCB-template holdout | YOLO11n | 1024 | 10 images/class |
| E06 | Tiny-defect optimization | PCB-template holdout | YOLO11n | 1280 | full, tuned augmentation |

E00 is diagnostic only and must not be quoted as the production estimate. E01 is the baseline used for decisions.

## 4. Optimization rules

1. If E01 is much worse than E00, treat the gap as evidence of template-domain shift. Do not tune against the test templates; improve geometric and photometric augmentation and repeat on validation templates.
2. If small defects dominate false negatives, increase input resolution or train overlapping crops while retaining full-image validation.
3. If E03 gives less than a two-point mAP50-95 gain over E01, keep the nano model.
4. If a class has low recall, inspect its box-size distribution and confusion matrix before using class weights or oversampling.
5. Select thresholds using validation data only. The held-out test split remains untouched until a configuration is frozen.

## 5. Follow-up normal-only anomaly detection

When matching normal images become available, add a second branch using EfficientAD-S or SuperSimpleNet. The supervised detector will identify the six known defects, while the anomaly branch will flag unknown contamination, scratches and process drift. Geometric displacement should remain a separate registration/tolerance measurement rather than being inferred only from an anomaly score.

