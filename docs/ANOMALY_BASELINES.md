# Anomaly baselines with anomaly-only source images

## What is valid

The source set contains defect images and complete VOC bounding boxes, but no known-good full images. This supports supervised detection directly. It does not support the standard EfficientAD protocol, whose teacher/student models learn a distribution from normal training images.

SuperSimpleNet is tested in a clearly labelled weak-supervision configuration:

1. Each source image is split into non-overlapping 512-by-512 tiles, with a final edge-aligned tile where needed.
2. Tiles intersecting any annotated box are anomalous; boxes become rectangular coarse masks.
3. Tiles not intersecting a box become local pseudo-normal samples.
4. PCB template IDs, not individual tiles, define train/validation/test membership, preventing source-image leakage.
5. A ResNet-18 ImageNet backbone replaces the default Wide-ResNet-50-2 to reduce parameters and latency.

This design can test local texture and contamination sensitivity, but it cannot prove full-image normal/anomaly discrimination. A box-free tile can still contain an unlabelled defect, and rectangular masks include background pixels.

## EfficientAD decision

Do not train standard EfficientAD on all supplied images as if they were normal. A future valid baseline requires representative good chips captured with the same camera, illumination, focus, product variant and alignment. Until then, an EfficientAD result would need to be labelled as a pseudo-normal experiment and kept out of production comparisons.

## Industrial recommendation

Use the selected lightweight detector for known defect localization. Add image registration and tolerance measurement for displacement. Collect known-good images per board template and production condition, then add EfficientAD-S or standard SuperSimpleNet as a second-stage unknown-defect guard. Calibrate operating thresholds on a validation stream that reflects the production false-reject cost.
