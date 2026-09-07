# Anomaly baselines with anomaly-only source images

## What is valid

The source set contains defect images and complete VOC bounding boxes, but no known-good full images. This supports supervised detection directly. It does not support the standard EfficientAD protocol, whose teacher/student models learn a distribution from normal training images.

SuperSimpleNet is tested in a clearly labelled weak-supervision configuration:

1. Each source image is split into non-overlapping 512-by-512 tiles, with a final edge-aligned tile where needed.
2. Tiles intersecting any annotated box are anomalous; boxes become rectangular coarse masks.
3. Tiles not intersecting a box become local pseudo-normal samples.
4. PCB template IDs, not individual tiles, define train/validation/test membership, preventing source-image leakage.
5. A ResNet-18 ImageNet backbone replaces the default Wide-ResNet-50-2, and the 512-pixel tile is resized to 256 for the network, reducing parameters, feature-map memory and latency.

This design can test local texture and contamination sensitivity, but it cannot prove full-image normal/anomaly discrimination. A box-free tile can still contain an unlabelled defect, and rectangular masks include background pixels.

## Result

The 256-input ResNet-18 run uses all 13,050 training tiles with balanced sampling of the 2,446 anomalous tiles. On held-out test templates it reaches 0.900 tile AUROC, 0.797 tile AP, 0.960 coarse-mask pixel AUROC, 0.419 pixel AP and 0.470 best pixel F1. It has 4.56M parameters and averages 4.03ms per tile on the RTX 3090.

Using only validation data, the frozen tile and pixel thresholds are 0.65 and 0.66. Applied unchanged to the test set, they produce tile F1 0.663 and coarse-mask pixel F1 0.439. Qualitative heat maps show false activation around crop boundaries and high-contrast solder features; production tiled inference should therefore use overlapping crops with center-weighted blending.

These values show that the box-derived task is learnable. They do not establish a production false-reject rate because every source image contains defects and the negative class consists only of local box-free crops.

## EfficientAD decision

Do not train standard EfficientAD on all supplied images as if they were normal. A future valid baseline requires representative good chips captured with the same camera, illumination, focus, product variant and alignment. Until then, an EfficientAD result would need to be labelled as a pseudo-normal experiment and kept out of production comparisons.

## Industrial recommendation

Use the selected lightweight detector for known defect localization. Add image registration and tolerance measurement for displacement. Collect known-good images per board template and production condition, then add EfficientAD-S or standard SuperSimpleNet as a second-stage unknown-defect guard. Calibrate operating thresholds on a validation stream that reflects the production false-reject cost.
