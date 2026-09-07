# Deployment note

## Selected detector

The frozen deployment candidate is E06a: YOLO11n with a 1280-pixel input. Its held-out template test result is 0.975 mAP50, 0.489 mAP50-95, 0.920 recall and 0.967 precision. The checkpoint contains 2.59M parameters and is approximately 5.35MB.

The exported ONNX graph has a fixed batch-1 input of `[1, 3, 1280, 1280]` and output `[1, 10, 33600]`. It is 10.56MB, passes `onnx.checker`, and has SHA-256 `d9e1d9f546f808c528b1b2c44da4bacd0f1b200327f630335f292c89bddf7bdf`. The binary is excluded from Git; its server location and metadata are recorded under `artifacts/deployment/yolo11n_1280/`.

The measured PyTorch end-to-end latency on the RTX 3090 is 59.6ms/image. Raw ONNX Runtime CPU graph execution on the server's AMD EPYC 7601 averages 2.48s and excludes preprocessing and NMS, so it is a validity check rather than a production benchmark. For a production GPU, build an FP16 TensorRT engine on the target CUDA/TensorRT stack and benchmark the complete camera-to-decision path.

## Recommended inspection flow

1. Apply camera calibration, flat-field correction and PCB registration.
2. Reject or separately measure geometric displacement using fiducials and dimensional tolerances.
3. Run YOLO11n for the six known defect classes.
4. When verified normal images become available, run EfficientAD-S or standard SuperSimpleNet as an unknown-defect guard.
5. Fuse decisions with class-specific confidence thresholds calibrated on production validation data.
6. Store the source image, model version, threshold version, detections and latency for traceability.

## Current blockers to production acceptance

- There are no defect-free boards, so false positives per normal image and false-reject rate cannot be measured.
- The archive has no displacement class or fiducial tolerance labels; the detector cannot validate offset specifications.
- Rectangular boxes are not precise segmentation masks, so scratch/contamination area estimates are not yet calibrated.
- The reported test set covers held-out PCB templates but not new cameras, illumination settings or production batches.

Before a line trial, collect known-good boards and at least one independently captured batch per product template. Lock the model and thresholds before that acceptance test.
