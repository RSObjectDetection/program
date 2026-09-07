# Experiment results

All values below are produced by the reproducible runner. `val` is used for model selection; the held-out `test` set must only be evaluated after configuration freeze.

| Experiment | Input | mAP50 | mAP50-95 | Recall | Params (M) | Mean latency (ms) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| e00_random_yolo11n_1024 | 1024 | 0.924 | 0.516 | 0.882 | 2.59 | 64.3 | 15.6 |
| e01_group_yolo11n_1024 | 1024 | 0.901 | 0.473 | 0.851 | 2.59 | 63.5 | 15.7 |
| e02_group_yolo11n_640 | 640 | 0.695 | 0.317 | 0.621 | 2.59 | 57.8 | 17.3 |
| e04_group_yolo11n_1024_5shot | 1024 | 0.313 | 0.153 | 0.316 | 2.59 | 62.2 | 16.1 |
| e04b_group_yolo11n_1024_5shot_freeze10 | 1024 | 0.258 | 0.118 | 0.284 | 2.59 | 61.9 | 16.2 |
| e05_group_yolo11n_1024_10shot | 1024 | 0.612 | 0.303 | 0.601 | 2.59 | 76.7 | 13.0 |
| e06a_group_yolo11n_1280_default | 1280 | 0.956 | 0.491 | 0.910 | 2.59 | 68.9 | 14.5 |
| e06b_group_yolo11n_1280_tuned_aug | 1280 | 0.926 | 0.492 | 0.850 | 2.59 | 63.9 | 15.7 |
| e07_group_yolo11n_p2_1024 | 1024 | 0.867 | 0.445 | 0.801 | 2.64 | 58.6 | 17.1 |

## Per-class mAP50-95

| Experiment | missing_hole | mouse_bite | open_circuit | short | spur | spurious_copper |
|---|---:|---:|---:|---:|---:|---:|
| e00_random_yolo11n_1024 | 0.610 | 0.496 | 0.552 | 0.497 | 0.389 | 0.553 |
| e01_group_yolo11n_1024 | 0.526 | 0.508 | 0.459 | 0.328 | 0.445 | 0.568 |
| e02_group_yolo11n_640 | 0.486 | 0.205 | 0.283 | 0.269 | 0.220 | 0.441 |
| e04_group_yolo11n_1024_5shot | 0.452 | 0.075 | 0.041 | 0.090 | 0.036 | 0.224 |
| e04b_group_yolo11n_1024_5shot_freeze10 | 0.342 | 0.051 | 0.075 | 0.030 | 0.047 | 0.164 |
| e05_group_yolo11n_1024_10shot | 0.520 | 0.223 | 0.167 | 0.188 | 0.281 | 0.440 |
| e06a_group_yolo11n_1280_default | 0.497 | 0.485 | 0.489 | 0.450 | 0.509 | 0.517 |
| e06b_group_yolo11n_1280_tuned_aug | 0.521 | 0.488 | 0.468 | 0.387 | 0.496 | 0.596 |
| e07_group_yolo11n_p2_1024 | 0.501 | 0.422 | 0.463 | 0.301 | 0.453 | 0.528 |

## Frozen-model test result

After selecting E06a from validation results, its checkpoint was evaluated once on the untouched PCB-template test split (groups 11 and 12; 120 images and 606 boxes).

| Model | Precision | Recall | mAP50 | mAP50-95 | Params (M) | Size (MB) | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n, 1280 | 0.967 | 0.920 | 0.975 | 0.489 | 2.59 | 5.35 | 59.6 |

Test per-class mAP50-95: missing hole 0.462, mouse bite 0.528, open circuit 0.497, short 0.443, spur 0.504, and spurious copper 0.499.

## Decisions

- Use E06a as the current detector baseline. Compared with 1024 input, it improves validation recall from 0.851 to 0.910 with the same 2.59M-parameter model.
- Reject 640 input: mAP50-95 falls from 0.473 to 0.317 for only about a 9% reduction in measured end-to-end latency.
- Reject the tuned-augmentation run as the primary checkpoint: its 0.001 mAP50-95 gain is outweighed by a 0.060 recall drop.
- Reject the custom P2 head in its present form. Only part of the pretrained detection head transfers, and accuracy regresses.
- Do not freeze ten modules in the 5-shot setting. It reduces mAP50-95 from 0.153 to 0.118.
- Ten images per class materially outperform five (0.303 versus 0.153 mAP50-95), but the full training set remains substantially better.

Latency numbers include preprocessing and postprocessing in the PyTorch runner on an RTX 3090 and should not be interpreted as edge-device TensorRT latency.
