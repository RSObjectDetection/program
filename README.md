# Lightweight PCB Defect Detection

Reproducible experiments for a small-data industrial PCB defect dataset.

## Dataset discovered on the server

- 693 high-resolution JPEG images
- 693 Pascal VOC XML annotation files
- Six bounding-box classes: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, and `spurious_copper`
- No defect-free images are included in the supplied archive

Because the archive contains bounding-box-labelled defect images but no normal-only training set, the primary task is formulated as **few-shot lightweight object detection**. An adapted SuperSimpleNet branch is also included: box-free 512-pixel crops from anomalous images are treated as *pseudo-normal local regions* and VOC boxes are converted to coarse masks. This is a weak-supervision experiment, not a standard normal-only anomaly-detection benchmark.

## Current result

The selected YOLO11n model uses a 1280-pixel input and a PCB-template group holdout. On the untouched 120-image test split it reaches mAP50 **0.975**, mAP50-95 **0.489**, recall **0.920**, with 2.59M parameters and a 5.35MB checkpoint. The random-split score is retained only as a leakage diagnostic.

## Reproduce

```bash
python scripts/audit_dataset.py \
  --dataset-root /hy-tmp/data/raw_deeppcb \
  --output-dir artifacts/data_audit

python scripts/prepare_yolo.py \
  --dataset-root /hy-tmp/data/raw_deeppcb \
  --output-root /hy-tmp/data/pcb_yolo \
  --split group_holdout

python scripts/run_yolo_experiment.py \
  --data /hy-tmp/data/pcb_yolo/group_holdout/data.yaml \
  --model /hy-tmp/weights/yolo11n.pt \
  --name e01_group_yolo11n_1024 \
  --imgsz 1024 --epochs 100 --batch 8

python scripts/prepare_ssn_tiles.py \
  --yolo-root /hy-tmp/data/pcb_yolo/group_holdout \
  --output-root /hy-tmp/data/pcb_ssn_tiles_512 \
  --tile-size 512

# Pin the external implementation used by the experiment.
git clone https://github.com/blaz-r/SuperSimpleNet.git /hy-tmp/SuperSimpleNet
git -C /hy-tmp/SuperSimpleNet checkout 98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d
git -C /hy-tmp/SuperSimpleNet apply /hy-tmp/program/patches/supersimplenet_py38.patch

python scripts/run_supersimplenet_pcb.py \
  --manifest /hy-tmp/data/pcb_ssn_tiles_512/manifest.csv \
  --official-repo /hy-tmp/SuperSimpleNet \
  --output-dir artifacts/anomaly_baselines/supersimplenet_box_tiles_r18 \
  --image-size 512 --epochs 30 --batch 16
```

Every experiment writes its configuration, logs, metrics and model metadata under `runs/`. Large model weights and the raw dataset are deliberately excluded from Git.

See [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md), [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md), and [docs/LITERATURE_REVIEW_2024_2025.md](docs/LITERATURE_REVIEW_2024_2025.md).
