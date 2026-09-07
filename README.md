# Lightweight PCB Defect Detection

Reproducible experiments for a small-data industrial PCB defect dataset.

## Dataset discovered on the server

- 693 high-resolution JPEG images
- 693 Pascal VOC XML annotation files
- Six bounding-box classes: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, and `spurious_copper`
- No defect-free images are included in the supplied archive

Because the archive contains bounding-box-labelled defect images but no normal-only training set, the first project stage is formulated as **few-shot lightweight object detection**. A normal-only anomaly-detection branch will be added when matching defect-free images become available.

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
```

Every experiment writes its configuration, logs, metrics and model metadata under `runs/`. Large model weights and the raw dataset are deliberately excluded from Git.

See [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) for the experiment matrix and decision rules.

