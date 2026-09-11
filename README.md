# Lightweight Chip OK/NG Classification

Reproducible SuperSimpleNet/CPG-SSN experiments for small-data industrial chip inspection. The current task outputs one image-level label only: **OK or NG**.

## Dataset discovered on the server

- 693 high-resolution JPEG images
- 693 Pascal VOC XML annotation files
- Six bounding-box classes: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, and `spurious_copper`
- No defect-free images are included in the supplied archive

The six defect variants with the same template/sample index are almost perfectly aligned. One synthetic OK image is generated per complete group using a channel-wise pixel median, and the entire group is kept in one split to prevent near-duplicate base-image leakage. XML boxes are used only to select defect-containing training tiles; inference does not output boxes or masks.

## Current OK/NG result

S01–S09 have been executed. The leading deployment candidate is **CPG-SSN with a ResNet-18 backbone and Top-2 image aggregation**. On the isolated synthetic-OK test split (20 OK + 120 NG), it reaches AUROC/AUPRC/F1/NG recall/OK specificity of **1.000**, with **4.56M parameters** and **0.709 ms per 256×256 tile** on an RTX 4090.

This is a closed-dataset feasibility result, not a production claim: all available real images are NG and validation/test OK images are synthetic. Real production OK images are required for threshold calibration and final blind testing. See the [complete S01–S09 report](artifacts/S01-S09/REPORT.md) and [machine-readable summary](artifacts/S01-S09/summary.csv).

## Reproduce

```bash
python scripts/prepare_okng_dataset.py \
  --dataset-root /hy-tmp/data/raw \
  --output-root /hy-tmp/data/okng_512_o128 \
  --tile-size 512 --overlap 128 \
  --train-groups 80 --val-groups 15 --test-groups 20

# Pin the external implementation used by the experiment.
git clone https://github.com/blaz-r/SuperSimpleNet.git /root/SuperSimpleNet
git -C /root/SuperSimpleNet checkout 98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d
git -C /root/SuperSimpleNet apply /root/program/patches/supersimplenet_py38.patch

python scripts/run_cpg_ssn.py \
  --manifest /hy-tmp/data/okng_512_o128/tiles.csv \
  --official-repo /root/SuperSimpleNet \
  --output-dir /hy-tmp/runs/deployment_candidate \
  --name deployment_candidate --backbone resnet18 \
  --image-size 256 --epochs 12 --batch 16 \
  --prototype --glass
```

Every experiment writes weights and full run output under `/hy-tmp/runs/`. Lightweight summaries, histories and image-level predictions are collected under `artifacts/S01-S09/`; raw data and model weights are deliberately excluded from Git.

The current implementation-aligned technical solution is [docs/TECHNICAL_PLAN_OK_NG_SUPERSIMPLENET.md](docs/TECHNICAL_PLAN_OK_NG_SUPERSIMPLENET.md). It explains the full SuperSimpleNet flow and the exact injection points of the position-normal prototype and GLASS-style hard-feature modules. Earlier detection and multi-class studies are retained in [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md), [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md), [docs/ANOMALY_BASELINES.md](docs/ANOMALY_BASELINES.md), [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), [docs/LITERATURE_REVIEW_2024_2025.md](docs/LITERATURE_REVIEW_2024_2025.md), [docs/FEW_SHOT_SUPERVISED_2024_2026.md](docs/FEW_SHOT_SUPERVISED_2024_2026.md), and [docs/TECHNICAL_PLAN_IMAGE_LEVEL_2024_2026.md](docs/TECHNICAL_PLAN_IMAGE_LEVEL_2024_2026.md).
