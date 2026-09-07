# 2024–2025 industrial anomaly-detection method review

This note records the methods considered for the PCB/chip inspection project and the reason for the current implementation choice. It intentionally distinguishes *supervised defect detection* from *normal-reference anomaly detection* because their data requirements are different.

## Methods most relevant to this project

| Method | Publication | Supervision / input | Industrial value | Constraint for the supplied data |
|---|---|---|---|---|
| EfficientAD | WACV 2024 | Normal images only; student–teacher plus an autoencoder | Explicitly designed for millisecond-level anomaly localization and logical anomalies; a strong lightweight production candidate | Cannot be trained honestly because the archive contains no verified normal images |
| PromptAD | CVPR 2024 | A few normal images; learned one-class vision-language prompts | Strong few-shot option when each new product has only several golden samples | Also requires verified normal references and a larger CLIP-style encoder |
| SuperSimpleNet | ICPR 2024; mixed-supervision extension in JIM 2025 | Unsupervised, weak, or supervised labels | One implementation can use different annotation regimes and synthetic anomalies; practical second-stage candidate | Pixel masks or normal images would be preferable; current boxes can only provide weak supervision |
| AnomalyDINO | WACV 2025 | One/few normal references; training-free DINOv2 patch matching | Very quick adaptation to a new product and strong few-shot localization without fine-tuning | Foundation encoder and patch memory are heavier than the nano detector; normal references are absent |
| SegAD | CVPR 2024 | Supervised real anomaly data plus anomaly maps | Shows that supervised anomaly learning is valuable on complex real industrial data | More complex ensemble pipeline than needed for six labelled PCB defect classes |
| ReMP-AD | ICCV 2025 | Few-shot normal references with retrieval and vision-language priors | Reported specifically on PCB-Bank and tackles noisy/incomplete few-shot memories | Heavier multi-modal stack and still needs normal references |
| YOLO-pdd | arXiv 2024 | Supervised PCB boxes | Multi-scale PCB detector intended for small defects and real-time inference | Sequential/reference-image assumptions are not available in the supplied archive |

Primary sources:

- EfficientAD: <https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html>
- PromptAD: <https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.html>
- SuperSimpleNet official implementation and paper metadata: <https://github.com/blaz-r/SuperSimpleNet>
- AnomalyDINO: <https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html>
- SegAD: <https://openaccess.thecvf.com/content/CVPR2024/html/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.html>
- ReMP-AD: <https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html>
- YOLO-pdd: <https://arxiv.org/abs/2407.15427>
- Industrial evaluation cautions and best practices: <https://openaccess.thecvf.com/content/CVPR2025W/VAND/html/Baitieva_Beyond_Academic_Benchmarks_Critical_Analysis_and_Best_Practices_for_Visual_CVPRW_2025_paper.html>

## Decision for the present archive

The archive contains 693 anomalous images, six known classes, and Pascal VOC bounding boxes, but no verified normal images. The first executable branch is therefore a lightweight supervised detector. YOLO11n is used as a reproducible baseline because it is only about 2.6 million parameters and supports straightforward ONNX/TensorRT deployment. High input resolution and template-group holdout are more important here than adding a large backbone because the defects occupy roughly 0.04%–0.10% of image area.

The planned production system is hybrid once normal data are collected:

1. YOLO11n (or a small-object variant) detects and classifies the six known defects.
2. EfficientAD-S provides a fast unknown-defect heat map from normal images.
3. Registration plus dimensional tolerances handles displacement/misalignment, which is a geometric measurement problem rather than only an appearance anomaly.
4. AnomalyDINO or ReMP-AD is evaluated only if rapid adaptation to many new PCB templates is more important than edge-device cost.

## Data collection required for the anomaly branch

For each PCB template, collect at least 20–50 verified normal images spanning illumination, focus, position, and production-batch variation. Keep entire production batches and template IDs isolated between training, validation, and test. Preserve pixel-level masks for a small set of scratches and contamination; they are much more informative than image labels for localization evaluation.
