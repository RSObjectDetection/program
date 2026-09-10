# 芯片整图少样本缺陷分类技术方案（2024–2026，已被简化方案取代）

> 状态：本方案包含多类分类和多个研究模块，已不再作为当前执行方案。当前仅做OK/NG，并采用单一SuperSimpleNet主线，见 [TECHNICAL_PLAN_OK_NG_SUPERSIMPLENET.md](TECHNICAL_PLAN_OK_NG_SUPERSIMPLENET.md)。

## 1. 已确认的任务

- 输入是一张完整芯片二维图像。
- 每张图至多对应一种缺陷类别。
- 最终不需要缺陷框、掩码或偏移量，只需输出整图类别。
- 每个已知缺陷类别约有 30–100 张真实图像，可参与训练。
- 若多类识别达不到要求，允许降级为 `OK/NG` 二分类。
- 当前没有真实正常图，但同型号芯片结构基本一致，可对齐不同缺陷图并用正常位置覆盖缺陷区域，合成近似正常图。
- 当前部署目标是服务器 GPU，后续硬件未确定。
- 不以 YOLO 或目标检测为主线。

因此任务定义为：**弱定位辅助的少样本整图分类 + 正常参考驱动的异常判定**。虽然最终不输出位置，但芯片缺陷通常占整图比例很小，模型内部仍需保留局部尺度并自动寻找最可疑区域。

## 2. 推荐总体方案：RA-GLPC

暂将方案命名为 **RA-GLPC（Reference-Aligned Global–Local Prototype Classifier）**。它不是复刻单篇论文，而是组合近两年顶会方法中适合工业落地的机制。

```text
原始芯片图
   │
   ├─ M1：模板识别、ROI裁剪、亚像素配准、亮度归一化
   │
   ├─ M2：多参考正常原型库
   │      ├─ 对齐图像的稳健中值正常模板
   │      ├─ 跨图同坐标正常区域替换
   │      └─ 合成质量门控
   │
   ├─ M3：全图 + 重叠局部块的冻结视觉编码器
   │
   ├─ M4：正常原型差异图与缺陷显著性
   │      ├─ 多尺度patch最近邻差异
   │      ├─ 查询感知原型细化
   │      └─ 早期纹理/边缘缺陷线索保留
   │
   ├─ M5：显著性引导的MIL整图分类
   │      ├─ 全局特征
   │      ├─ Top-k可疑局部特征
   │      └─ 类别余弦原型/线性适配头
   │
   └─ M6：分层工业决策
          ├─ OK
          ├─ 已知缺陷类别
          └─ NG-未知/低置信度
```

最终接口只返回整图结论、置信度和模型版本。内部差异图可以作为调试证据保存，但不属于业务输出要求。

## 3. 各模块设计

### M1：图像标准化与配准

这是同型号芯片场景最重要的工程前提，优先使用可解释且稳定的传统几何方法。

1. 使用治具边缘、焊盘或固定结构确定芯片 ROI。
2. 先用相位相关估计平移，再用 ECC 或特征匹配估计仿射/单应变换。
3. 以固定模板坐标系重采样，记录变换矩阵和配准残差。
4. 使用局部亮度场校正、白平衡和对比度归一化，避免模型把光照当缺陷。
5. 配准残差超过阈值时返回 `INVALID_IMAGE`，不进入分类器。

不建议用过强的弹性形变，因为它可能改变偏移类缺陷。旋转、缩放和平移增强必须限制在相机与治具真实公差内。

### M2：正常图与正常原型库

#### M2.1 稳健中值模板

将同型号、同视角的多张缺陷图配准后，在像素或特征层计算稳健中值。由于每张图只有一个缺陷且缺陷位置不完全相同，中值能够消除大部分局部异常，形成第一版正常模板。

#### M2.2 同坐标跨图替换

对图像 `x_i` 的疑似缺陷区域 `R_i`，从另一张已配准图 `x_j` 的同坐标区域取正常候选。供体需要满足：

- 供体自身缺陷不与 `R_i` 重叠；
- `R_i` 周围环带与目标图颜色、梯度和结构相似；
- 供体区域相对中值模板的异常分数低；
- 替换后边界梯度连续。

使用颜色矩匹配和窄边缘 alpha/Poisson 融合，避免硬拼接边界。

#### M2.3 缺陷区域从哪里来

- 如果已有历史框标注，框只用于离线正常合成，不要求模型输出框。
- 如果只有整图类别，先用“对齐图像－稳健中值模板”的像素差、SSIM差和冻结特征差生成候选掩码，再取连通区域。
- 对低置信候选不生成正常图，避免把真实缺陷残留在 `OK` 标签中。

#### M2.4 合成质量门控

每张合成正常图必须同时通过：

- 替换区内部相对正常原型的特征距离阈值；
- 边界环带的梯度跳变阈值；
- 替换区外与原图的像素一致性；
- 至少两个不同正常参考的一致投票；
- 人工抽检通过率。

合成正常图只能进入训练集。没有真实正常图时，可以建立“合成验证集”用于开发，但不能据此宣称真实产线 OK 误报率。

### M3：多尺度冻结视觉特征

初始教师模型使用 DINOv2 ViT-S/14。选择依据不是其发布时间，而是 AnomalyDINO（WACV 2025）、UniVAD（CVPR 2025）和 FastRef（CVPR 2026）已验证其 patch 特征适合工业少样本异常检测。

输入采用两种视图：

- 全图视图：保留芯片整体结构、全局偏移和污染范围。
- 局部视图：按固定坐标切成重叠块，保持划痕、污染和微小结构异常的原始像素尺度。

编码器第一阶段完全冻结，只训练轻量投影层和分类头。若 50–100 张/类时仍欠拟合，再仅解冻最后 1–2 个 block，并使用较小学习率。

### M4：正常差异与缺陷显著性

对查询 patch 特征 `f_i(p)` 和正常原型库 `M(p)` 计算位置约束的最近邻距离：

```text
A_i(p) = min_m distance(normalize(f_i(p)), normalize(M_m(p)))
```

其中 `A_i(p)` 是内部异常显著图。实现上融合以下思路：

- **AnomalyDINO**：冻结视觉特征和 patch 最近邻差异，提供简单可靠的少样本异常分数。
- **FastRef**：在不更新骨干的情况下，用查询统计细化正常原型，并抑制异常被正常原型错误重构。
- **DCP-SFR**：融合浅层边缘/纹理和深层语义特征，避免划痕、轻污染等弱缺陷在线路深层中逐渐消失。
- **UniVAD**：在固定芯片区域内做位置/部件约束匹配，避免把不同功能区的相似纹理互相匹配。

与这些论文不同，本方案不追求像素级分割分数，而是把显著图作为分类注意力和二分类证据。

### M5：显著性引导的整图分类

#### M5.1 多实例学习聚合

每张完整图是一组局部块。根据 `A_i` 选择异常分数最高的 Top-k 个块，用可学习注意力聚合；全图特征保留为另一条分支：

```text
z_local  = AttentionPool(top-k suspicious patches)
z_global = GlobalPool(full-image tokens)
z        = LayerNorm([z_global, z_local, statistics(A_i)])
```

这样只使用整图类别标签，也能让分类器重点学习真正的缺陷区域。

#### M5.2 已知缺陷分类头

采用“类别原型 + 轻量线性头”的混合形式：

- 每类原型是该类训练样本特征的稳健均值；
- 余弦相似度提供对小数据更稳定的类间距离；
- 轻量线性残差头学习芯片数据中的细粒度差异；
- 训练策略参考 CVPR 2024 CLAP 的结论：先以简单、低参数的线性适配为基准，避免少量验证数据下大规模超参数搜索和全模型微调。

AnomalyNCD（CVPR 2025）的“缺陷中心化和掩码引导表示”被改造成显著性加权监督分类；不采用其未知类别聚类，因为当前类别标签已知。

#### M5.3 OK/NG 二分类头

二分类头联合输入：

- 最大/均值/Top-k 异常分数；
- 全局特征与正常全局原型的距离；
- 最可疑局部块特征；
- 配准质量和图像质量特征。

该头用真实缺陷图作为 NG、质量门控通过的合成正常图作为 OK。它与多类分类头分别训练和校准，因此多类性能不足时，可以独立部署为 OK/NG 模型。

### M6：分层决策与拒识

```text
若图像质量/配准失败                  -> INVALID_IMAGE
否则若 P(NG) < T_ng 且正常距离 < T_d -> OK
否则若 max(P(class)) >= T_cls         -> 对应已知缺陷类别
否则                                  -> NG_UNKNOWN
```

阈值按工业目标选择：优先约束缺陷漏检率，再优化正常品误报。`NG_UNKNOWN` 很重要，它防止模型被迫把未见过的新缺陷错误归入某个已知类别。

## 4. 损失函数与训练策略

总损失建议为：

```text
L = L_class-balanced-CE
  + 0.3 * L_supervised-contrastive
  + 0.5 * L_OK/NG-BCE
  + 0.2 * L_view-consistency
  + 0.1 * L_normal-compactness
```

- `L_class-balanced-CE`：已知缺陷类别监督。
- `L_supervised-contrastive`：同类聚合、易混类别分离。
- `L_OK/NG-BCE`：独立二分类头。
- `L_view-consistency`：全图与局部视图、原图与安全增强之间预测一致。
- `L_normal-compactness`：约束合成正常特征靠近正常原型。

训练分三阶段：

1. 冻结编码器，建立正常原型和类别原型，仅训练投影层、MIL聚合器和双头。
2. 在验证集确认无严重过拟合后，解冻最后 1–2 个编码 block，小学习率联合微调。
3. 冻结模型，在验证集做温度缩放和三类阈值校准。

## 5. 数据增强

### 可以使用

- 相机真实范围内的亮度、色温、轻噪声和模糊；
- 治具公差内的平移、轻旋转和尺度变化；
- 同功能区、同坐标系内的缺陷 Copy-Paste；
- 不改变缺陷语义的小范围裁剪；
- 多参考正常区域替换。

### 谨慎使用

- MixUp/CutMix：可能制造不真实的芯片结构，只作为消融项；
- 水平/垂直翻转：只有芯片和类别语义对称时允许；
- 颜色强增强：污染类可能依赖颜色，不能破坏标签语义；
- 任意位置 Copy-Paste：缺陷必须出现在物理上可能的部件区域。

### 第二阶段生成增强

- AnoGen（ECCV 2024）证明少量真实异常可以引导扩散模型生成更贴近真实分布的异常。
- SeaS（ICCV 2025）将正常产品、不同异常属性和掩码生成进行解耦。

本项目第一版不直接引入扩散模型。只有真实数据曲线显示 30/50 张仍显著欠拟合时，再对最差类别加入生成增强；每次只增加一种生成策略，并在纯真实测试集上验证。

## 6. 实验矩阵

相同芯片实例、拍摄序列、生产批次不得跨训练/验证/测试。30/50/100 张采用嵌套训练子集，验证集和测试集固定。每个主要实验运行 3 个随机种子。

| ID | 方法 | 目的 |
|---|---|---|
| C00 | RGB全图 + ResNet18/RepViT + CE | 最低成本整图分类基线 |
| C01 | 冻结DINOv2 + 全图线性分类头 | 基础视觉特征基线 |
| C02 | 冻结DINOv2 + 多尺度局部块 + MIL | 验证微小缺陷尺度问题 |
| C03 | C02 + 正常原型差异图 | 验证参考差异是否提高弱缺陷识别 |
| C04 | C03 + 显著性引导类别原型头 | 完整多类主模型 |
| B00 | AnomalyDINO式正常原型 + 阈值 | 无训练/少训练OK-NG基线 |
| B01 | C03的独立OK/NG头 | 可部署二分类候选 |
| H00 | C04 + B01分层决策和拒识 | 最终服务器候选 |
| G01 | H00 + 受控同坐标缺陷Copy-Paste | 真实纹理增强消融 |
| G02 | H00 + AnoGen或SeaS增强 | 仅针对最差类别的生成消融 |
| D01 | 将H00蒸馏到RepViT/MobileNetV4 | 后续轻量部署候选 |

优先执行顺序为 `C00 → C02 → B00 → C03 → C04 → B01 → H00`。生成和蒸馏放在主流程有效之后。

## 7. 评价指标与验收

### 多类分类

- Macro-F1、Balanced Accuracy、Top-1 Accuracy；
- 每类召回率、精确率和混淆矩阵；
- 最差类别召回率；
- ECE/Brier score，检查置信度是否可信；
- `NG_UNKNOWN` 的覆盖率与已知类准确率曲线。

### OK/NG

- NG召回率/漏检率；
- OK误报率；
- AUROC和AUPRC；
- FPR@95TPR；
- 不同批次、亮度、焦距和配准误差下的稳定性。

### 工业效率

- batch=1 的P50/P95延迟；
- 显存峰值、吞吐率、模型大小；
- 预处理、配准、特征提取、决策各模块的分项耗时；
- 长时间运行中的失败率和输入质量拒绝率。

第一阶段建议的门槛不是直接设定绝对准确率，而是：完整方案必须稳定超过简单整图分类基线，并且三个随机种子的最差类别召回率没有明显坍缩。真实产线验收前仍需补充真实OK样本；合成OK只能用于研发训练。

## 8. 部署策略

### 服务器GPU第一版

- 教师骨干：DINOv2 ViT-S/14；
- 全图和局部块特征可批量一次性推理；
- 正常原型提前离线编码并常驻GPU或CPU内存；
- FP16推理，导出ONNX/TensorRT前验证数值一致性；
- 输出JSON：`decision`、`class_name`、`p_ng`、`p_class`、`quality_status`、`model_version`。

### 后续轻量化

若部署目标变成CPU或Jetson，采用知识蒸馏：以完整RA-GLPC为教师，将分类概率、正常距离和显著性加权特征蒸馏到RepViT或MobileNetV4学生模型。这样保留研发阶段的强模型，同时避免一开始为了未知硬件过早牺牲精度。

## 9. 主要论文来源与采用方式

- AnomalyDINO, WACV 2025：冻结DINO patch特征和少样本正常最近邻。<https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html>
- CLAP, CVPR 2024：少样本视觉语言模型的类别自适应线性探测。<https://openaccess.thecvf.com/content/CVPR2024/html/Silva-Rodriguez_A_Closer_Look_at_the_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2024_paper.html>
- GeneralAD, ECCV 2024：patch特征伪异常和注意力判别。<https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5481_ECCV_2024_paper.php>
- GLASS, ECCV 2024：全局/局部可控异常合成，尤其关注弱缺陷。<https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8382_ECCV_2024_paper.php>
- AnoGen, ECCV 2024：少量真实缺陷驱动的异常生成。<https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11002_ECCV_2024_paper.php>
- UniVAD, CVPR 2025：部件感知的正常参考匹配。<https://openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html>
- AnomalyNCD, CVPR 2025：缺陷中心化和掩码引导的多类异常表示。<https://openaccess.thecvf.com/content/CVPR2025/html/Huang_AnomalyNCD_Towards_Novel_Anomaly_Class_Discovery_in_Industrial_Scenarios_CVPR_2025_paper.html>
- ReMP-AD, ICCV 2025：正常参考检索去噪和视觉语言先验融合。<https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html>
- SeaS, ICCV 2025：正常/异常属性解耦的工业图像生成。<https://openaccess.thecvf.com/content/ICCV2025/html/Dai_SeaS_Few-shot_Industrial_Anomaly_Image_Generation_with_Separation_and_Sharing_ICCV_2025_paper.html>
- FastRef, CVPR 2026：查询感知的正常原型快速细化和异常抑制。<https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html>
- DCP-SFR, CVPR 2026：保留浅层微弱缺陷线索的结构特征细化。<https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html>
