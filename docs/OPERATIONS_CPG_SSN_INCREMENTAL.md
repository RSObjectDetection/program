# CPG-SSN 开发配置、数据制作、实验复现与推理操作手册

本文档用于从一台空的 GPU 服务器开始，完成 CPG-SSN 项目部署、环境安装、正常样本与数据集制作、S01–S09 对比实验、累计/严格无回放迭代实验，以及单图或批量 OK/NG 推理。当前任务只输出整图二分类结果：`OK` 或 `NG`，不输出缺陷类别、位置或偏移量。

当前实验的关键约束是：每个新阶段加载上一阶段权重，但只使用本阶段新增的 NG 图像训练；历史 NG 不回放；80 张训练 OK 始终固定。H01 和 V010 从 ImageNet 预训练的 ResNet-18 基础模型开始，**不使用 S08 权重**。

推荐按以下顺序操作：

1. 恢复项目包或克隆两个仓库；
2. 创建 `/root/chip_env` 并安装锁定依赖；
3. 准备原始 NG 数据，并生成或导入 OK；
4. 构建一个全新的版本化数据目录，检查可视化和数据审计；
5. 先执行 S01、S04、S08 等单次对比实验，再执行累计或严格无回放迭代实验；
6. 使用验证集确定阈值，最后使用同一阶段的权重、原型和配置推理。

## 1 目录和文件约定

当前服务器使用以下路径。复现实验时建议保持一致，避免 `tiles.csv` 中保存的绝对图像路径失效。

| 内容 | 服务器路径 |
|---|---|
| 项目仓库 | `/root/program` |
| Python 环境 | `/root/chip_env` |
| SuperSimpleNet 官方仓库 | `/root/SuperSimpleNet` |
| 原始数据根目录 | `/root/chip_project/data/raw` |
| 已构建数据集 | `/root/chip_project/data/okng_512_o128` |
| 训练清单 | `/root/chip_project/data/okng_512_o128/tiles.csv` |
| 已完成无回放实验 | `/root/program_runs/iterative_no_replay` |
| 横向最终模型 H06 | `/root/program_runs/iterative_no_replay/horizontal/seed_20260910/H06_spurious_copper` |
| 纵向最终模型 V100 | `/root/program_runs/iterative_no_replay/vertical/seed_20260910/V100` |
| 用户待测图片目录 | `/root/test` |
| S01–S09 对比实验 | `/root/program_runs/s01_s09` |
| 项目归档包 | `/root/cpg_ssn_project_20260930.tar.gz` |

所有数据、环境、模型和实验结果均放在 `/root` 下，不要放入 `/hy-tmp`。

## 2 开发环境和软件版本

已完成实验记录的核心环境如下：

| 项目 | 版本或配置 |
|---|---|
| 操作系统 | Ubuntu 20.04.5 LTS，Linux 5.15 |
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| PyTorch CUDA | 12.8 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| SciPy | 1.17.1 |
| 训练 GPU | NVIDIA GeForce RTX 5080 |
| 项目提交 | 以 `git rev-parse HEAD` 和各实验的 `command.json` 为准 |
| SuperSimpleNet 提交 | `98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d` |

本项目现在提供两个依赖文件：

- `requirements-cpg-ssn-cu128.txt`：当前 CPG-SSN 训练和推理的锁定环境，严格复现应使用它；
- `requirements.txt`：同时覆盖早期 YOLO、ONNX 等通用实验，不应用于覆盖现有 CPG-SSN 环境。

当前修订后的 SuperSimpleNet 路径只依赖 PyTorch、torchvision、NumPy、Pillow、scikit-learn、SciPy 和仓库内置 Perlin 实现，不需要安装上游 `requirements.txt` 中的旧版 `anomalib==0.7`。不要混装 CUDA 11.8 的上游依赖与当前 CUDA 12.8 环境。

登录服务器后先检查环境：

```bash
ssh -p 45024 root@i-2.gpushare.com

source /root/chip_env/bin/activate
cd /root/program

python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
git rev-parse HEAD
git -C /root/SuperSimpleNet rev-parse HEAD
```

如果输出 `NO GPU`，脚本会自动退回 CPU，但训练时间和推理速度将无法复现现有结果。正式训练前应先为服务器挂载 GPU，并确认 `torch.cuda.is_available()` 为 `True`。多 GPU 服务器可在启动前设置 `export CUDA_VISIBLE_DEVICES=0`；当前程序按单 GPU 设计，不是分布式训练。

## 3 安装包、部署和配置

### 3.1 从现有项目压缩包恢复

当前精简归档为 `/root/cpg_ssn_project_20260930.tar.gz`，大小约 1.8 GiB，SHA-256 为：

```text
4adc0c3e765770c4abfb6a9ebc1dac082a4c846b83ac420cd31589a5ab12e64a
```

归档包含 `/root/program`、`/root/SuperSimpleNet`、`/root/chip_project`、已完成的严格无回放实验和 `/root/test`，不包含可重新创建的 `/root/chip_env`、下载缓存 `/root/.cache` 和原始压缩包副本。因此 4.1 GiB 的缓存和 7.1 GiB 的虚拟环境不需要随项目迁移。

在新服务器上传归档后执行：

```bash
cd /root
echo "4adc0c3e765770c4abfb6a9ebc1dac082a4c846b83ac420cd31589a5ab12e64a  /root/cpg_ssn_project_20260930.tar.gz" | sha256sum -c -
tar -tzf /root/cpg_ssn_project_20260930.tar.gz | head
tar -xzf /root/cpg_ssn_project_20260930.tar.gz -C /
git -C /root/program pull --ff-only
```

解压前应确认目标服务器的 `/root/program`、`/root/SuperSimpleNet` 等目录没有需要保留的同名文件。归档不会恢复 Python 环境，仍需执行 3.3。

### 3.2 从 Git 仓库安装

没有归档时使用以下流程。密码和访问令牌不要写入仓库或命令日志。

```bash
cd /root
git clone https://github.com/RSObjectDetection/program.git /root/program

git clone https://github.com/blaz-r/SuperSimpleNet.git /root/SuperSimpleNet
git -C /root/SuperSimpleNet checkout 98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d
git -C /root/SuperSimpleNet apply --check /root/program/patches/supersimplenet_py38.patch
git -C /root/SuperSimpleNet apply /root/program/patches/supersimplenet_py38.patch
```

补丁会修改 `/root/SuperSimpleNet/model/feature_extractor.py` 和 `/root/SuperSimpleNet/model/supersimplenet.py`，因此该仓库显示这两个文件为已修改属于正常现象。不要再次重复应用补丁。

### 3.3 创建 Python 环境和安装依赖

服务器需要预先具有 Python 3.11、`venv`、Git、NVIDIA 驱动及能够支持 CUDA 12.8 PyTorch 的 GPU。创建环境后只安装专用锁定清单：

```bash
python3.11 -m venv /root/chip_env
source /root/chip_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r /root/program/requirements-cpg-ssn-cu128.txt
```

安装完成后执行一次完整自检：

```bash
python - <<'PY'
import numpy, PIL, scipy, sklearn, torch, torchvision

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("Pillow", PIL.__version__)
print("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
PY
```

首次运行 ResNet-18 或 WideResNet-50-2 时需要下载 ImageNet 预训练权重。联网服务器可提前缓存：

```bash
python - <<'PY'
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)

resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
print("ImageNet weights are ready")
PY
```

严格无回放实验的 H01/V010 使用 ImageNet 预训练骨干，不是随机初始化，也不是 S08 训练权重。特征提取器在训练中冻结。

### 3.4 路径和运行配置检查

当前项目不依赖统一 YAML 配置，运行参数以命令行和每个实验目录内的 `command.json` 为准。启动实验前检查：

```bash
test -f /root/program/scripts/run_cpg_ssn.py
test -f /root/program/scripts/run_no_replay_experiments.py
test -f /root/SuperSimpleNet/model/supersimplenet.py
test -f /root/chip_project/data/okng_512_o128/tiles.csv
mkdir -p /root/program_runs /root/test

git -C /root/program rev-parse HEAD
git -C /root/SuperSimpleNet rev-parse HEAD
git -C /root/SuperSimpleNet status --short
```

SuperSimpleNet 只应显示预期补丁文件为修改状态。所有输出根目录显式写为 `/root/...`；不要使用 `/hy-tmp`，也不要把实验输出写进 Git 仓库。

## 4 正常样本和数据集制作

### 4.1 原始目录结构

数据准备脚本要求原始数据至少满足：

```text
/root/chip_project/data/raw/
├── Annotations/
│   └── .../*.xml
└── images/
    └── .../*.jpg
```

每个 XML 必须只对应一种缺陷，图像文件名需保留当前规则。图像与 XML 的类别子目录名称必须一致，XML 中的 `filename`、宽高和标注框必须有效。脚本将文件名前缀解析为模板编号，将末尾编号解析为样本编号，例如：

```text
01_spur_16.jpg
│  │    └─ 样本编号 16
│  └────── 缺陷类别 spur
└───────── 模板编号 01
```

当前六类名称必须为 `missing_hole`、`mouse_bite`、`open_circuit`、`short`、`spur`、`spurious_copper`。同一个“模板编号 + 样本编号”必须具有六张尺寸一致、空间对齐的缺陷变体，才能生成一张合成 OK；不完整组会被排除。

### 4.2 当前实验如何生成正常样本

当前原始数据没有真实 OK。数据准备脚本对同一模板、同一样本编号下的六类缺陷图逐像素、逐 RGB 通道排序，取中间两个值的均值，得到一张候选正常图。稀疏缺陷在六张图中的位置通常不同，因此中位数能够保留共同底图并抑制异常区域。

生成后的正常图自动存放在：

```text
/root/chip_project/data/okng_512_o128/normal_images/
├── train/<pair_key>_ok.jpg
├── val/<pair_key>_ok.jpg
└── test/<pair_key>_ok.jpg
```

训练脚本不扫描文件夹，而是严格读取 `tiles.csv`。因此不要只把正常图复制进 `normal_images/`；图片路径、`split`、`label=0`、模板、切块坐标和 `prototype_key` 都必须同时写入清单。最安全的做法始终是运行数据准备脚本重新生成整个版本化数据目录。

正常样本在当前方法中有三项作用：

1. 作为 OK 负类，约束判别头降低误检；
2. 仅使用训练划分的 OK 特征构建按模板和空间位置组织的正常原型库；
3. 使用验证 OK 与验证 NG 标定阈值，测试 OK 只用于最终评估。

当前固定使用 80 张训练 OK、15 张验证 OK 和 20 张测试 OK。原型不是一张“平均图片”，而是 80 张训练 OK 经冻结 ResNet-18 提取后形成的多模板、多位置特征均值。

### 4.3 构建当前 80/15/20 数据集

以下命令复刻当前的 80/15/20 分组、512 像素切块和 128 像素重叠设置：

```bash
source /root/chip_env/bin/activate
cd /root/program

python scripts/prepare_okng_dataset.py \
  --dataset-root /root/chip_project/data/raw \
  --output-root /root/chip_project/data/okng_512_o128 \
  --tile-size 512 \
  --overlap 128 \
  --train-groups 80 \
  --val-groups 15 \
  --test-groups 20 \
  --seed 20260910 \
  --jpeg-quality 95
```

一个合成 OK 与对应的六张 NG 始终进入同一数据划分，避免近重复底图跨训练集和测试集泄漏。划分单位是 `pair_key`，不是切块，也不是单张缺陷图。

训练 NG 仅保留包含标注框中心的图块；验证集和测试集保留整图固定网格，以整图 Top-2 分数进行最终判断。输出包括：

- `tiles.csv`：训练、验证、测试实际读取的图块坐标和标签；
- `images.csv`：整图级清单；
- `normal_images/`：合成 OK 图像；
- `summary.json`：图像和图块数量；
- `normal_quality.json`：合成 OK 的像素差异统计。

`tiles.csv` 是训练和复现的唯一数据入口，关键字段如下：

| 字段 | 含义 |
|---|---|
| `split` | `train`、`val` 或 `test` |
| `image` | 原图绝对路径 |
| `label` | OK 为 0，NG 为 1 |
| `source_class` | NG 缺陷类别；正常图为 `ok` |
| `pair_key` | 同底图六类 NG 与合成 OK 的分组键 |
| `template` | 位置原型和推理使用的模板编号 |
| `train_rank` | 纵向迭代与 shot 截断使用的训练组次序 |
| `x/y/w/h` | 512 图块在原图中的坐标和尺寸 |
| `prototype_key` | 模板与空间位置组成的原型键 |
| `contains_defect` | XML 框中心是否落入当前图块 |
| `synthetic_ok` | 合成 OK 为 1，真实 OK 应为 0 |

当前数据集共 805 张整图：训练 80 OK + 480 NG，验证 15 OK + 90 NG，测试 20 OK + 120 NG。六类 NG 在每个划分中均匀分布。

如果输出目录已存在，脚本会拒绝覆盖。不要为了方便直接在正式目录使用 `--force`；应先使用新的输出目录构建并检查，确认无误后再切换路径。

### 4.4 使用真实 OK 的推荐方式

合成 OK 适合当前可行性验证，但可能残留缺陷、产生中位数纹理或与真实产线分布不一致。正式部署优先采集真实良品，建议原始文件按模板放置：

```text
/root/chip_project/data/real_ok_raw/
├── 01/*.jpg
├── 04/*.jpg
└── ...
```

真实 OK 应满足与 NG 完全相同的相机、镜头、光照、曝光、分辨率、工位和产品模板。采集时至少覆盖正常位置偏差、亮度波动、不同生产批次和允许范围内的外观变化。人工复核后，以生产批次或拍摄批次为组划分 train/val/test，禁止同一连拍序列跨划分。

当前 `prepare_okng_dataset.py` 尚未提供 `--real-ok-dir` 参数，因此真实 OK 不能仅靠复制文件自动接入。接入时应创建新目录，例如 `/root/chip_project/data/okng_real_v1`，并由数据转换程序生成完整 `images.csv` 与 `tiles.csv`；不要修改或覆盖合成数据集。生成后检查：

- 真实 OK 行的 `label` 为 0、`synthetic_ok` 为 0；
- 每行 `template` 与产品模板一致；
- OK 与 NG 使用相同 512/128 网格和 256 网络输入；
- 训练 OK 只参与训练和原型，验证 OK 只用于阈值，测试 OK 仅用于最终报告；
- 训练、验证、测试的来源批次没有交叉。

如果只能人工覆盖缺陷区域制作少量 OK，应从同模板、同工位的正常供体图复制对应区域，对掩膜边缘做扩张与羽化，并逐张检查接缝、亮度差和残余缺陷。该方法只能作为补充，不能替代真实 OK；当前官方复现仍以六图中位数为准。

### 4.5 数据质量检查

先生成带 Pascal VOC 框的原始数据概览：

```bash
python scripts/visualize_samples.py \
  --dataset-root /root/chip_project/data/raw \
  --output /root/chip_project/data/raw_sample_grid.jpg \
  --thumb-width 640
```

然后检查数据摘要和合成质量：

```bash
python -m json.tool /root/chip_project/data/okng_512_o128/summary.json
python -m json.tool /root/chip_project/data/okng_512_o128/normal_quality.json | head -n 80
find /root/chip_project/data/okng_512_o128/normal_images -type f | wc -l
```

人工至少抽查每个模板的 train/val/test 合成 OK 与对应六张 NG，确认图像尺寸一致、芯片没有明显错位、中位数图没有残留缺陷和鬼影。若同组图像没有像素级对齐，应先做配准；不能直接依赖中位数合成。

### 4.6 数据审计

只生成计划和审计文件、不启动训练：

```bash
python scripts/run_no_replay_experiments.py \
  --manifest /root/chip_project/data/okng_512_o128/tiles.csv \
  --official-repo /root/SuperSimpleNet \
  --output-root /root/program_runs/iterative_no_replay_audit \
  --tracks horizontal vertical \
  --seed 20260910 \
  --audit-only
```

检查 `/root/program_runs/iterative_no_replay_audit/no_replay_audit.json`：

- `valid` 必须为 `true`；
- `fixed_ok_images` 必须为 80；
- H01–H06 和 V010–V100 的 `historical_ng_overlap` 必须全部为 0；
- V010–V100 每阶段 `new_ng_images_per_class` 必须为 8。

数据审计通过并不代表合成 OK 一定真实，只能证明数量、划分和严格无回放约束正确。视觉质检和真实产线盲测仍然必需。

## 5 S01–S09 对比实验

### 5.1 对比原则和实验矩阵

除明确改变的单一变量外，S01–S04、S06 和 S08 应保持相同的 80/15/20 划分、随机种子、12 epochs、每轮 2000 图块、batch 16 和验证选阈值规则。S07 是独立的 100/5/10 小样本曲线，其测试集不同，不能直接与其他实验按 FP/FN 横向排名。

| 实验 | 唯一主要变化 | Backbone | 原型 | GLASS | 数据设置 |
|---|---|---|---:|---:|---|
| S01 | SuperSimpleNet 基线 | WideResNet-50-2 | 否 | 否 | 80/15/20，overlap 128 |
| S02 | 加位置正常原型 | WideResNet-50-2 | 是 | 否 | 同 S01 |
| S03 | 加困难特征扰动 | WideResNet-50-2 | 否 | 是 | 同 S01 |
| S04 | 完整 CPG-SSN | WideResNet-50-2 | 是 | 是 | 同 S01 |
| S05 | Max、Top-2、Top-3 聚合消融 | 复用 S04 | 复用 | 复用 | 不重新训练 |
| S06 | 减小滑窗重叠 | WideResNet-50-2 | 是 | 是 | overlap 64 |
| S07 | 30/50/100-shot 曲线 | WideResNet-50-2 | 是 | 是 | 100/5/10 独立划分 |
| S08 | 轻量骨干 | ResNet-18 | 是 | 是 | 同 S01 |
| S09 | 阈值操作点校准 | 复用 S08 | 复用 | 复用 | 不重新训练 |

### 5.2 运行 S01–S04 和 S08

先定义公共参数，再逐个运行。每个输出目录必须唯一：

```bash
source /root/chip_env/bin/activate
cd /root/program
mkdir -p /root/program_runs/s01_s09

COMMON_ARGS=(
  --manifest /root/chip_project/data/okng_512_o128/tiles.csv
  --official-repo /root/SuperSimpleNet
  --image-size 256
  --epochs 12
  --batch 16
  --workers 6
  --seed 20260910
  --max-train-samples 2000
  --eval-every 3
  --patience 4
  --benchmark-iters 20
)

python scripts/run_cpg_ssn.py "${COMMON_ARGS[@]}" \
  --output-dir /root/program_runs/s01_s09/S01_baseline \
  --name S01_baseline --backbone wide_resnet50_2

python scripts/run_cpg_ssn.py "${COMMON_ARGS[@]}" \
  --output-dir /root/program_runs/s01_s09/S02_prototype \
  --name S02_prototype --backbone wide_resnet50_2 --prototype

python scripts/run_cpg_ssn.py "${COMMON_ARGS[@]}" \
  --output-dir /root/program_runs/s01_s09/S03_glass \
  --name S03_glass --backbone wide_resnet50_2 --glass

python scripts/run_cpg_ssn.py "${COMMON_ARGS[@]}" \
  --output-dir /root/program_runs/s01_s09/S04_cpg_ssn \
  --name S04_cpg_ssn --backbone wide_resnet50_2 --prototype --glass

python scripts/run_cpg_ssn.py "${COMMON_ARGS[@]}" \
  --output-dir /root/program_runs/s01_s09/S08_resnet18 \
  --name S08_resnet18 --backbone resnet18 --prototype --glass
```

### 5.3 运行 S05 聚合消融

一次训练会同时保存 Max、Top-2 和 Top-3 的整图预测与指标。S05 直接读取 S04，不应为了聚合方式重新训练：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/root/program_runs/s01_s09/S04_cpg_ssn/experiment_summary.json")
summary = json.loads(path.read_text(encoding="utf-8"))
for mode in ("max", "top2", "top3"):
    metric = summary["test"][mode]
    print(mode, "threshold=", metric["threshold"], "F1=", metric["f1"],
          "recall=", metric["recall_ng"], "specificity=", metric["specificity_ok"],
          "confusion=", metric["confusion"])
PY
```

只能根据验证集选择聚合方式；测试集结果用于最终报告，不能看完测试表现再反向选 Top-2 或 Top-3。

### 5.4 运行 S06 Overlap 消融

先用新的目录生成 overlap 64 数据集，再运行完整模型：

```bash
python scripts/prepare_okng_dataset.py \
  --dataset-root /root/chip_project/data/raw \
  --output-root /root/chip_project/data/okng_512_o64 \
  --tile-size 512 --overlap 64 \
  --train-groups 80 --val-groups 15 --test-groups 20 \
  --seed 20260910 --jpeg-quality 95

python scripts/run_cpg_ssn.py \
  --manifest /root/chip_project/data/okng_512_o64/tiles.csv \
  --official-repo /root/SuperSimpleNet \
  --output-dir /root/program_runs/s01_s09/S06_overlap64 \
  --name S06_overlap64 --backbone wide_resnet50_2 \
  --image-size 256 --epochs 12 --batch 16 --workers 6 \
  --seed 20260910 --max-train-samples 2000 \
  --eval-every 3 --patience 4 --benchmark-iters 20 \
  --prototype --glass
```

### 5.5 运行 S07 小样本曲线

S07 使用全部 115 个完整组中的 100/5/10 划分，并从 100 个训练组内按 `train_rank` 取得嵌套的 30、50、100 张每类 NG：

```bash
python scripts/prepare_okng_dataset.py \
  --dataset-root /root/chip_project/data/raw \
  --output-root /root/chip_project/data/okng_512_o128_s07_100_5_10 \
  --tile-size 512 --overlap 128 \
  --train-groups 100 --val-groups 5 --test-groups 10 \
  --seed 20260910 --jpeg-quality 95

for SHOT in 30 50 100; do
  python scripts/run_cpg_ssn.py \
    --manifest /root/chip_project/data/okng_512_o128_s07_100_5_10/tiles.csv \
    --official-repo /root/SuperSimpleNet \
    --output-dir "/root/program_runs/s01_s09/S07_shot${SHOT}" \
    --name "S07_shot${SHOT}" --backbone wide_resnet50_2 \
    --image-size 256 --epochs 12 --batch 16 --workers 6 \
    --seed 20260910 --shots-per-class "${SHOT}" \
    --max-train-samples 2000 --eval-every 3 --patience 4 \
    --benchmark-iters 20 --prototype --glass
done
```

### 5.6 运行 S09 阈值校准并汇总

S09 复用 S08 保存的验证/测试分数，输出 F1 最优、验证零漏检和验证零误检三种操作点：

```bash
python scripts/calibrate_okng_thresholds.py \
  --run-dir /root/program_runs/s01_s09/S08_resnet18 \
  --aggregation top2 \
  --output /root/program_runs/s01_s09/S09_calibration/top2.json

python scripts/calibrate_okng_thresholds.py \
  --run-dir /root/program_runs/s01_s09/S08_resnet18 \
  --aggregation top3 \
  --output /root/program_runs/s01_s09/S09_calibration/top3.json

python scripts/collect_s01_s09.py \
  --runs-root /root/program_runs/s01_s09 \
  --output-root /root/program/artifacts/S01-S09_repro
```

权重仍保留在 `/root/program_runs/s01_s09`，收集脚本只把轻量 JSON、CSV 和日志复制进仓库结果目录。

### 5.7 历史参考结果

| 实验 | F1 | NG 召回 | OK 特异度 | FP/FN | 参数量 | 单个 256 图块延迟 |
|---|---:|---:|---:|---:|---:|---:|
| S01 | 0.9958 | 0.9917 | 1.0000 | 0/1 | 33.72M | 2.397 ms |
| S02 | 0.9794 | 0.9917 | 0.8000 | 4/1 | 33.72M | 2.433 ms |
| S03 | 0.9917 | 0.9917 | 0.9500 | 1/1 | 33.72M | 2.393 ms |
| S04 | 0.9877 | 1.0000 | 0.8500 | 3/0 | 33.72M | 2.419 ms |
| S06 | 1.0000 | 1.0000 | 1.0000 | 0/0 | 33.72M | 2.443 ms |
| S08 | 1.0000 | 1.0000 | 1.0000 | 0/0 | 4.56M | 0.709 ms |

这些延迟来自历史 RTX 4090 单块基准，不能直接与不同 GPU、不同 batch 或整图端到端耗时比较。S08 是当前轻量部署候选，但全部 OK 都是合成样本，因此结果仍是封闭数据可行性结论。

## 6 累计训练与严格无回放训练对比

两个迭代脚本回答不同问题：

| 协议 | 脚本 | H02 的训练数据 | V020 的训练数据 | 用途 |
|---|---|---|---|---|
| 累计训练 | `run_iterative_experiments.py` | 第一、二类全部 NG + 固定 OK | 每类前 16 张 NG + 固定 OK | 有历史数据可回放时的性能上限 |
| 严格无回放 | `run_no_replay_experiments.py` | 只训练第二类 NG + 固定 OK | 每类第 9–16 张 NG + 固定 OK | 验证连续学习和灾难性遗忘 |

两者都从 ImageNet ResNet-18 开始，后一阶段加载前一阶段权重；区别只在历史 NG 是否再次进入训练。要公平比较，必须使用同一 `tiles.csv`、seed、epochs、batch、类别顺序和测试集。

累计训练复现命令为：

```bash
source /root/chip_env/bin/activate
cd /root/program
mkdir -p /root/program_runs/iterative_cumulative_repro_20260910

nohup /root/chip_env/bin/python \
  /root/program/scripts/run_iterative_experiments.py \
  --manifest /root/chip_project/data/okng_512_o128/tiles.csv \
  --official-repo /root/SuperSimpleNet \
  --output-root /root/program_runs/iterative_cumulative_repro_20260910 \
  --tracks horizontal vertical \
  --seed 20260910 --epochs 12 --batch 16 --workers 6 \
  --max-train-samples 2000 --benchmark-iters 20 \
  --incremental-lr-scale 0.5 \
  > /root/program_runs/iterative_cumulative_repro_20260910/main_seed_20260910.log \
  2>&1 &

echo $! | tee /root/program_runs/iterative_cumulative_repro_20260910/runner.pid
```

累计结果写入 `iterative_summary_seed_20260910.csv` 和 `forgetting_seed_20260910.csv`；严格无回放结果见第 7–9 节。对比时同时报告自适应阈值和固定首阶段阈值，避免把分数尺度漂移误判为模型遗忘。

## 7 严格无回放迭代方案

### 7.1 横向类别迭代

横向实验验证“新增一种缺陷后，旧缺陷是否仍能识别”。每阶段始终训练固定 80 张 OK，并且只训练当前新类别的 80 张 NG。

| 阶段 | 初始化权重 | 本阶段训练 NG | 测试 NG 范围 |
|---|---|---|---|
| H01 | ImageNet ResNet-18 基础模型 | `missing_hole` | `missing_hole` |
| H02 | H01 | `mouse_bite` | H01–H02 两类 |
| H03 | H02 | `open_circuit` | H01–H03 三类 |
| H04 | H03 | `short` | H01–H04 四类 |
| H05 | H04 | `spur` | H01–H05 五类 |
| H06 | H05 | `spurious_copper` | 全部六类 |

历史 NG 只用于评估，不参与后续阶段梯度更新。

### 7.2 纵向数据迭代

纵向实验验证“持续加入新的训练样本块后，模型如何变化”。每阶段每类只训练 8 张新 NG，六类合计 48 张；已训练过的数据块不回放。

| 阶段 | 初始化权重 | 本阶段每类 `train_rank` | 本阶段 NG 总数 |
|---|---|---:|---:|
| V010 | ImageNet ResNet-18 基础模型 | 1–8 | 48 |
| V020 | V010 | 9–16 | 48 |
| V030 | V020 | 17–24 | 48 |
| V040 | V030 | 25–32 | 48 |
| V050 | V040 | 33–40 | 48 |
| V060 | V050 | 41–48 | 48 |
| V070 | V060 | 49–56 | 48 |
| V080 | V070 | 57–64 | 48 |
| V090 | V080 | 65–72 | 48 |
| V100 | V090 | 73–80 | 48 |

所有纵向阶段都在同一个固定测试集上评估：20 张 OK + 六类各 20 张 NG。

## 8 严格无回放完整复现实验

### 8.1 推荐的独立复现命令

不要直接覆盖 `/root/program_runs/iterative_no_replay`。使用新的输出根目录可以同时保留原始结果和复现结果。

```bash
source /root/chip_env/bin/activate
cd /root/program

mkdir -p /root/program_runs/iterative_no_replay_repro_20260910

nohup /root/chip_env/bin/python \
  /root/program/scripts/run_no_replay_experiments.py \
  --manifest /root/chip_project/data/okng_512_o128/tiles.csv \
  --official-repo /root/SuperSimpleNet \
  --output-root /root/program_runs/iterative_no_replay_repro_20260910 \
  --tracks horizontal vertical \
  --seed 20260910 \
  --epochs 12 \
  --batch 16 \
  --workers 6 \
  --max-train-samples 2000 \
  --benchmark-iters 20 \
  --incremental-lr-scale 0.5 \
  > /root/program_runs/iterative_no_replay_repro_20260910/main_seed_20260910.log \
  2>&1 &

echo $! | tee /root/program_runs/iterative_no_replay_repro_20260910/runner.pid
```

该命令依次执行 H01–H06，再执行 V010–V100。横向和纵向是两条独立权重链；V010 不从 H06 继续。

只运行其中一条实验轨道时使用：

```bash
# 只运行横向
--tracks horizontal

# 只运行纵向
--tracks vertical
```

### 8.2 当前实验参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `--seed` | 20260910 | 数据顺序、采样和模型随机种子 |
| `--epochs` | 12 | 每阶段最大 epoch 数 |
| `--batch` | 16 | 每个训练 batch 的图块数，不是整图数 |
| `--workers` | 6 | DataLoader 进程数 |
| `--max-train-samples` | 2000 | WeightedRandomSampler 每个 epoch 最多抽取的图块数 |
| `--benchmark-iters` | 20 | 阶段结束后的模型图块延迟测试次数，不影响训练 |
| `--incremental-lr-scale` | 0.5 | H02–H06、V020–V100 相对基础学习率的倍率 |
| `--tracks` | horizontal vertical | 要执行的实验轨道 |
| `--audit-only` | 关闭 | 只审计数据，不训练 |
| `--force` | 关闭 | 强制重跑已完成阶段；会覆盖权重并向现有日志追加内容，通常不建议使用 |

迭代调度脚本还固定传入以下模型参数：

| 模型参数 | 当前值 | 说明 |
|---|---:|---|
| Backbone | ResNet-18 | ImageNet 预训练，特征提取器冻结 |
| 模型输入 | 256×256 | 512×512 原始图块缩放后的网络输入 |
| 图像聚合 | Top-2 | 取整图最高两个图块分数的平均值 |
| `eval_every` | 3 | 每 3 个 epoch 做一次验证 |
| `patience` | 4 | 验证无提升的容忍次数 |
| 位置原型 | 开启 | 8×8 正常特征原型，来自固定 80 张训练 OK |
| GLASS 风格困难化 | 开启 | warm-up 3 epochs 后，每次执行 1 步、步长 0.001 |
| 第一阶段基础学习率 | adaptor 0.0001；seg/decision 0.0002 | AdamW |
| 增量阶段学习率 | adaptor 0.00005；seg/decision 0.0001 | 基础学习率乘 0.5 |

改变 batch、GPU、CUDA 算子或依赖版本可能造成小幅数值差异。改变 seed、数据划分、图块尺寸、类别顺序、训练块顺序或学习率后，不再属于对当前实验的严格复现。

## 9 监控、断点续跑和结果检查

### 9.1 查看运行状态

```bash
cat /root/program_runs/iterative_no_replay_repro_20260910/runner.pid
ps -fp "$(cat /root/program_runs/iterative_no_replay_repro_20260910/runner.pid)"
tail -f /root/program_runs/iterative_no_replay_repro_20260910/main_seed_20260910.log
nvidia-smi
```

每个阶段的独立日志位于：

```text
<output-root>/<horizontal|vertical>/seed_20260910/<stage>/run.log
```

主日志出现以下文本表示全部完成：

```text
All requested strict no-replay stages completed
```

### 9.2 断点续跑规则

使用完全相同的命令和 `--output-root` 重新启动即可续跑：

- 同时存在 `experiment_summary.json` 和 `weights.pt` 的阶段会自动跳过；
- 未生成 `experiment_summary.json` 的中断阶段会从该阶段开头重新训练，不是从中断 epoch 恢复；
- 后续阶段会继续加载上一已完成阶段的 `weights.pt`；
- 如果有 `experiment_summary.json` 但缺少 `weights.pt`，程序会报错，应先恢复完整文件或将该阶段移到备份目录后重跑；
- 不要为了断点续跑添加 `--force`。

### 9.3 输出文件

每个阶段目录包含：

| 文件 | 用途 |
|---|---|
| `weights.pt` | 当前阶段可训练模块权重 |
| `prototypes.pt` | 正常位置原型；推理时应与同阶段权重配套使用 |
| `experiment_summary.json` | 参数、阈值、验证/测试指标、耗时和显存 |
| `best_validation.json` | 最佳验证轮次 |
| `history.json` | epoch 训练历史 |
| `command.json` | 本阶段实际执行命令，可用于排查参数 |
| `run.log` | 本阶段标准输出和错误日志 |
| `test_predictions_top2.csv` | 测试整图分数和预测 |

输出根目录还会生成：

- `no_replay_summary_seed_20260910.csv`：16 个阶段的自适应阈值和固定阈值指标；
- `forgetting_adaptive_seed_20260910.csv`：横向自适应阈值遗忘；
- `forgetting_fixed_seed_20260910.csv`：横向固定阈值遗忘；
- `continual_summary_seed_20260910.json`：横向最终结果；
- `no_replay_audit.json`：无历史 NG 回放审计；
- `dataset_audit.json`：数据数量和泄漏审计。

快速查看核心指标：

```bash
python - <<'PY'
import csv
path = "/root/program_runs/iterative_no_replay_repro_20260910/no_replay_summary_seed_20260910.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for row in rows:
    print(
        row["stage"],
        "F1=", row["adaptive_f1"],
        "recall=", row["adaptive_recall_ng"],
        "specificity=", row["adaptive_specificity_ok"],
        "FP=", row["adaptive_fp"],
        "FN=", row["adaptive_fn"],
    )
PY
```

当前参考结果为：H06 自适应阈值 0.58，F1/NG 召回/OK 特异度均为 1.0000；V100 自适应阈值约 0.5686，F1 为 0.9916、NG 召回为 0.9833、OK 特异度为 1.0000。完整结果见 [严格无回放实验报告](../artifacts/iterative_no_replay/REPORT.md)。

## 10 阈值和混淆矩阵

每个阶段先在验证集上选择使 NG 类 F1 最大的阈值，再把该阈值用于测试集。推理脚本默认从 checkpoint 同目录的 `experiment_summary.json` 中读取所选聚合方式的验证阈值。

汇总表同时提供两种口径：

- 自适应阈值：每阶段使用自己的验证阈值，反映重新标定后的效果；
- 固定阈值：横向始终使用 H01 阈值，纵向始终使用 V010 阈值，用于观察分数尺度漂移。

混淆矩阵定义如下，NG 为正类：

| 字段 | 含义 |
|---|---|
| `TN` | 实际 OK，预测 OK |
| `FP` | 实际 OK，预测 NG，即误检或过杀 |
| `FN` | 实际 NG，预测 OK，即漏检 |
| `TP` | 实际 NG，预测 NG |

当前 OK 均为合成图。上线前必须用真实产线 OK 和独立 NG 批次重新标定阈值，不能把当前测试结果直接当作量产误检率或漏检率承诺。

## 11 单图和批量推理

### 11.1 推荐模型

当前六类缺陷整图 OK/NG 判断优先使用 H06，因为它是横向类别迭代的最终模型，当前测试结果最好。权重、原型和阈值摘要必须来自同一阶段目录：

```text
/root/program_runs/iterative_no_replay/horizontal/seed_20260910/H06_spurious_copper/
├── weights.pt
├── prototypes.pt
└── experiment_summary.json
```

不要混用 H06 权重与 V100/S08 原型。V100 主要用于研究分批增加训练样本的纵向实验。

### 11.2 确定模板编号

当前位置原型按模板和切块坐标保存。当前可用模板为：

```text
01 04 05 06 07 08 09 10 11 12
```

数据集内文件名的第一个下划线前缀就是模板编号，例如 `01_spur_16.jpg` 使用 `--template 01`。对于产线新图片，模板编号应由产品配方、工位或上游型号系统提供；当前推理脚本不会自动识别模板。一个目录中含多个模板时，应先按模板分组，分别运行推理。

可以从清单再次查询模板：

```bash
python - <<'PY'
import csv
path = "/root/chip_project/data/okng_512_o128/tiles.csv"
with open(path, newline="", encoding="utf-8") as f:
    print(sorted({row["template"] for row in csv.DictReader(f)}))
PY
```

### 11.3 推理一张图片

```bash
source /root/chip_env/bin/activate
cd /root/program

python scripts/infer_cpg_ssn.py \
  --image /root/test/01_spur_16.jpg \
  --checkpoint /root/program_runs/iterative_no_replay/horizontal/seed_20260910/H06_spurious_copper/weights.pt \
  --prototypes /root/program_runs/iterative_no_replay/horizontal/seed_20260910/H06_spurious_copper/prototypes.pt \
  --official-repo /root/SuperSimpleNet \
  --template 01 \
  --aggregation top2 \
  --batch 32 \
  --preprocess-workers 4 \
  --output /root/predictions_single.csv
```

未指定 `--threshold` 时，程序读取 H06 的 `experiment_summary.json`，在 `top2` 模式下当前值为 0.58。分数大于或等于阈值判为 NG，否则判为 OK。

如果需要固定使用 H01 阈值做增量稳定性对照，显式增加：

```bash
--threshold 0.69
```

该固定阈值适合实验对照，不代表产线最终阈值。

### 11.4 批量推理

同一目录中的图片必须属于同一模板：

```bash
python scripts/infer_cpg_ssn.py \
  --input-dir /root/test/template_01 \
  --checkpoint /root/program_runs/iterative_no_replay/horizontal/seed_20260910/H06_spurious_copper/weights.pt \
  --prototypes /root/program_runs/iterative_no_replay/horizontal/seed_20260910/H06_spurious_copper/prototypes.pt \
  --official-repo /root/SuperSimpleNet \
  --template 01 \
  --aggregation top2 \
  --batch 32 \
  --preprocess-workers 4 \
  --output /root/predictions_template_01.csv
```

支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif` 和 `.tiff`。`--input-dir` 会递归查找图片。

### 11.5 没有 prototypes.pt 时

可用训练清单临时重建原型：

```bash
python scripts/infer_cpg_ssn.py \
  --image /root/test/01_spur_16.jpg \
  --checkpoint /root/program_runs/iterative_no_replay/horizontal/seed_20260910/H06_spurious_copper/weights.pt \
  --manifest /root/chip_project/data/okng_512_o128/tiles.csv \
  --official-repo /root/SuperSimpleNet \
  --template 01 \
  --output /root/predictions_single.csv
```

重建原型会读取固定训练 OK 并运行特征提取，启动明显更慢。生产推理应保存并加载同阶段 `prototypes.pt`。

## 12 推理参数

| 参数 | 默认值 | 当前建议 | 含义 |
|---|---:|---:|---|
| `--image` | 无 | 单图时必填 | 单张待测图片 |
| `--input-dir` | 无 | 批量时必填 | 待测图片目录，与 `--image` 二选一 |
| `--checkpoint` | 无 | H06 `weights.pt` | 模型权重 |
| `--prototypes` | 无 | H06 `prototypes.pt` | 正常位置原型 |
| `--manifest` | 无 | 仅缺原型时使用 | 重建原型所需 `tiles.csv` |
| `--template` | 无 | 按产品模板填写 | 当前原型库有多个模板，因此必须填写 |
| `--threshold` | 自动读取 | 默认自动读取 | 手工覆盖 OK/NG 阈值 |
| `--aggregation` | `top2` | `top2` | 整图分数采用 max、Top-2 或 Top-3 |
| `--tile-size` | 512 | 512 | 原图切块尺寸，必须与训练一致 |
| `--overlap` | 128 | 128 | 相邻图块重叠，必须与训练一致 |
| `--image-size` | 256 | 256 | 模型输入尺寸，必须与训练一致 |
| `--prototype-size` | 8 | 8 | 原型特征尺寸，必须与训练一致 |
| `--batch` | 32 | 32 | 一次送入 GPU 的图块数 |
| `--preprocess-workers` | 4 | 4 | 解码一次后，在内存中裁剪/缩放的线程数 |
| `--workers` | 4 | 仅 legacy 模式使用 | 旧 DataLoader 的进程数 |
| `--legacy-loader` | 关闭 | 关闭 | 仅用于对比旧版逐图块 JPEG 加载流程 |
| `--output` | `predictions.csv` | 显式指定绝对路径 | 推理结果 CSV |

推理输出中的主要字段：

- `score`：整图 Top-2 异常分数；
- `prediction`：最终 `OK` 或 `NG`；
- `threshold`：本次判定阈值；
- `tile_count`：该图被切成的图块数量；
- `top_tile_scores`：最高三个图块分数；
- `startup_total_ms`：Python 导入、模型构建、权重/原型加载和 CUDA warm-up 的总启动时间；
- `pipeline_total_ms`：单图解码、预处理、模型推理和聚合时间；
- `model_forward_ms`：该图所有图块的纯模型前向时间。

单次命令的总等待时间不等于模型前向时间。若每张图都重新启动 Python，模型加载、原型加载和 CUDA warm-up 会反复发生。工业部署应让模型常驻进程，并连续处理图片队列。

## 13 常见问题

### 13.1 Multiple templates are available

原因：原型库包含多个模板但未传 `--template`。根据产品配方或数据文件名前缀补充，例如 `--template 01`。

### 13.2 Unknown template

原因：输入模板不在原型库中。先查询模板列表；如果是新产品模板，需要收集该模板的正常参考图并重建原型，不能随意使用其他模板。

### 13.3 找不到自动阈值

`weights.pt` 同目录必须有 `experiment_summary.json`。如果移动权重，应同时复制摘要文件；否则显式传入经验证集标定的 `--threshold`。

### 13.4 CUDA out of memory

先把训练 `--batch 16` 或推理 `--batch 32` 调小。这样可以完成运行，但不再是完全一致的性能复现；应记录新 batch 和 GPU 型号。

### 13.5 DataLoader worker 异常

将训练 `--workers` 临时改为 0 排查。推理默认的新流程主要使用 `--preprocess-workers`，不依赖 DataLoader worker。

### 13.6 tiles.csv 中图像不存在

`tiles.csv` 保存绝对路径。迁移服务器后，应让数据保持在相同 `/root/chip_project/data/...` 路径，或重新运行数据准备脚本生成清单。不要只复制 CSV 而不复制对应图片。

### 13.7 结果与参考值略有差异

依次核对项目提交、SuperSimpleNet 提交和补丁、seed、数据审计摘要、PyTorch/CUDA、GPU 型号、batch、worker、类别顺序及 `train_rank`。当前只运行了一个随机种子，少量数值差异不应直接解释为模型退化。

## 14 复现验收清单

完成实验后至少确认：

- [ ] GPU 可用，环境版本已记录；
- [ ] `/root/program` 和 `/root/SuperSimpleNet` 提交正确；
- [ ] `dataset_audit.json` 的 `valid` 为 `true`；
- [ ] `no_replay_audit.json` 的全部历史 NG 重叠为 0；
- [ ] H01–H06 共 6 个 `experiment_summary.json`；
- [ ] V010–V100 共 10 个 `experiment_summary.json`；
- [ ] `no_replay_summary_seed_20260910.csv` 共 16 行结果；
- [ ] 最终权重和原型文件均存在；
- [ ] 推理使用同阶段的权重、原型、摘要和正确模板；
- [ ] 生产验收前已使用真实 OK/NG 独立数据重新标定阈值。

实验结果、遗忘分析和当前局限见 [CPG-SSN 严格无回放迭代训练报告](../artifacts/iterative_no_replay/REPORT.md)。模型原理和模块融合见 [OK/NG SuperSimpleNet 技术方案](TECHNICAL_PLAN_OK_NG_SUPERSIMPLENET.md)。
