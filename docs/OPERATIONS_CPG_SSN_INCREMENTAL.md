# CPG-SSN 严格无回放迭代训练与推理操作手册

本文档用于在 GPU 服务器上复刻当前的 CPG-SSN 严格无回放迭代实验，并使用最终模型执行单图或批量 OK/NG 推理。当前任务只输出整图二分类结果：`OK` 或 `NG`，不输出缺陷类别、位置或偏移量。

当前实验的关键约束是：每个新阶段加载上一阶段权重，但只使用本阶段新增的 NG 图像训练；历史 NG 不回放；80 张训练 OK 始终固定。H01 和 V010 从 ImageNet 预训练的 ResNet-18 基础模型开始，**不使用 S08 权重**。

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

所有数据、环境、模型和实验结果均放在 `/root` 下，不要放入 `/hy-tmp`。

## 2 当前实验环境

已完成实验记录的核心环境如下：

| 项目 | 版本或配置 |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| PyTorch CUDA | 12.8 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| SciPy | 1.17.1 |
| 训练 GPU | NVIDIA GeForce RTX 5080 |
| 项目提交 | `45bdce07c0ea3fc415811a79d076647405e26d6d` 或包含本文档的更新提交 |
| SuperSimpleNet 提交 | `98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d` |

项目的 `requirements.txt` 不包含 PyTorch，且其中部分通用依赖与已经完成实验的实际环境并非完全锁定。要做严格复现，优先使用现有 `/root/chip_env`；新服务器应安装与上表一致的 PyTorch 和 torchvision，再安装仓库依赖。不要在现有环境中直接执行不受控的全量升级。

登录服务器后先检查环境：

```bash
ssh -p 45024 root@i-2.gpushare.com

source /root/chip_env/bin/activate
cd /root/program

python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
git rev-parse HEAD
git -C /root/SuperSimpleNet rev-parse HEAD
```

如果输出 `NO GPU`，脚本会自动退回 CPU，但训练时间和推理速度将无法复现现有结果。正式训练前应先为服务器挂载 GPU，并确认 `torch.cuda.is_available()` 为 `True`。

## 3 首次部署项目

已有服务器可以跳过本节。新服务器的基本流程如下，密码和访问令牌不要写入仓库。

```bash
cd /root
git clone https://github.com/RSObjectDetection/program.git /root/program

git clone https://github.com/blaz-r/SuperSimpleNet.git /root/SuperSimpleNet
git -C /root/SuperSimpleNet checkout 98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d
git -C /root/SuperSimpleNet apply /root/program/patches/supersimplenet_py38.patch
```

补丁会修改 `/root/SuperSimpleNet/model/feature_extractor.py` 和 `/root/SuperSimpleNet/model/supersimplenet.py`，因此该仓库显示这两个文件为已修改属于正常现象。不要再次重复应用补丁。

创建环境时，先安装适配服务器 GPU 的 PyTorch，再安装仓库依赖：

```bash
python3 -m venv /root/chip_env
source /root/chip_env/bin/activate

# 先安装与目标 CUDA 对应的 torch 和 torchvision。
# 严格复现当前结果需要 torch 2.11.0+cu128 和 torchvision 0.26.0+cu128。

python -m pip install -r /root/program/requirements.txt
```

首次运行 ResNet-18 时需要能够下载 ImageNet 预训练权重，或提前将相同权重放入 PyTorch 缓存。H01/V10 使用的是 ImageNet 预训练骨干，而不是从随机权重训练 ResNet-18。

## 4 数据集构建

### 4.1 原始目录结构

数据准备脚本要求原始数据至少满足：

```text
/root/chip_project/data/raw/
├── Annotations/
│   └── .../*.xml
└── images/
    └── .../*.jpg
```

每个 XML 必须只对应一种缺陷，图像文件名需保留当前规则。脚本将文件名前缀解析为模板编号，将末尾编号解析为样本编号。

### 4.2 生成合成 OK 和数据清单

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

脚本把同一模板、同一样本编号下的六类缺陷图按像素通道取中位数，生成一张候选 OK。一个合成 OK 与对应的六张 NG 始终进入同一数据划分，避免近重复图像跨训练集和测试集泄漏。

训练 NG 仅保留包含标注框中心的图块；验证集和测试集保留整图固定网格，以整图 Top-2 分数进行最终判断。输出包括：

- `tiles.csv`：训练、验证、测试实际读取的图块坐标和标签；
- `images.csv`：整图级清单；
- `normal_images/`：合成 OK 图像；
- `summary.json`：图像和图块数量；
- `normal_quality.json`：合成 OK 的像素差异统计。

当前数据集共 805 张整图：训练 80 OK + 480 NG，验证 15 OK + 90 NG，测试 20 OK + 120 NG。六类 NG 在每个划分中均匀分布。

如果输出目录已存在，脚本会拒绝覆盖。不要为了方便直接在正式目录使用 `--force`；应先使用新的输出目录构建并检查，确认无误后再切换路径。

### 4.3 数据审计

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

## 5 严格无回放迭代方案

### 5.1 横向类别迭代

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

### 5.2 纵向数据迭代

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

## 6 完整复现实验

### 6.1 推荐的独立复现命令

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

### 6.2 当前实验参数

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

## 7 监控、断点续跑和结果检查

### 7.1 查看运行状态

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

### 7.2 断点续跑规则

使用完全相同的命令和 `--output-root` 重新启动即可续跑：

- 同时存在 `experiment_summary.json` 和 `weights.pt` 的阶段会自动跳过；
- 未生成 `experiment_summary.json` 的中断阶段会从该阶段开头重新训练，不是从中断 epoch 恢复；
- 后续阶段会继续加载上一已完成阶段的 `weights.pt`；
- 如果有 `experiment_summary.json` 但缺少 `weights.pt`，程序会报错，应先恢复完整文件或将该阶段移到备份目录后重跑；
- 不要为了断点续跑添加 `--force`。

### 7.3 输出文件

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

## 8 阈值和混淆矩阵

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

## 9 单图推理

### 9.1 推荐模型

当前六类缺陷整图 OK/NG 判断优先使用 H06，因为它是横向类别迭代的最终模型，当前测试结果最好。权重、原型和阈值摘要必须来自同一阶段目录：

```text
/root/program_runs/iterative_no_replay/horizontal/seed_20260910/H06_spurious_copper/
├── weights.pt
├── prototypes.pt
└── experiment_summary.json
```

不要混用 H06 权重与 V100/S08 原型。V100 主要用于研究分批增加训练样本的纵向实验。

### 9.2 确定模板编号

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

### 9.3 推理一张图片

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

### 9.4 批量推理

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

### 9.5 没有 prototypes.pt 时

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

## 10 推理参数

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

## 11 常见问题

### 11.1 Multiple templates are available

原因：原型库包含多个模板但未传 `--template`。根据产品配方或数据文件名前缀补充，例如 `--template 01`。

### 11.2 Unknown template

原因：输入模板不在原型库中。先查询模板列表；如果是新产品模板，需要收集该模板的正常参考图并重建原型，不能随意使用其他模板。

### 11.3 找不到自动阈值

`weights.pt` 同目录必须有 `experiment_summary.json`。如果移动权重，应同时复制摘要文件；否则显式传入经验证集标定的 `--threshold`。

### 11.4 CUDA out of memory

先把训练 `--batch 16` 或推理 `--batch 32` 调小。这样可以完成运行，但不再是完全一致的性能复现；应记录新 batch 和 GPU 型号。

### 11.5 DataLoader worker 异常

将训练 `--workers` 临时改为 0 排查。推理默认的新流程主要使用 `--preprocess-workers`，不依赖 DataLoader worker。

### 11.6 tiles.csv 中图像不存在

`tiles.csv` 保存绝对路径。迁移服务器后，应让数据保持在相同 `/root/chip_project/data/...` 路径，或重新运行数据准备脚本生成清单。不要只复制 CSV 而不复制对应图片。

### 11.7 结果与参考值略有差异

依次核对项目提交、SuperSimpleNet 提交和补丁、seed、数据审计摘要、PyTorch/CUDA、GPU 型号、batch、worker、类别顺序及 `train_rank`。当前只运行了一个随机种子，少量数值差异不应直接解释为模型退化。

## 12 复现验收清单

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
