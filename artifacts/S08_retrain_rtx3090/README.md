# S08 ResNet-18 CPG-SSN 重训（RTX 3090）

重训日期：2026-09-14。所有原始数据、生成数据、运行结果、权重和原型均位于服务器 `/root`，未使用 `/hy-tmp`。

## 数据与配置

- 原始数据：693 张 NG 图和 693 个 VOC XML。
- 有效同版组：115；按组划分为 80/15/20 个 train/validation/test 组。
- 训练整图：80 张合成 OK、480 张真实 NG。
- 验证整图：15 张合成 OK、90 张真实 NG。
- 测试整图：20 张合成 OK、120 张真实 NG。
- 滑窗：512×512，overlap 128；网络输入 256×256。
- 模型：ResNet-18 CPG-SSN，位置原型 + GLASS 式困难扰动。
- 训练：12 epochs、batch 16、每轮平衡采样 2,000 个图块、随机种子 20260910。
- 硬件：NVIDIA GeForce RTX 3090 24GB。

训练命令：

```bash
python scripts/run_cpg_ssn.py \
  --manifest /root/chip_project/data/okng_512_o128/tiles.csv \
  --official-repo /root/SuperSimpleNet \
  --output-dir /root/program_runs/S08_resnet18 \
  --name S08_resnet18_retrain --backbone resnet18 \
  --image-size 256 --epochs 12 --batch 16 --workers 6 \
  --max-train-samples 2000 --prototype --glass
```

## 核心结果

验证集 Top-2：AUROC/AUPRC/F1/NG Recall/OK Specificity 均为 1.000，FP=0、FN=0；验证阈值为 0.7839441。

| 聚合 | 测试 AUROC | 测试 AUPRC | F1 | NG Recall | OK Specificity | FP/FN |
|---|---:|---:|---:|---:|---:|---:|
| Max | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0/0 |
| Top-2（预设主指标） | 0.9979 | 0.9997 | 0.9958 | 0.9917 | 1.0000 | 0/1 |
| Top-3 | 0.9958 | 0.9993 | 0.9916 | 0.9833 | 1.0000 | 0/2 |

Top-2 唯一漏检为 `04_spurious_copper_03`，分数 0.7657565。三种聚合在验证集均为全对，因此不能根据测试集表现事后把主聚合从 Top-2 改为 Max；Max 的全对结果只作为诊断记录。

- 参数量：4,563,783。
- RTX 3090 单图块平均模型延迟：1.154 ms；P95：1.195 ms。
- 独立推理冒烟测试：一张含 32 个图块的合成 OK 图正确判为 OK。后续已消除 JPEG 重复解码和单图 DataLoader 进程启动；RTX 4090 上优化前后分阶段结果见 [`artifacts/inference_optimization_rtx4090/README.md`](../inference_optimization_rtx4090/README.md)。

## 服务器持久化文件

```text
/root/chip_project/data/raw/
/root/chip_project/data/okng_512_o128/
/root/program_runs/S08_resnet18/weights.pt
/root/program_runs/S08_resnet18/prototypes.pt
/root/program_runs/S08_resnet18/experiment_summary.json
/root/S08_resnet18_retrained_20260914.tar.gz
```

SHA-256：

```text
weights.pt     7dec448f230202380b2d0d753b56043b2227d458b88bf44832a03373e9dd684e
prototypes.pt  56b2d3031f47e4b7ece00a2e4d95cb1fce86a1068c9b1ab3b5381795313432e1
archive        a02e1fcdb5514e45e58a9a8b605c64a1ac84b6decc4decbc1f297fa42e2b3071
```

所有 OK 验证/测试图仍为合成图；这些结果是封闭数据可行性验证，不代表真实产线误报率。
