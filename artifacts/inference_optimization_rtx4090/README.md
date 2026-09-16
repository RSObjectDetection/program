# S08 单图推理耗时优化（RTX 4090）

测试日期：2026-09-16。测试图为 `/root/test/01_02_ok.jpg`，分辨率 3034×1586，经 512×512、overlap 128 的滑窗生成 32 个图块；模型为 S08 ResNet-18 CPG-SSN，输入 256×256，Top-2 阈值 0.7839441001415253。

## 结论

旧脚本每个图块都重新打开并解码同一张 JPEG，且单图也启动 4 个 DataLoader worker。优化后整图只解码一次，在内存中并行裁块与缩放，32 个图块组成一个 batch；模型、原型和 CUDA 上下文在进程内只加载、预热一次。

| 指标（3 次均值） | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 图像推理流水线 | 1304.49 ms | 301.61 ms | -76.9%，4.33× 加速 |
| 一次性 CLI 完整墙钟时间 | 4.663 s | 3.897 s | -16.4%，1.20× 加速 |
| 输出 | OK，0.7831257 | OK，0.7831285 | 标签不变；batch 浮点差 2.8e-6 |

一次性 CLI 的加速幅度较小，是因为 Python/PyTorch 导入、建模、读取权重/原型、CUDA 初始化与进程退出仍然存在。产线应使用常驻进程或一次输入同模板的多张图，避免每张图重复支付启动成本。

## 优化后阶段耗时

三次流水线耗时为 298.45、294.06、312.32 ms，均值 301.61 ms。

| 阶段 | 3 次均值 | 占流水线 |
|---|---:|---:|
| JPEG 单次解码 | 56.37 ms | 18.7% |
| 滑窗坐标生成 | 0.16 ms | <0.1% |
| 32 块并行裁剪、PIL 缩放与归一化 | 72.47 ms | 24.0% |
| 拼装连续 batch 张量 | 76.24 ms | 25.3% |
| 图像 CPU→GPU | 50.59 ms | 16.8% |
| 位置原型选择 | 0.31 ms | 0.1% |
| ResNet-18 CPG-SSN 前向 | 40.85 ms | 13.5% |
| 分数 GPU→CPU | 4.10 ms | 1.4% |
| Top-2 聚合 | 0.28 ms | 0.1% |

GPU 调度会使“CPU→GPU”和“模型前向”的单次计时互相波动，但两者与流水线总时间在三次测试中稳定。模型前向的两次稳定值约为 21.6 ms；均值被一次 79.2 ms 调度抖动拉高。

启动阶段均值为 2831.87 ms，其中 Python/依赖导入 1588.84 ms、模型初始化 765.29 ms、CUDA 预热 294.55 ms；其余为参数解析、权重/原型读取和原型传输。完整墙钟时间还包含平均约 0.76 s 的解释器启动前后、CUDA 清理和进程退出成本。

## 实现变化

- 同一整图从 32 次 JPEG 打开/解码降为 1 次。
- 删除单图路径的 DataLoader 多进程启动；保留 `--legacy-loader` 便于回归对照。
- 使用 4 个线程并行执行与训练一致的 PIL 裁剪、缩放和 ImageNet 归一化。
- 默认 batch 从 16 调整为 32，一次送入本图的全部图块。
- 原型仅在启动时整体传入 GPU，随后按 key 在 GPU 内组 batch。
- CUDA 首次前向被显式预热并归入启动阶段。
- CSV 自动记录导入、初始化、解码、预处理、传输、前向和聚合等阶段。
- 未采用 GPU 重采样，因为它会改变训练时的 PIL 插值像素；在完整测试集重新校准阈值前，不以潜在精度回归换取速度。

## 复现

```bash
cd /root/program
python scripts/infer_cpg_ssn.py \
  --image /root/test/01_02_ok.jpg \
  --checkpoint /root/program_runs/S08_resnet18/weights.pt \
  --official-repo /root/SuperSimpleNet \
  --prototypes /root/program_runs/S08_resnet18/prototypes.pt \
  --template 01 \
  --output /root/program_runs/S08_resnet18/prediction.csv
```

未显式传入 `--threshold` 时，脚本会读取权重同目录下 `experiment_summary.json` 的验证集阈值。服务器原始 CSV 位于：

```text
/root/program_runs/S08_resnet18/baseline_run1.csv
/root/program_runs/S08_resnet18/baseline_run2.csv
/root/program_runs/S08_resnet18/baseline_run3.csv
/root/program_runs/S08_resnet18/final_optimized_run1.csv
/root/program_runs/S08_resnet18/final_optimized_run2.csv
/root/program_runs/S08_resnet18/final_optimized_run3.csv
```

