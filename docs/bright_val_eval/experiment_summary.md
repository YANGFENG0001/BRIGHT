# BRIGHT 本地 val 泛化评估总结

## 评估设置

- 评估日期：2026-05-24
- 代码分支：`test`
- 数据目录：`D:\PycharmProjects\SegEarth-OV-3\data\bright_damage\val`
- 样本数：697
- 运行环境：`D:\anaconda3\envs\adaptovcd\python.exe`，`torch 2.7.1+cu128`，GPU `NVIDIA GeForce RTX 5080`
- 评估脚本：`bda_benchmark/script/standard_ML/eval_local_bright_val.py`
- 输出目录：`docs/bright_val_eval/results/`
- 日志目录：`docs/bright_val_eval/logs/`

## 数据集核对

本地 `val/test_ids.txt` 与官方 `bda_benchmark/dataset/splitname/standard_ML/test_set.txt` 的交集为 697 张，本地没有额外样本；官方 standard test 共 878 张。本地 val 缺少三个事件：

- `mexico-hurricane`：47 张
- `myanmar-hurricane`：28 张
- `ukraine-conflict`：106 张

因此，本次评估不是一个新增事件集合，而是官方 standard ML test 的可用子集。事件级 mIoU 对于保留的 11 个事件与论文 Table 6 基本一致；set-level 指标变化主要来自样本子集变化。

## 权重来源

已下载并评估的官方 BRIGHT Zenodo `15349462` standard ML 权重：

- `ckpt_UNet_standard_ML_split.pth`
- `ckpt_DeepLabV3Plus_standard_ML_split.pth`
- `ckpt_SiamAttnUNet_standard_ML_split.pth`
- `ckpt_SiamCRNN_standard_ML_split.pth`
- `ckpt_DamageFormer_standard_ML_split.pth`

另已下载 ChangeMamba 官方 Zenodo `15479555` 权重：

- `ckpt_ChangeMamba_bright_standard_ML_split.pth`

当前 BRIGHT 仓库没有 ChangeMamba/ChangeOS 模型实现或 infer 脚本。ChangeMamba 官方代码在独立仓库中，依赖 `kernels/selective_scan` CUDA 扩展，官方 README 说明只在 Linux 测试。因此本轮没有把 ChangeMamba 纳入同一 Windows/BRIGHT 仓库评估流程。

## Set-Level 对比

| 模型 | 本地 F1loc | 本地 F1clf | 本地 OA | 本地 mIoU | 论文 OA | 论文 mIoU | OA 变化 | mIoU 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UNet | 89.07 | 62.63 | 95.95 | 63.37 | 95.47 | 64.94 | +0.48 | -1.57 |
| DeepLabV3Plus | 88.81 | 63.90 | 95.97 | 63.93 | 95.43 | 64.80 | +0.54 | -0.87 |
| SiamAttnUNet | 89.20 | 67.53 | 95.98 | 65.01 | 95.45 | 64.26 | +0.53 | +0.75 |
| SiamCRNN | 90.45 | 62.00 | 96.20 | 64.10 | 95.76 | 65.73 | +0.44 | -1.63 |
| DamageFormer | 91.09 | 62.00 | 96.54 | 65.06 | 96.13 | 67.09 | +0.41 | -2.03 |

## 观察

- 本地 val 上五个可评估模型的 OA 均高于论文完整 test set 约 0.4 到 0.5 个百分点。
- mIoU 变化更敏感：DamageFormer、SiamCRNN、UNet、DeepLabV3Plus 均下降，SiamAttnUNet 上升。
- 本地 val 的 Damaged 类 IoU 明显低于论文完整 test set；Destroyed 类 IoU 普遍高于或接近论文值。这说明去掉三个事件后，类别分布变化对 set-level mIoU 有直接影响。
- 在本地 val set-level mIoU 排名中，DamageFormer 最高，其次是 SiamAttnUNet、SiamCRNN、DeepLabV3Plus、UNet。
- 11 个保留事件的事件级 mIoU 平均值为：DamageFormer 53.18，SiamCRNN 51.35，SiamAttnUNet 51.26，UNet 50.09，DeepLabV3Plus 48.94。

## 复现实验命令模板

```powershell
D:\anaconda3\envs\adaptovcd\python.exe bda_benchmark\script\standard_ML\eval_local_bright_val.py `
  --model DamageFormer `
  --model_path checkpoints\official_zenodo\standard_ml\ckpt_DamageFormer_standard_ML_split.pth `
  --split_dir D:\PycharmProjects\SegEarth-OV-3\data\bright_damage\val `
  --output_dir docs\bright_val_eval\results `
  --device cuda `
  --no_progress
```

