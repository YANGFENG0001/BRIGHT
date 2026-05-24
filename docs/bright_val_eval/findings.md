# BRIGHT 泛化验证集评估发现

## 资料阅读

- 官方仓库当前 `bda_benchmark/model/` 提供 `UNet`、`DeepLabV3Plus`、`SiamAttnUNet`、`SiamCRNN`、`DamageFormer` 五个监督基线实现。
- 论文第 3.4 节采用 OA、F1 和 mIoU；F1 分别用于定位子任务和损伤分类子任务，OA/mIoU 衡量最终 4 类损伤图。
- 论文 Table 5 给出 standard ML test set 的 set-level 指标；Table 6 给出事件级 mIoU。
- 本地 `D:\PycharmProjects\SegEarth-OV-3\data\bright_damage\val` 包含 697 个样本，目录为 `images_pre/`、`images_post_sar/`、`targets_cvt_damage/`、`test_ids.txt`。
- 本地 `val/test_ids.txt` 是官方 `bda_benchmark/dataset/splitname/standard_ML/test_set.txt` 的子集：官方 test 为 878 张，本地 val 为 697 张，交集 697 张。
- 本地 val 缺少 `mexico-hurricane`、`myanmar-hurricane`、`ukraine-conflict` 三类事件样本；这与官方 README 提到部分光学数据不可再分发相吻合。
- 本地 `targets_cvt_damage` 标签值为 0/1/2/3；697 张总体像素统计为 `{0: 622348970, 1: 95139515, 2: 5436711, 3: 7932276}`。

## 模型可用性

- 当前官方 BRIGHT 仓库没有 `ChangeOS` 和 `ChangeMamba` 的模型实现或 infer 脚本。
- 官方 Zenodo 权重页 `15349462` 可见 `DamageFormer`、`DeepLabV3Plus`、`SiamAttnUNet`、`SiamCRNN`、`UNet` 的 standard ML 权重，以及若干 cross-event 权重。
- `ChangeMamba` 出现在论文 Table 5/6 和 Appendix D 的实现说明中，但尚未在当前 BRIGHT 仓库代码中发现可直接运行的实现。
- 独立 ChangeMamba 仓库提供 BRIGHT 训练/测试说明和 `ckpt_ChangeMamba_bright_standard_ML_split.pth`，但依赖 `kernels/selective_scan` CUDA 扩展，官方说明为 Linux 环境。

## 指标映射

- 最终 4 类指标：`Final OA = Pixel_Accuracy`，`Final mIoU = Mean_Intersection_over_Union`，每类 IoU 来自 4 类混淆矩阵。
- `F1loc`：二类建筑定位 F1，背景为 0，任意建筑类为 1。
- `F1clf`：官方代码中保留的计算方式倾向于对非背景三类 F1 做调和平均；本次评估脚本按该方式记录 `clf_f1_hmean`。
- 官方 decoupled 模型论文描述为 `Ydam = Yloc * Yclf`，但仓库 infer 脚本当前最终输出直接使用 `argmax(output_clf)`，未乘定位图。本次脚本默认使用仓库 infer 行为，并保留 `--final_mode mask` 可选模式。
- `F1clf` 按官方 infer 脚本中保留的注释逻辑，只在 `labels_loc > 0` 的建筑像素范围内累计混淆矩阵。

## 风险与缺口

- 官方 README 说明代码在 Linux 环境运行，当前执行环境为 Windows PowerShell，可能需要做路径和设备兼容适配。
- 默认 Python 环境没有 `torch/torchvision`；本机 `adaptovcd` conda 环境可用，`torch 2.7.1+cu128`，CUDA 可用，GPU 为 `NVIDIA GeForce RTX 5080`。
- 当前 conda 环境存在另一个名为 `dataset` 的包，新增评估脚本已显式导入本仓库文件，避免包名冲突。

## 官方论文 standard ML 参考指标

| 模型 | F1loc(%) | F1clf(%) | Final OA(%) | Final mIoU(%) | 背景 IoU(%) | Intact IoU(%) | Damaged IoU(%) | Destroyed IoU(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UNet | 87.97 | 72.24 | 95.47 | 64.94 | 96.19 | 71.27 | 39.13 | 53.17 |
| DeepLabV3+ | 87.00 | 70.33 | 95.43 | 64.80 | 95.98 | 70.98 | 37.53 | 54.69 |
| SiamAttnUNet | 88.16 | 70.13 | 95.45 | 64.26 | 96.14 | 71.90 | 35.06 | 53.92 |
| SiamCRNN | 89.45 | 72.02 | 95.76 | 65.73 | 96.48 | 73.44 | 38.91 | 54.18 |
| ChangeOS | 89.60 | 71.88 | 95.84 | 65.98 | 96.54 | 73.85 | 38.99 | 54.53 |
| DamageFormer | 90.29 | 72.51 | 96.13 | 67.09 | 96.87 | 75.04 | 39.86 | 56.59 |
| ChangeMamba | 90.90 | 72.70 | 96.22 | 67.63 | 96.96 | 75.59 | 40.05 | 57.91 |
