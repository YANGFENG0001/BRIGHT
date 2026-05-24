# BRIGHT 泛化验证集评估进度

## 2026-05-24

- 从跟踪 `origin/master` 的干净 `master` 工作区开始。
- 确认 `origin` 指向 `https://github.com/YANGFENG0001/BRIGHT.git`。
- 切换到跟踪 `origin/test` 的本地 `test` 分支。
- 创建专用文档目录 `docs/bright_val_eval/`。
- 根据用户偏好，将实验记录语言调整为中文。
- 阅读官方 README、standard ML infer 脚本、`metrics.py`、本地论文第 2.4/3.4/4.1 节和 Table 5/6。
- 确认本地 val 共 697 张，是官方 standard ML test 的 697 张子集。
- 新增 `bda_benchmark/script/standard_ML/eval_local_bright_val.py`，用于直接评估本地 `images_pre/images_post_sar/targets_cvt_damage` 目录。
- 为 `SiamCRNN`、`DamageFormer`、`DeepLabV3Plus` 增加 `pretrained=False` 构造入口，避免加载完整 checkpoint 前额外下载 ImageNet backbone。
- 使用随机 UNet checkpoint 对 1 张样本完成烟测；烟测只验证流程可执行，不作为实验结果。
- 第一轮改动已提交：`6811905 Add local BRIGHT val evaluation workflow`。
- 从官方 Zenodo `15349462` 下载五个 standard ML 权重，并校验文件大小与 API 清单一致。
- 对 `UNet`、`DeepLabV3Plus`、`SiamAttnUNet`、`SiamCRNN`、`DamageFormer` 分别完成 697 张本地 val 完整评估。
- 发现并修正 `F1clf` 计算范围：按官方脚本注释，只在建筑像素区域计算损伤分类 F1。
- 下载 ChangeMamba 官方 Zenodo `15479555` 的 BRIGHT standard ML 权重；由于当前 BRIGHT 仓库没有模型实现，且独立 ChangeMamba 代码依赖 Linux 下的 `selective_scan` CUDA 扩展，本轮未执行 ChangeMamba 评估。
- 生成 `comparison_metrics.csv`、`event_miou_local_val.csv` 和 `experiment_summary.md`。
