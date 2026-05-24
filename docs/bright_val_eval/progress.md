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
