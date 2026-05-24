# BRIGHT 泛化验证集评估计划

## 目标

在本地更泛化的 BRIGHT 验证集上评估官方基线模型，并与官方论文指标进行对比。

- 官方仓库：`https://github.com/ChenHongruixuan/BRIGHT`
- 本地论文：`E:\本科毕设\可能参考文献\Chen 等 - 2025 - BRIGHT A globally distributed multimodal building damage assessment dataset with very-high-resoluti.pdf`
- 本地数据集根目录：`D:\PycharmProjects\SegEarth-OV-3\data\bright_damage`
- 本次评估目录：`D:\PycharmProjects\SegEarth-OV-3\data\bright_damage\val`
- 目标分支：`test`
- 远程仓库：`https://github.com/YANGFENG0001/BRIGHT`

## 工作范围

1. 阅读官方代码、论文指标定义和本地数据目录结构。
2. 从官方项目公开链接下载可用模型权重。
3. 在保持官方指标定义一致的前提下，适配本地 `val` 目录评估。
4. 运行可复现的官方基线评估，保存原始日志。
5. 在本目录保存实验记录、指标表和问题说明。
6. 将每轮有意义改动提交到 Git，并推送到 `origin/test`。

## 约束与假设

- 不凭空补写官方没有提供的模型实现或权重。
- 如果 `ChangeMamba` 等用户点名模型未出现在官方仓库或官方权重页，将在实验记录中明确说明。
- 大体积权重和预测图不纳入 Git 跟踪，只提交代码、配置、日志摘要和文档。

