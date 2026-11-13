# FlyEM三分类正式结果（2026-11-13）

固定epoch033表征；6400作者exact-center实例/96 bodies，三类为axon、dendrite、connecting cable。完全沿用二分类的body分区及所有外层/内层body折，不重新挑选holdout。

development：5718实例/85 bodies；独立holdout：682实例/11 bodies。holdout中cable有94实例、仅6个body。

| 特征 | 分类器 | Holdout BA / Macro-F1 | OOF BA / Macro-F1 | Holdout Cable P / R / F1 |
|---|---|---|---|---|
| handcrafted | logistic | 0.5940 / 0.5589 | 0.5714 / 0.5432 | 0.3409 / 0.6383 / 0.4444 |
| handcrafted | random_forest | 0.6151 / 0.6184 | 0.5683 / 0.5693 | 0.4943 / 0.4574 / 0.4751 |
| embedding | logistic | 0.6354 / 0.6174 | 0.5888 / 0.5690 | 0.3786 / 0.5638 / 0.4530 |
| embedding | random_forest | 0.5251 / 0.5300 | 0.5079 / 0.5083 | 0.3509 / 0.2128 / 0.2649 |

独立holdout，Embedding−Handcrafted，5000次body配对bootstrap、95%区间：

| 分类器 | ΔBA | ΔMacro-F1 | ΔCable Recall | ΔCable F1 |
|---|---|---|---|---|
| logistic | +0.0414 [-0.0271, +0.1062] | +0.0585 [+0.0069, +0.1096] | -0.0745 [-0.2381, +0.0921] | +0.0085 [-0.0984, +0.1242] |
| random_forest | -0.0900 [-0.1418, -0.0281] | -0.0884 [-0.1406, -0.0198] | -0.2447 [-0.4040, -0.0732] | -0.2102 [-0.3465, -0.0544] |

结论：本协议下不支持384维embedding更容易区分三个区室，尤其没有cable优势。LR的整体Macro-F1有所改善，但cable Recall下降、cable F1差异不明确；RF的总体BA、Macro-F1以及cable Recall/F1均支持手工特征更好。结论限于本次冻结表征、LR/RF和当前测试body，不证明向量本身不包含cable信息。

Dummy holdout BA=0.3333，Macro-F1=0.2218。30个保存模型的复核通过，所有分区、内外层折均覆盖三类，且body隔离；保存预测、概率、标准化统计、指标与参数选择均复核通过。未训练/导出编码器，未修改原二分类产物。

限制：cable外测独立body仅6个；bootstrap仅描述固定预测下body抽样的不确定性；手工特征含骨架长度和PCA几何量；历史图像预训练暴露限制沿用原实验。没有按神经元类型拆分或聚合分析。完整预测、划分、覆盖清单和运行记录在私有结果包中，原始数据和权重不入Git。
