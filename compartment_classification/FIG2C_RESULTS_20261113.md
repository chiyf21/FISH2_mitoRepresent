# FlyEM Figure 2C 风格三分类聚合实验

| 训练方式 | Handcrafted 22D：BA / Macro-F1 | Embedding 384D：BA / Macro-F1 |
|---|---|---|
| A：按LC类型分别训练 | 0.8746 / 0.8854 | 0.7322 / 0.7257 |
| B：混合类型统一训练 | 0.8573 / 0.8667 | 0.7826 / 0.7903 |

**结论：**Handcrafted在A、B两种方式下均更好；Handcrafted以按LC训练更好（F1 0.8854 vs 0.8667），Embedding以混合训练更好（0.7903 vs 0.7257）。最高F1 0.8854，距用户给出的论文0.90约1.46个百分点；未追求该数值更换划分或调参。

| 条件 | axon Recall | dendrite Recall | connecting cable Recall |
|---|---|---|---|
| A_by_type_handcrafted | 0.9688 | 0.9655 | 0.6897 |
| A_by_type_embedding | 0.9896 | 0.8793 | 0.3276 |
| B_pooled_handcrafted | 0.9167 | 0.8448 | 0.8103 |
| B_pooled_embedding | 0.9167 | 0.8103 | 0.6207 |

**样本与聚合：**作者exact-center子集6,400个线粒体、96个body、6种LC，共212个实际存在的body×区室组（axon 96、dendrite 58、connecting cable 58）。每组2–116个线粒体，中位26.5，四分位18–37，均值30.19。缺失于该子集的区室不虚构组。

**协议：**原五折只覆盖85个development bodies，另11个为holdout；本次为覆盖全部96个body，按LC类型分层生成四组共享的固定五折（seed=20261113）。A的每个模型只使用对应LC的训练body；B使用其余折全部LC的训练body。六类LC各9–18个body，70个RF模型均没有缺失训练类别，全部实例恰有一次OOF预测。

测试类别覆盖：LC4 fold0、LC6 fold1、LC22 fold3（fold从0编号）的测试body在exact子集中只有axon，没有dendrite/connecting cable组；未跳过这些body或fold，也未补造区室组。70个模型的训练集均包含三类，最终四组仍共享全部212个实际存在的区室组。

RF固定100树、min_weight_fraction_leaf=0.01、训练区室样本数倒数权重；无内层调参或新holdout。聚合为组内平均log10概率后argmax，使用1e-15防止log(0)；四组与未截断原式的预测差异均为0。RF直接输入特征，不使用原notebook的全样本zscore。

**解释边界：**组由body_id＋真实compartment定义，属于oracle grouping，不代表自动发现未知区室。参考notebook保存的运行覆盖21种LC，本次仅6类的exact-center子集；样本覆盖、固定折与随机种子不同，不能当作严格同数据复现。手工22维包含神经元骨架长度和PCA几何信息。本OOF隔离的是RF训练body，冻结embedding的历史预训练暴露未重新控制；结论限于本次固定划分和聚合协议。

四组逐组预测见paired_group_predictions.csv；组大小见groups.csv；固定划分见body_folds.csv、instances.csv；每类Recall和逐折结果见summary.csv、fold_metrics.csv；输入及checkpoint哈希见provenance.json；验证见validation.json。未运行编码器或其他分类器，既有二分类/三分类产物保持不变。

作者代码：https://github.com/ClarkLabCode/MitochondrialMorphologyPosition/blob/main/morphology%20classifier%20-%20LC%20neurons.ipynb

![四组聚合混淆矩阵](four_conditions_confusion.png)
