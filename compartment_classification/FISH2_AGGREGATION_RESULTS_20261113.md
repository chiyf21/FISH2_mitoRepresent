# Fish2 二分类 oracle 聚合

| 训练方式 | Handcrafted 22D：BA / Macro-F1 | Embedding 384D：BA / Macro-F1 |
|---|---|---|
| A：按神经元类型分别训练 | 0.8697 / 0.8695 | 0.8387 / 0.8384 |
| B：混合类型训练 | 0.8328 / 0.8314 | 0.8018 / 0.7999 |

本次固定五折OOF：Handcrafted在按类型、混合两种训练方式下均优于Embedding；按类型分别训练对两种特征都优于混合训练。最优A1的BA=0.8697、Macro-F1=0.8695。该判断是当前固定划分下的点估计，不作普遍或因果结论。

| 条件 | axon Recall | dendrite Recall |
|---|---|---|
| A_by_type_handcrafted | 0.8875 | 0.8519 |
| A_by_type_embedding | 0.8750 | 0.8025 |
| B_pooled_handcrafted | 0.9125 | 0.7531 |
| B_pooled_embedding | 0.8875 | 0.7160 |

类型body数量：{"CON": 12, "GC": 22, "MON1": 12, "MON2": 12, "PC": 12, "VMN": 11}。全部支持五折；训练缺类模型数：0，测试缺类模型数：0。未跳过任何实例或实际存在的区室组。

组大小分布：{"instances": 5059, "bodies": 81, "groups": 161, "groups_by_class": {"dendrite": 81, "axon": 80}, "min": 1, "q25": 9.0, "median": 27.0, "q75": 45.0, "max": 134, "mean": 31.422360248447205}

四组共享同一套全81个body的固定五折（按神经元类型分层，seed=20261113）。原二分类外层五折仅覆盖58个development bodies、另23个body外测，因此本次重新生成覆盖全体的OOF划分。A按已知类型路由，训练集中不混入其他类型；B混合训练，cell_type不作为数值输入。
复用FlyEM的RF及聚合实现：100树、min_weight_fraction_leaf=0.01、训练类别计数倒数权重；无调参、不标准化、不补缺失值（所有数值有限）。聚合为body_id+真实compartment的oracle grouping，平均log10概率后argmax；log(0)下界固定1e-15。

数值保护改变原未截断式预测的组数：{"A_by_type_handcrafted": 0, "A_by_type_embedding": 0, "B_pooled_handcrafted": 0, "B_pooled_embedding": 0}

差异（后者−前者）：
- A_by_type_embedding − A_by_type_handcrafted：ΔBA=-0.0309，ΔMacro-F1=-0.0312；后者/前者独有正确组=6/11。
- B_pooled_embedding − B_pooled_handcrafted：ΔBA=-0.0310，ΔMacro-F1=-0.0314；后者/前者独有正确组=10/15。
- B_pooled_handcrafted − A_by_type_handcrafted：ΔBA=-0.0369，ΔMacro-F1=-0.0382；后者/前者独有正确组=12/18。
- B_pooled_embedding − A_by_type_embedding：ΔBA=-0.0370，ΔMacro-F1=-0.0384；后者/前者独有正确组=10/16。

边界：评价单位是已知真实区室的聚合组，不代表自动发现区室，不能直接与单线粒体或FlyEM三分类成绩比较。Fish2 reviewed标签仍缺完整人工审核来源追溯；手工特征包含神经元骨架长度和PCA几何；既有编码器的预训练暴露未重新控制。本次结论限于固定五折下游OOF。未运行Logistic Regression、三分类或编码器。

文件：summary.csv含每类Recall；paired_group_predictions.csv含四组配对预测；groups.csv含每组大小；body_folds.csv、instances.csv记录固定划分；training_records.json记录每个RF的训练范围与类别覆盖；provenance.json记录epoch050输入身份与哈希。

图及完整运行产物保存在独立实验输出目录。
