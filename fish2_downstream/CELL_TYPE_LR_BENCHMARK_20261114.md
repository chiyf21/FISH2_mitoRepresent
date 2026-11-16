# Fish2 Cell Type：固定LR基准

reviewed81_v1：5568实例、81 bodies、6类。三种表示严格复用冻结五折body-disjoint划分（seed=20261113），不调参、不选最佳checkpoint。
每折全部训练类型混合拟合一个多分类LR；训练折StandardScaler，L2、C=1、lbfgs、class_weight=balanced、max_iter=3000、random_state=SEED+fold。实例OOF后按body平均log10(clip(p,1e-15,1))取argmax。
RF采用原100树/叶权重0.01/逆训练类别计数sample_weight协议，不标准化。两分类器共享实例、标签、fold及聚合定义。

| 表示 | 评价 | RF BA | RF Macro-F1 | LR BA | LR Macro-F1 | Δ BA | Δ Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Handcrafted22D | 实例 | 0.2519 | 0.2337 | 0.2554 | 0.2221 | +0.0035 | -0.0116 |
| Handcrafted22D | body聚合 | 0.5316 | 0.5159 | 0.5253 | 0.5270 | -0.0063 | +0.0111 |
| Balanced1152D | 实例 | 0.3235 | 0.3027 | 0.3876 | 0.3817 | +0.0641 | +0.0790 |
| Balanced1152D | body聚合 | 0.6591 | 0.6190 | 0.7285 | 0.6798 | +0.0694 | +0.0607 |
| Fused384D | 实例 | 0.3119 | 0.2970 | 0.3180 | 0.3006 | +0.0061 | +0.0036 |
| Fused384D | body聚合 | 0.6111 | 0.5814 | 0.6957 | 0.6785 | +0.0846 | +0.0971 |

Δ=LR−RF，0–1分数绝对差值。没有进行统计显著性检验。
15次LR拟合全部收敛，迭代次数129–1109；7项单元测试通过。Compartment LR与三组Cell Type RF历史回归通过：BA/Macro-F1容差1e-12，混淆矩阵计数完全一致。

Balanced与Fused来自同一Fish2 Fusion50 refine_epoch050.pt；仅复用已注册向量和实例映射，没有Encoder推理。Handcrafted使用固定22列。

**Handcrafted22D**

- LR运行：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261114T124740Z_codex_benchmark_celltype_epoch050_balanced_lr_c3a955b5
- RF运行：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261114T122756Z_codex_validation_historical_rf_e8448c2f
- LR代码提交：383e5d5e3e7f1f84d67a00aca169f70ff3dfee9a
- 向量descriptor：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/assets/reviewed81_v1/canonical_balanced1152.json

**Balanced1152D**

- LR运行：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261114T124740Z_codex_benchmark_celltype_epoch050_balanced_lr_c3a955b5
- RF运行：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261114T122756Z_codex_validation_historical_rf_e8448c2f
- LR代码提交：383e5d5e3e7f1f84d67a00aca169f70ff3dfee9a
- 向量descriptor：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/assets/reviewed81_v1/canonical_balanced1152.json

**Fused384D**

- LR运行：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261114T124741Z_codex_benchmark_celltype_epoch050_fused_lr_e3485d15
- RF运行：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261114T124752Z_codex_validation_celltype_epoch050_fused_rf_reference_df4c087b
- LR代码提交：383e5d5e3e7f1f84d67a00aca169f70ff3dfee9a
- 向量descriptor：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/assets/reviewed81_v1/canonical_fused384.json

每个运行均保存owner、checkpoint/embedding身份与hash、冻结实例/fold、实例及body OOF、混淆矩阵、指标、训练记录和完整日志。

RF对照来源：Handcrafted/Balanced复用既有已核验OOF；Fused补算同协议RF OOF以补齐实例指标，body混淆矩阵/BA/Macro-F1复现历史结果。未覆盖旧RF运行或冻结资产。

本固定基准中，Balanced的body BA/Macro-F1相对RF提高0.0694/0.0607，Fused提高0.0846/0.0971。建议将这套固定LR作为当前learned embedding的线性评价方法，并保留RF非线性对照；不据此断言所有表示、所有新checkpoint或其他任务上LR都更好。没有进行超参数搜索或选择最佳模型。

输入checkpoint为同一文件：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/point-MAE/runs/mito_newmae_vicreg_reg02_cls005_20261105_01/checkpoints/refine_epoch050.pt。
