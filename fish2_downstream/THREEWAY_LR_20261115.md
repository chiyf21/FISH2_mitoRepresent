# Fish2 fixed-LR：三种表征完整对比

| Representation | Cell Type — Per Mito BA | Cell Type — Per Mito Macro-F1 | Cell Type — Per Neuron BA | Cell Type — Per Neuron Macro-F1 | Compartment — Per Mito BA | Compartment — Per Mito Macro-F1 | Compartment — Per Compartment BA | Compartment — Per Compartment Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No Projection Head (Balanced1152D) | 33.26% | 32.63% | 63.01% | 58.39% | 71.28% | 70.99% | 90.05% | 90.06% |
| With Projection Head — Balanced1152D | **38.76%** | **38.17%** | **72.85%** | **67.98%** | **74.91%** | **74.72%** | **91.32%** | **91.30%** |
| With Projection Head — Fused384D | 31.80% | 30.06% | 69.57% | 67.85% | 67.69% | 67.33% | 85.75% | 85.68% |

粗体为三种表征正式固定 LR 结果中的最高值。Fused384D 的四项 Compartment 指标已按相同协议补齐，其他 20 项原始指标保持不变。

No Projection Head 同时包含主 Covariance 系数校准（0.2 → 0.0004），CLS Covariance 保持 0.05。

来源（113，各指标直接读取对应正式 metrics.json；本次仅补跑 Fused384D Compartment 固定 LR）：

- No Projection Head (Balanced1152D): `/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261115T015720Z_chiyf21_nohead50_maincov0004_L12_balanced1152_c9cd0415/metrics.json`
- With Projection Head — Balanced1152D: `/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261115T015158Z_chiyf21_fusion50_L12_balanced1152_nohead50_comparison_4edf26ae/metrics.json`
- With Projection Head — Fused384D: `/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261114T124741Z_codex_benchmark_celltype_epoch050_fused_lr_e3485d15/metrics.json`
- With Projection Head — Fused384D (Compartment): `/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261115T023813Z_chiyf21_fusion50_fused384_compartment_fixed_lr_1722e56c/metrics.json`

补齐记录：使用统一入口 run.py、canonical_fused384.json，task=compartment、classifier=lr；只在 CPU 上运行，无 Encoder 推理或训练、无调参。

固定 reviewed81_v1：5,059 实例、81 bodies、161 个 body×真实区室组，seed=20261113；按神经元类型、五折 body-disjoint。每折仅用训练集拟合 StandardScaler，LR 为 L2/C=1/lbfgs/class_weight=balanced/max_iter=3000；组内平均 log10(clip(p,1e-15,1)) 后 argmax。

30 个类型×折模型全部收敛，最大迭代次数 162。OOF 覆盖、body/fold 隔离、有限概率、混淆矩阵与 BA/Macro-F1 重算核验通过；更新前后其他 20 项原始指标精确相同。

冻结资产 SHA256：27afd1a373aa9ea4ed6334bbb13828b37193f81bf3735bd7865943bb6fc9a38a。

更新版 CSV/PNG/SVG/PDF、来源及未舍入分数：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/reports/nohead_head_threeway_lr_20261115_02

旧版报告 01 和原始评价结果均保留，配色、字体、尺寸、布局及 No-Head 校准脚注沿用原图。
