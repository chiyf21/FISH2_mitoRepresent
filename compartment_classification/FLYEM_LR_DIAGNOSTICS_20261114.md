# FlyEM LR：Cable聚合错误与22项线性可恢复性

固定身份：epoch033，6,400实例、96 bodies、212个body×真实区室组；原分类器按LC类型训练、五折body-disjoint OOF。直接复用现成概率并复算原成绩一致；未重训Encoder，原实验文件未改。

**聚合落差：Cable本身已更弱，并非只有聚合后才变差。** Handcrafted / Balanced的逐实例BA为0.5999 / 0.6142，但Cable实例Recall为**0.6048 / 0.4889**。因此Balanced总体BA提高并不表示Cable改善。聚合Cable正确组为52/58 / 35/58。

58组配对：**双方正确32、双方错误3、仅Handcrafted正确20、仅Balanced正确3**。逐组body、LC类型、fold、实例数、实例正确率、聚合类别及margin见 `cable_paired_58.csv`；逐实例贡献见 `cable_instance_margins.csv`。

margin定义为组内平均的 `ln p(true) − ln p(c*)`；c*是该组平均log probability最高的竞争类，组内固定；单位nats。

Balanced的23个Cable错组：

- **20/23**的多数实例被误分，组间实例正确率中位数仅**31.6%**。
- **16/23**的多数实例对主要竞争类呈负margin；负margin比例中位数**57.9%**。
- 组平均margin中位数**−1.008**，范围−7.462至−0.013，包含严重偏移与近界错误，不能统称为“轻微负偏差”。
- 最差1个实例占负margin总量的中位数**25.4%**；最差10%占**42.9%**，有明显尾部贡献。
- 按真实类margin删除最差1个，可挽回原错组**9/23**；删除最差ceil(10%×n)个，可挽回**12/23**，仍有11组错误。Handcrafted对应为1/6、2/6。

结论是**普遍Cable判界偏差与尾部放大并存**，不能只归因于几个异常实例。删除操作使用真实标签，只统计原错组挽回情况，**不是新的正式成绩**。

按LC类型，Handcrafted / Balanced错组数：LC4为1/6 / 2/6，LC6为0/11 / 1/11，LC10为1/12 / 1/12，LC12为1/7 / 5/7，LC17为3/13 / 7/13，LC22为0/9 / 7/9。Balanced问题集中在LC12、LC17、LC22。

按组大小，Balanced错组：1–5实例0/1，6–10为6/10，11–20为13/33，>20为4/14；Handcrafted对应0/1、1/10、5/33、0/14。Balanced错组/正确组的大小中位数为16/19。LC12本身多为小组，类型和大小有混杂，不能作独立因果解释。

**数学等价性。** 对同一LR，`log p(c|z)=w_cᵀz+b_c−logsumexp(logits)`，最后一项对类别相同，所以：

`argmax_c mean_i log p(c|z_i) = argmax_c [w_cᵀ mean_i(z_i)+b_c]`。

一个组共用同一LC类型、测试fold及其scaler/LR，故等价于对组内平均标准化embedding做同一LR判类；不等价于多数投票。原实现1e−15截断涉及Handcrafted的14个概率项/12组、Balanced的1项/1组；去掉截断后，各自212组和58个Cable组均无判类变化。因此截断不能解释本次落差。

**固定Ridge probe。** 完全复用原body五折，每折汇合六种LC类型训练一套多输出Ridge，等价于22项独立回归；仅训练折StandardScaler，float64、alpha=1、cholesky，无调参或元数据输入。目标保留作者表原数值，SBI本来就是log10球形度；按实例等权，报告合并OOF R²。此处是信息回归，不是新增pooled分类实验。

| 人工特征 | OOF R² |
|---|---:|
| size | 0.8074 |
| mito SA | 0.8476 |
| mito CA | 0.7099 |
| convex hull compactness | 0.8321 |
| PC1 Length | 0.9272 |
| PC1 inertia moment | 0.2621 |
| PC1 symmetry | 0.8020 |
| PC1 CA | 0.8646 |
| PC1 Circum | 0.7825 |
| PC2 Length | 0.7860 |
| PC2 inertia moment | 0.3946 |
| PC2 symmetry | 0.7543 |
| PC2 CA | 0.6346 |
| PC2 Circum | 0.4537 |
| PC3 Length | 0.8348 |
| PC3 inertia moment | 0.4068 |
| PC3 symmetry | 0.6426 |
| PC3 CA | 0.6628 |
| PC3 Circum | 0.5342 |
| mito diameter | 0.9271 |
| length along skeleton | 0.3284 |
| SBI | 0.9177 |

22项均有正R²，17项≥0.5，中位数0.7684，范围0.2621–0.9272。**许多人工量有较强线性可预测性，但不能声称已完整、精确线性包含全部22项。** 骨架长度仅0.3284，五折为0.2925/0.3748/0.4075/0.3430/0.1903；惯性矩等也较弱。低R²不等于信息不存在，结论限于本固定alpha、有限样本线性读出。

**骨架长度定义与单项移除。** 作者先用神经元骨架局部方向判断线粒体与骨架的重叠和端点投影，再将两端overhang与中间骨架边路径长度相加；两节点间或负长度回退时采用PC1 Length。它依赖神经元骨架的位置、拓扑和方向，不是孤立mask可严格定义的量。本次保留作者CSV单位，未转换。

仅去掉 `length along skeleton`，沿用原分类型LR、fold、训练折scaler、L2、C=1、balanced权重和max_iter=3000：

| 特征 | 聚合BA | Cable Recall | 正确Cable组 |
|---|---:|---:|---:|
| 原Handcrafted22D | 0.9367 | 0.8966 | 52/58 |
| 删除该项的21D | **0.9735** | **0.9483** | **55/58** |
| 原Balanced1152D | 0.8356 | 0.6034 | 35/58 |

30次新增拟合全部收敛，最多100次迭代。移除后BA上升0.0369，**不支持“骨架长度这一项造成Handcrafted优势”**。这是固定划分观察结果，未做统计显著性检验。

但21D仍不能称为纯mask：作者 `mito CA` 的截面方向由神经元局部分割优化得到，再测同平面线粒体截面。因此其他上下文是否贡献优势，未被此次单项移除排除。

**已支持与候选解释。** 数据支持Cable本来偏弱、组内偏差累积且尾部有贡献、部分几何量可高精度线性读出；单独骨架长度解释不获支持。同C下22D/1152D的有效正则差异、判界方向与表示的对齐、置信度校准，以及其他上下文的作用仍是候选，未增加实验检验。Ridge能恢复几何量，不保证分类LR会学到与人工特征模型相同的判界。

定义依据为本地作者仓库提交 `a1874cfa9d8af2cda70fca3f8bfa1efe5c9d684e` 的 [骨架长度计算](https://github.com/ClarkLabCode/MitochondrialMorphologyPosition/blob/a1874cfa9d8af2cda70fca3f8bfa1efe5c9d684e/HPC/save_mito_synapse/mito_synapse_main.py#L173-L190)、[方向与投影](https://github.com/ClarkLabCode/MitochondrialMorphologyPosition/blob/a1874cfa9d8af2cda70fca3f8bfa1efe5c9d684e/util_files/utils.py#L594-L632) 及 [神经元截面方向](https://github.com/ClarkLabCode/MitochondrialMorphologyPosition/blob/a1874cfa9d8af2cda70fca3f8bfa1efe5c9d684e/util_files/voxel_utils.py#L586-L645)。

原OOF目录：`/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/compartment_classification/balanced_lr_20261113_v1/flyem/`。
新输出目录：`/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/compartment_classification/flyem_lr_diagnostics_20261114_v1/`。
完整输入路径/SHA256、逐折R²、逐组与逐实例诊断、21D新OOF均已保存。

代码：`compartment_classification/flyem_lr_diagnostics.py`，运行提交 `af261b5`；独立分支 `codex/flyem-lr-diagnostics-20261114`。
