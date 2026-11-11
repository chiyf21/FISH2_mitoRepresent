# 当前 Point-MAE 线粒体表征流程与诊断记录

**整理日期：2026-10-10**  
**范围：** 113 服务器上的 `/path/to/JaneliaEM/point-MAE` 代码，以及硬盘目录 `/path/to/project_data/EM/point-MAE` 中的 E16 完整数据训练、checkpoint、embedding 与诊断结果。本文记录的是已运行和已有文件支持的事实；推断和未验证事项会单独标明。本文不修改模型或训练结果。

## 1. 先看结论

目前训练的是**线粒体单实例点云的自监督表示模型**：每个线粒体被转换成 2048 个带属性的三维点；Point-MAE 对随机遮挡的局部点组进行重建，并用同一线粒体的第二种采样/旋转视图做对比约束；最后把全局表示映射为 128 维向量。

当前用于观察的 checkpoint 来自 **2166 个预测线粒体 + 391 个手工标注线粒体，共 2557 个实例**，训练了 50 个 epoch。最佳 checkpoint 是第 48 个 epoch（日志 epoch 编号 47），验证总 loss 为 **1.7322**，独立 test split 总 loss 为 **1.7346**。

已经确认的诊断结论：

1. 最终 128D embedding **不是所有实例都坍缩成同一个向量**：实例间 median cosine similarity 约 0.0051，未发现近乎重复的向量。但 embedding 有明显低秩结构（effective rank 27.93/128）。
2. 基础形态/纹理信息在 tokenizer、encoder 的 pooled 表示中可读出；经过最终 projection 后，若干形态指标的线性可预测性变弱，尤其 elongation。这是“信息在最终表征中变弱”的证据，不等同于 projection head 未训练；checkpoint 中 projection 参数确实收到梯度。
3. reconstruction decoder 确实会用 encoder 内容，但对 token 与位置的正确对应关系敏感度很弱：shuffle encoder tokens 的 loss 没有可测上升，置零后只上升约 9.8%。
4. UMAP 上出现的区域不能直接当成稳定的生物学类别。Leiden 在某一参数下可画出 4 个区域，但分群对邻居数/随机种子敏感；不同聚类方法间一致性也不高。当前证据更支持“有连续形态变化，也可能混有来源/采集差异”，还不能断言存在四种离散类型。

## 2. 数据从哪里来、如何筛选

### 2.1 合并训练集

训练 cache：

`/path/to/project_data/EM/point-MAE/cache/manual391_plus_predE16_complete_min5000.zarr`

总计 2557 个线粒体实例：

| 来源 | 实例数 | 说明 |
|---|---:|---|
| E16 预测线粒体 | 2166 | 从 neuron_mito 预测结果中导入，过滤阈值 min voxel = 5000 |
| 三组手工标注 | 391 | dataset_1280：40；dataset_2500：229；dataset_2500_v2：122 |
| 合计 | 2557 | 目前均作为 mitochondria 训练 |

预测数据导入汇总显示：2483 个候选中，2166 个成功导入，17 个因不完整而排除，300 个因为缺少对应 raw EM block 而无法导入，错误数为 0；覆盖 24 个 neuron/body ID。这里的“完整”按跨 block 边界规则判断：若实例碰到当前 block 的边界、且相邻 block 不在可用范围内，则不纳入完整实例集合。小于 5000 voxel 的实例由导入阈值排除。

预测数据附带细胞类型元数据，包含 GC、CON、VMN、PC、MON1、MON2 六类；手工标注数据只有三个数据集来源标识，没有与预测数据同口径的细胞类型标签。**细胞类型没有作为点云输入特征，也没有作为训练监督目标**，只用于训练后画图和分组检查。

### 2.2 划分方式与代表性限制

当前 split 文件记录 train/validation/test 分别为 **1968 / 330 / 259**。预测数据按 neuron/body ID 分组，避免同一预测 neuron 的实例同时出现在不同 split。手工标注数据没有逐 neuron ID，因此按三个数据集来源整体分组，而不是逐 neuron 分组。test 的三个 held-out group 是预测数据 body ID `102250085`、`106638875`、`101468420`；validation 也包含预测 body ID 和一个手工数据来源组。

这比随机按实例拆分更能避免明显的同 neuron 泄漏，但手工数据只能做到**数据集级别隔离**。test 只有三个预测 neuron/group，因此 test 指标可用于当前的 sanity check，不足以代表所有 neuron 或细胞类型的泛化能力；手工来源差异也可能被误当成形态差异。

## 3. 一个线粒体怎样变成模型输入

### 3.1 点数与空间坐标

每个实例默认采样 **2048 个点**，其中目标为 1024 个 surface 点、1024 个 interior 点。Surface 由 3×3×3 结构元素的 binary erosion 得到：原 mask 减去腐蚀后的 mask 是 surface，腐蚀后的部分是 interior。若没有 interior 候选，则回退到整个实例作为 interior 候选。每类候选点通过最远点采样（FPS）选择；候选点数不超过目标数时会有放回采样，因此可能出现重复坐标。

原始数组按 ZYX 轴处理，物理 voxel spacing 是 **(30, 8, 8) nm**。点坐标先换算成物理距离，再减去当前线粒体的质心，并统一除以 **1000 nm**。这会去掉绝对空间位置，但**不会把每个线粒体单独缩放到单位大小**，所以总体尺度仍保留在输入坐标里。

### 3.2 每个点的 7 个通道

每个点由 3 个空间坐标和 4 个属性组成，共 7 个数。前三个空间通道是以 ZYX 数组轴顺序换算的中心化物理坐标；后四个属性依次为 intensity、gradient、distance、surface。

#### `local crop` 的精确定义

这里的 local crop **不是固定大小的 patch，也不是围绕每个点取的小邻域**。它是当前单个线粒体分割 mask 的**紧致轴对齐 3D 外接框**：沿 Z、Y、X 分别取该实例坐标的最小值到最大值（Python slice 的右端不包含），不额外加 padding。不同线粒体的 crop 尺寸因此不同。

```text
一个 Z 截面示意（真实处理是 3D bbox）：
+---------------- bbox ----------------+
| raw 背景 | mitochondrion mask | 背景 |
| 背景     | mask | mask | 背景        |
| raw 背景 / mask 空隙 / 其他标签区域  |
+--------------------------------------+
           ↑ intensity 的 median/MAD 用 bbox 内整块 raw 灰度统计
```

- 对全 volume 的手工/实例标签，`preprocess.py` 用 `scipy.ndimage.find_objects(labels)` 得到每个 instance 的 bbox，再将同一个 bbox 同时应用到 label volume 和 raw EM volume。
- 对预测 mask，`import_predicted_instances.py` 从实例汇总表读取 `bbox_xyz`，按这个 bbox 从 raw EM blocks 拼出同尺寸 crop；mask 文件本身也是 bbox 范围内的局部 mask。
- `pointmae.py::sample_instance()` 接收 bbox 后切出 `crop_labels`、`crop_raw`，并在该 crop 内得到实例 mask。crop 覆盖的是一个**长方体**，不是仅由 mitochondrion 组成的 mask。

#### 四个属性具体怎样算

1. **XYZ/空间坐标**：采样 voxel 坐标乘以 `(30,8,8) nm`，减去该实例所有 mask voxel 的物理质心，再除以全局固定 `1000 nm`。去掉绝对位置，不做每个实例的尺寸归一化。
2. **intensity**：先从 `crop_raw` 的**所有体素**（把整个长方体展平）计算中位数 `m` 和 MAD=`median(|I-m|)`；然后只在被采样的线粒体点位置读取原始灰度 `I_p`，计算 `clip((I_p-m)/(1.4826×MAD+1e-6), -5, 5)`。因此归一化统计量**不只来自 mask 内线粒体 voxel**：bbox 内属于背景、空隙或其他标签的 raw 灰度也参与 median/MAD；bbox 之外的 voxel 不参与。它是每实例 bbox 级稳健标准化，不是每个点的局部 neighborhood normalization，也不是原始灰度。
3. **gradient**：在完整 `crop_raw` 上按 Z/Y/X 三轴分别用 `(30,8,8) nm` 的物理步长计算 `numpy.gradient`，合成梯度幅值，再在采样点位置取值。没有额外做 robust scaling。由于 crop 没有 padding，位于 crop 面上的梯度使用 `numpy.gradient` 的边界差分；这与先在完整 raw volume 求梯度再裁切可能不同。
4. **distance**：对当前实例的二值 mask 做 Euclidean distance transform，按 `(30,8,8) nm` 计物理距离，在采样 voxel 位置取值后除以 `1000 nm`。
5. **surface**：来自 3×3×3 binary erosion 后形成的 surface/interior 两个候选池。surface 池采出的点标 1，interior 池采出的点标 0；所以这个通道精确说是“采样时来自哪个池”的标记。若 erosion 后没有 interior，代码会把整个 mask 回退为 interior 候选，但仍标记为 0；因此这些退化实例中，surface 通道不一定等于该点按几何重新判断的 surface 真值。

Surface/interior 的几何定义使用 voxel 网格上的 3×3×3 erosion。由于 Z spacing (30 nm) 与 XY spacing (8 nm) 不同，这个结构元素不是物理空间中的各向同性小球；它在 z 方向对应的物理厚度更大。这是当前预处理定义，是否影响薄/细长结构尚未单独做对照。

当前 surface/interior 采样比例被固定为一半一半，因此**每个实例整体的 surface 比例都相同**。单点的 surface 标志仍提供局部信息，但它的实例级平均值没有区分度。对 intensity 做局部归一化有助于减少不同 volume 灰度范围影响，但会抹去一部分绝对灰度尺度差异；这是当前选择的取舍。

### 3.3 输入构造的已知注意点

- 诊断 cache 的 2557 个点云均为有限数值，实例 ID 没发现重复映射。
- 52/2557 个点云出现重复坐标；24/2557 个实例存在 surface/interior 标志与基于点坐标回查的类别不一致。上面说明的“无 interior 时回退但仍标 0”是可能机制；目前尚未逐一确认这 24 个实例是否都由该分支导致。
- 由于每个实例都强制采样 1024+1024，surface 比例本身不携带 instance-level 差异。
- 当前 cache 保存的是采样点云，不包含从原始 mask 精确计算的 voxel volume 和 mesh surface area。现有诊断中的 bbox/凸包体积与面积是点云 proxy，不能当作精确生物形态测量。

## 4. 模型结构与一次前向过程

可以把当前模型理解为两条配合的路径：训练时一条负责“补回被遮挡的点组”，另一条约束同一线粒体不同视图的全局表示接近。

```text
实例 mask + raw EM crop
        ↓
2048 个 7D 点（XYZ + 4 个属性）
        ↓
64 个局部 group，每组 32 个邻近点
        ↓
point tokenizer：组内点 MLP + max pooling + center positional MLP
        ↓
随机遮挡约 60% group；可见 tokens 进入 Transformer encoder
        ├── 全局 mean pool → 384D → projection MLP → 128D embedding
        └── decoder + mask tokens → 重建被遮挡 group 的 32×7 点属性

第二视图：同一实例重新采样/旋转 → 同一 encoder/projection → 对比损失
```

### 4.1 局部 tokenizer 和 encoder

FPS 选出 64 个 group center，每个 center 取 32 个最近点。组内 XYZ 改为相对中心的坐标，与 4 个属性拼成 7D 点特征；经 MLP 映射并对组内 32 点 max-pool，再加 center 位置编码。large preset 使用 384 维、10 层、8 头 Transformer encoder，MLP ratio=4。

用于最终 embedding 的路径会对 encoder 输出的 64 个 tokens 做 mean pooling，再经 `384→384→128` projection MLP。训练和提取 embedding 使用同一 projection head；提取时 `eval()`、关闭随机 mask。

### 4.2 MAE decoder 和重建目标

64 个 patches 中随机遮挡 60%，每次大约 38 个 masked、26 个 visible。可见 tokens 经 encoder 后投影到 decoder 的 192 维；mask token 与位置编码一起送入 4 层、8 头 Transformer decoder。decoder 预测每个被遮挡 patch 的 32×7 点属性，只在被遮挡 patches 上计算重建损失。

重建损失包含五项，当前配置权重均为 1：

1. `geo`：预测点与目标点的 Chamfer 距离。
2. `intensity`：按几何匹配后的 Smooth-L1 属性误差。
3. `gradient`：同上，预测梯度属性误差。
4. `distance`：同上，预测 distance-to-boundary 属性误差。
5. `surface`：surface/interior 的 BCE-with-logits。

Chamfer 匹配的点属性损失会根据几何最近邻对应来比较，避免任意点序号对应。各 loss 的数值尺度不同，虽配置同权，实际贡献并不相同。

### 4.3 两视图对比目标

第二视图从同一实例的点云中分别对 surface/interior 做随机保留/再采样（约保留 90%，再采样回原点数），并在 XY 平面做旋转；属性随点保留，XYZ 旋转。两视图通过相同的 encoder 和 128D projection head，使用温度 0.1 的对称 in-batch NT-Xent。训练脚本中的对比项权重为 **0.1**，当前 batch size 为 8。

因此每个训练 batch 的目标可写成：

`total loss = geo + intensity + gradient + distance + surface + 0.1 × contrastive`

（各重建项另乘 YAML 配置权重；本次均为 1。）

这意味着对比学习的负样本主要来自同 batch 的其他实例；batch 较小，且同一 neuron 内线粒体可能相似，也可能带来 false negatives。当前尚未专门量化这一影响。

## 5. 当前这一轮训练的设置与结果

### 5.1 配置

配置文件：`/path/to/JaneliaEM/point-MAE/configs/mitochondria_large.yaml`

核心参数如下：

| 参数 | 当前值 |
|---|---:|
| preset | large |
| group 数 / 每组点数 | 64 / 32 |
| embedding 维度 | 128 |
| mask ratio | 0.60 |
| batch size | 8 |
| learning rate | 1e-4 |
| weight decay | 0.05 |
| gradient clipping | 1.0 |
| AMP | 开启 |
| workers | 4 |
| YAML 默认 epoch 数 | 100 |
| 本次实际训练 | 50 epoch（命令行覆盖） |

本次没有重新从 raw 数据制作新格式；直接使用已经构建的 Zarr cache。训练脚本和运行输出均位于服务器硬盘上的项目目录。根据当前 run 目录和配置重建的命令如下（原始 shell 历史没有随 run 一起保存，因此这是可复现命令写法，不声称是当时逐字执行的历史命令）：

```bash
python /path/to/JaneliaEM/point-MAE/train.py \
  --config /path/to/JaneliaEM/point-MAE/configs/mitochondria_large.yaml \
  --cache /path/to/project_data/EM/point-MAE/cache/manual391_plus_predE16_complete_min5000.zarr \
  --output /path/to/project_data/EM/point-MAE/runs/mito_e16_full_plus_manual_50ep \
  --device cuda --epochs 50
```

运行目录：`/path/to/project_data/EM/point-MAE/runs/mito_e16_full_plus_manual_50ep/`

包含 `best.pt`、`last.pt`、`metrics.jsonl`、`train.log`、`split.json`、`test_metrics.json`、`embeddings_all.npz` 和 diagnostics。`metrics.jsonl` 有 50 条 epoch 记录。当前最佳权重 `best.pt` 的 SHA-256 为：

`37ac3c32b0c249e3a8327f8938934fa6b1e035c307dfce41aed2b4eceae71bd0`

最佳 epoch 是编号 47（第 48 轮）；依据验证集 loss 选出。训练集 loss 继续缓慢下降，但验证 loss 在最佳轮之后略有反弹，所以最终保留 best，而不是直接取 last。

### 5.2 最佳轮与 test 指标

第 47 轮最佳 validation：总 loss 1.7322；重建 loss 1.7229；contrastive loss 0.0924。各重建项：geo 0.0463、intensity 0.2043、gradient 0.8852、distance 0.000476、surface 0.5867。

最佳 checkpoint 在 held-out test split 上：

| 指标 | test 值 |
|---|---:|
| 总 loss | 1.7346 |
| 重建 loss | 1.7294 |
| contrastive loss | 0.0520 |
| geo | 0.0355 |
| intensity | 0.2154 |
| gradient | 0.9070 |
| distance | 0.000326 |
| surface | 0.5712 |

Validation 和 test 总 loss 接近，是一个有用的 sanity check，但并不证明跨数据来源/细胞类型泛化良好。不同 loss 尺度不可直接横向当作“任务重要性”；不过在等权配置下，最佳验证轮的 gradient 与 surface 数值占重建总量大部分，distance 数值几乎为零，说明优化贡献严重不均衡。需要结合梯度量级与目标归一化进一步判断，而不能仅凭 loss 值判定训练 bug。

## 6. 训练后怎样得到 embedding 和 UMAP/cluster

### 6.1 embedding 提取

`extract_embeddings.py` 从 `best.pt` 加载完整训练权重，要求 projection head 是训练过的；进入 eval mode、关闭随机 masking，将每个实例完整输入 encoder，取 mean-pooled encoder 表示，经 128D projection 输出向量。结果存于 `embeddings_all.npz`，并保留 instance ID、neuron/source ID、organelle type 等元数据。

### 6.2 当前的 PC30 UMAP 和分群实验

为比较数据中的全局结构，当前采用：128D embedding 先 L2 normalize，再做 PCA 到 30 维（PC30 累计解释方差约 86.25%），再归一化并以 cosine 距离做 UMAP。UMAP 参数调整版是 `n_neighbors=30`、`min_dist=0.2`、seed 42。UMAP 是二维可视化，不是训练目标，也不保留全局距离的严格尺度；图上相邻或分开的区域不应单独作为类别证据。

已做的分群结果：

- PC30 上 K-means K=2：两组 1218、1339。bootstrap 子样本内部一致性较好（6 次平均 ARI 约 0.866），PC20/PC30/PC50 的两两结果也较接近；但 silhouette 只有约 0.053，与 average-linkage 和 30-NN spectral 的结果一致性低。它说明 K-means 在自身设定下可重复，不说明数据天然就是两个离散群。
- PC30 cosine kNN 图上的 Leiden：`n_neighbors=30, resolution=0.2` 可得到 4 群（851、694、608、404）；在 5 个 seed 的 ARI 约 0.489，扩展到 10 个 seed 后约 0.472±0.123，silhouette 约 0.028。改变邻居数到 50 后，seed 42 的分配 ARI 约 0.462；不同参数下成员会明显换组。
- 这版 Leiden 的一个群中手工标注占比 28.2%（240/851），其余群合计约 8.9%（151/1706），提示来源/标注策略可能与群结构有关。六种细胞类型在群中均有混合，尚未看到明确的一群对应一种 cell type。

因此，“UMAP 上看起来有几片区域”与“存在稳定的离散生物学群”是两个不同判断。当前 Leiden 是探索性结果，不能把它当作最终分类。

## 7. 已经执行的表示诊断及结果

诊断报告位于 run 目录的 `diagnostics/diagnosis_report.md`，统计文件与图也保存在 `diagnostics/` 下。下面是主要结果。

### 7.1 是否 representation collapse

对 2557 个不同实例的最终 embedding（总 pair 数 3,267,846）统计：

- 不同实例 median cosine similarity：**0.0051**。
- centered + L2-normalized 后 median cosine：**0.0089**。
- 未发现 cosine > 0.9999 的近重复 embedding。
- 每维标准差、范数分布和 PCA 均显示 embedding 非常量；effective rank 为 **27.93/128**，前 5 个 PC 累计解释约 **45.92%** 方差。

判断：**不是“全部 embedding 几乎相同”的全面 collapse；存在明显低秩/方差集中。** 仅凭 cosine 很低也不能说表征已具备足够的生物学区分力。

### 7.2 哪一层开始丢失区分度

层级统计（mean-pooled 特征）：

| 表示层 | effective rank | 不同实例 median cosine | 观察 |
|---|---:|---:|---|
| tokenizer pooled | 4.61 / 384 | 0.9670 | 各实例共享强烈的共同方向，区分度很弱 |
| encoder pooled | 5.46 / 384 | 0.5763 | 相对 tokenizer 有下降，但仍低秩且相似度偏高 |
| 最终 128D projection | 27.93 / 128 | 0.0051 | 整体更分散，秩提高；但不是说形态信息都保留 |

这个表要谨慎解读：mean pooling 会降低很多 token 变化；这里不代表未经 pooling 的所有 local token 也同样低秩。最终 projection 的低秩并非从 projection 才开始出现，且 cosine 变低并不等于语义结构正确。

### 7.3 checkpoint 与 projection head

checkpoint 配置与 `mitochondria_large.yaml` 相符，保存的 best 权重 SHA-256 已记录。projection head 在 4/4 参数检查中有非零梯度/更新迹象，因此**“projection head 完全没参与训练”不符合现有证据**。更准确的问题是：projection head 后的 morphology probe 表现弱于 encoder/tokenizer pooled 表示，可能发生了形态信息重排或损失，而不是 head 未训练。

### 7.4 Decoder 是否依赖 encoder，以及位置对应关系

用 108 个实例做固定 mask 的小规模 ablation：

- 正常 encoder tokens：作为参考重建 loss。
- shuffle 可见 encoder tokens：loss 变化约 **0.000%**。
- encoder tokens 置零：loss 增加约 **9.8%**。

解释：decoder 并非完全忽略 encoder（置零会变差），但相对位置/内容与具体 token index 的正确配对，对当前重建误差几乎没有影响。Decoder 可能更多利用位置编码、可见 token 汇总统计或自身先验来完成重建，而没有充分依靠“这个区域对应的这个 encoder token”。这是一个值得优先处理的已确认风险；ablation 只覆盖 108 个实例和当前实现，不能过度外推。

### 7.5 frozen linear probe：基础形态/纹理是否可读

用 neuron/source-group holdout 做 frozen probe。目标来自点云 proxy，而不是从原始 segmentation 精确计算的真实 volume/surface：

| 预测目标（proxy） | tokenizer pooled R² | encoder pooled R² | projection 128D R² |
|---|---:|---:|---:|
| bbox volume | 0.637 | 0.443 | 0.354 |
| convex-hull volume | 0.599 | 0.383 | 0.300 |
| elongation | 0.495 | 0.138 | 0.034 |
| convex-hull area | 0.897 | 0.783 | 0.569 |
| internal intensity contrast（std proxy） | 0.915 | 0.883 | 0.743 |

判断：模型中间层能读出部分大小、表面积与纹理代理量；最终 embedding 仍保留若干信息，但 elongation 几乎无法由简单线性 probe 读出。因为 probe 目标本身是 proxy，且 manual 数据按 dataset group 划分，以上数值是诊断证据而非严格生物学性能评估。

### 7.6 同实例增强稳定性和尺度敏感性

- 同一缓存点云两种重采样/旋转视图的最终 embedding cosine：median **0.9679**，5th percentile **0.8553**；不同 instance 的 median cosine 为 0.0051。表明当前增强下，同实例一致性明显高于实例间一致性。
- 把同一输入 XYZ 整体放大两倍，embedding cosine median **0.5207**。说明当前 embedding 对绝对尺度变化有相当敏感性；这是诊断扰动，不代表应当做尺度不变增强。
- 当前“重采样”是在已缓存的 2048 点上进行，而不是从原始 mask voxel 重新独立采样，因此不能完全回答“从 voxel 独立抽点是否稳定”。

## 8. 目前最值得注意的问题（按证据强弱）

### 已确认 / 强证据

1. **中间表示低秩且实例相似度高。** Tokenizer/encoder pooled rank 分别约 4.61/5.46；说明即使最终 embedding 不全面坍缩，局部 token 到 pooled global representation 的区分性仍值得优化。
2. **Decoder 对 token-position 对应关系不敏感。** Shuffle 近乎不改变重建 loss，置零只造成约 9.8% 损失上升。当前 MAE 目标可能允许 decoder 借助位置先验完成相当多的重建。
3. **最终 embedding 的线性形态可读性下降。** 尤其 elongation probe 从 encoder 的 R² 0.138 降到 projection 的 0.034。Projection 已训练，但其表示未必保留当前 probe 所关心的方向。
4. **重建损失项量级失衡。** 验证时 gradient/surface 占大头，distance 极小。是否是单位/权重的设计问题需要检查梯度，而不能只看 scalar loss。
5. **当前 split 与标签支持的结论有限。** 三个 test predicted neuron，以及 dataset-level manual split；manual 数据缺 cell type。当前不能据此声称六种细胞类型可分，也无法排除数据集来源混杂。
6. **Leiden/K-means 分群依赖算法与参数。** UMAP 区域不能作为聚类稳定性的替代证据，现有聚类不足以确认离散分群。

### 已有迹象、但需进一步确认

- **强制 50% surface/interior 采样**使实例层面的 surface 频率恒定，可能削弱不同表面/内部比例特征；但单点标签仍存在，所以是否实际伤害形态表征尚未做消融。
- **局部 intensity normalization**去掉了绝对灰度尺度；可能有助于跨 neuron 灰度稳定，也可能移除生物/成像差异。需要以固定 point sampling 做对照，而非直接改掉归一化。
- **小 batch 的 in-batch contrastive negatives**可能带来 false negatives 或弱对比信号；尚未按同 neuron/不同 neuron 关系分层测量。
- 52 个重复坐标、24 个 surface flag 不一致需查明是否由候选点重复采样/边界定义造成，以及是否集中在小实例或无 interior 退化案例。
- Leiden 的一个群富集 manual 来源，提示数据来源影响，但尚未通过 source-conditioned test 或标签置换排除其他解释。

### 当前无法由已有实验回答

- 模型能否区分用户肉眼判断的 1/2/3/4 类真实线粒体形态：尚无明确的独立人工类型标签，也无 blinded review。
- 精确体积、surface area、sphericity、分支/骨架等形态指标上的 embedding 性能：当前 point cache/probe 用 proxy，需回到 mask/mesh 计算。
- 六种细胞类型的 embedding 是否存在稳定差异：manual 数据无细胞类型标签，预测数据 neuron 数量与类型分布不平衡，且尚未做 neuron-held-out 的正式检验。
- 是否应调整 mask ratio、重建 loss 或网络结构：用户此前明确先 debug 其他问题，尚未做这些改动的对照训练。

## 9. 建议的下一步验证顺序

在动模型和开始新一轮训练前，建议先完成以下闭环：

1. **从原始 instance mask 计算准确的目标量**：physical volume、mesh surface area、PCA elongation、sphericity；再与当前 proxy 对照。选取不同体积/形态代表实例做 3D 可视化，检查边界完整性和 voxel spacing。
2. **做同一实例的真正独立采样稳定性实验**：每次从 mask voxel 重新采样两组点，不复用缓存点；分层比较大/小、细长/球状实例，并与不同实例距离作参照。
3. **沿层级重复 frozen probes 和邻域检索**：对手工的形态类别（如果可以人工确认）、准确形态指标、source、cell type 分别做 neuron-held-out 评估；比较 tokenizer、encoder、projection 三层，明确信息损失发生在哪一段。
4. **针对 decoder 做位置/信息消融扩展**：增加打乱位置编码、保留 token 内容但打乱位置、仅位置编码等基线，分层记录各重建项；仍用固定 checkpoint/固定 mask，先不全量重训。
5. **做数据来源与细胞类型混杂检查**：分别在预测数据内、手工数据内评估，再做 source-stratified UMAP/neighborhood；仅在有足够独立 neuron 的 cell type 上进行 neuron-level holdout。
6. 根据前述结果再决定是否改 loss 尺度、采样比例、mask ratio 或 pooling。每次只改一个因素，保留当前 checkpoint 作为基线。

这套顺序能区分三种情况：输入本身没编码人眼关注的差异；中间表示有差异但 pooling/projection 抹掉；或者 embedding 中有信息但当前 UMAP/聚类方法没把它展示出来。

## 10. 当前代码文件索引

以下均是服务器项目 `/path/to/JaneliaEM/point-MAE/` 中当前存在的源文件。行号对应本次整理时服务器上的文件版本，代码改动后可能移动。

| 文件 | 职责与建议查看位置 |
|---|---|
| `pointmae.py` | **核心实现。** `sample_instance()` 第 97–175 行：tight bbox crop、7 通道、sampling、局部强度统计；第 205–240 行：按 neuron/group split；第 272–320 行：同实例第二视图增强和 NT-Xent；第 337–433 行：Point-MAE、encoder/projection、mask/decoder 与各项 reconstruction loss。 |
| `preprocess.py` | 将全 volume 的 instance-label 与 raw EM 转成 Zarr point-cloud cache。第 61 行用 `ndimage.find_objects` 建 instance bbox，第 96–107 行把 bbox 交给 `sample_instance()`。 |
| `extract_complete.py` | 手工标注 volume 的完整实例筛选：排除碰到 volume 六个面边界的实例，并可按 min voxel 数过滤。 |
| `import_predicted_instances.py` | 将预测实例与 raw EM blocks 合并成点云 cache。第 31–44 行判定跨 block 完整性；第 47–66 行按预测 bbox 拼接 raw crop；第 129–146 行执行 min-voxel 过滤、加载 mask 并调用 `sample_instance()`。 |
| `merge_caches.py` | 合并多个 Zarr point-cloud caches，并重排 `records.jsonl` 中的 cache index。 |
| `configs/mitochondria_large.yaml` | 本轮 large 模型配置：点属性、spacing、group 数/大小、mask ratio、embedding 维度及训练超参数。 |
| `train.py` | 训练入口。第 28–82 行为一个 epoch 的 MAE + 对比损失；第 85 行起加载 cache、按 neuron/source 划分、建立 optimizer；第 166–190 行控制 epoch、保存 best/last checkpoint 并评估 test。 |
| `extract_embeddings.py` | 从 checkpoint 提取无 mask 的 128D embedding。第 32–45 行加载权重、检查 projection head、调用 `model.encode()` 并写 NPZ。 |
| `diagnose_pointmae_representation.py` | 当前表示诊断入口；检查 cache/manifest 对应、逐层 embedding 统计、probe、checkpoint、loss 和相关消融，写出 diagnosis report 与 CSV/JSON。 |
| `pointmae_embedding_stability.py` | 检查同一实例不同采样/旋转的 embedding 稳定性；`--cache-only` 模式使用现存点云增强视图，不重新从原始 mask 采样。 |
| `make_embedding_umap.py` | 项目内通用的直接 UMAP 与 PCA→UMAP 绘图脚本，支持设置 neighbors、min_dist 和 PCA 维数。**当前报告中的 PC30 调参版和 Leiden 参数扫描结果是单独的分析产物，不应误认为由这个脚本完整复现。** |
| `README.md` | 项目约定、cache 生成和基础运行说明；局部 crop 的精确定义以 `pointmae.py::sample_instance()` 为准。 |

## 11. 本轮训练和分析产物位置

训练运行目录：`/path/to/project_data/EM/point-MAE/runs/mito_e16_full_plus_manual_50ep/`  
核心结果：`best.pt`、`last.pt`、`metrics.jsonl`、`test_metrics.json`、`split.json`、`embeddings_all.npz`  
诊断报告：`runs/mito_e16_full_plus_manual_50ep/diagnostics/diagnosis_report.md`  
PC30 Leiden 对比图（本地副本）：`pointmae_e16_full_training_20261010/diagnostics/umap_pc30_leiden_n30_vs_n50_res0p2.png`  
Leiden 参数稳定性表（本地副本）：`pointmae_e16_full_training_20261010/diagnostics/leiden_pc30_parameter_stability.csv`

---

**一句话总结：** 当前 Point-MAE 已完成 50 epoch 训练，projection 确实参与训练，最终向量并非完全坍缩；但 tokenizer/encoder pooled 表示低秩、decoder 对 token 位置对应不敏感、最终 projection 对 elongation 等形态指标的线性可读性下降，以及数据来源/标签和聚类稳定性限制，都是已有实验指出的主要风险。下一步应先用原始 mask 精确形态指标和更严格的分层验证确认问题位置，再决定是否改训练目标或结构。
