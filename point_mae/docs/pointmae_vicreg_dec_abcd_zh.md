# Point-MAE、VICReg 与 DEC：A/B/C/D 实验架构及结果说明

日期：2026-10-12  
项目：`/path/to/JaneliaEM/point-MAE`  
对象：线粒体点云无监督表征与聚类对照实验

## 1. 先看结论

这次比较的重点不是“哪张 UMAP 看起来分得最开”，而是区分三件事：

1. 表征是否保留了不同线粒体之间的信息；
2. 聚类结构是否会随训练随机种子改变；
3. 得到的分群是否能由形态或细胞类型等独立信息解释。

目前结果显示：

- 如果“好”指**跨随机种子稳定**，A（MAE baseline）的 K=5 最好，ARI 为 `0.916 ± 0.020`、NMI 为 `0.893 ± 0.022`。
- 如果“好”指**当前 embedding 空间中的簇分离度**，C（MAE+DEC）的 K=5 cosine silhouette 最高，约 `0.753`。但 DEC 直接优化聚类分配，因此 silhouette 不是独立验证；同时 C 的 Encoder 有效秩较低。
- B（MAE+VICReg）没有提高跨种子聚类稳定性。D（MAE+VICReg+DEC）在部分指标上介于 B 与 C 之间，但没有稳定地超过 A 或 C。
- 在有已知细胞类型标签的预测实例子集上，各方法与细胞类型标签的 ARI 约 `0.02–0.04`，尚无明显细胞类型分群证据。

因此，当前建议把 **A/K=5 作为稳定基线**，把 **C/K=5 作为分离度较高、但需要额外形态学验证的对照**。这不是生物学类别已经得到确认的结论。

## 2. 数据和每个线粒体的输入

本次使用的缓存包含 2,558 个线粒体、27 个 neuron ID。数据来源包括预测实例与手工标注实例。所有方法使用相同缓存、相同 neuron-level 划分和相同预训练初始化。

每个实例以 `2048 × 7` 点云输入模型：

| 通道 | 含义 | 生成方式 |
|---|---|---|
| 坐标（3维） | 物理坐标 | 按体素间距 `[30, 8, 8] nm` 转成物理坐标，以实例质心为原点，再除以 1000；数组顺序保留为 Z/Y/X |
| Intensity | EM 强度 | 在实例 bounding crop 内以 median/MAD 做 robust normalization，并截断到 `[-5, 5]` |
| Gradient | 强度梯度幅值 | 按物理间距计算的 3D gradient magnitude |
| Distance | 到实例边界的距离 | 物理单位的 Euclidean distance transform，再除以 1000 |
| Surface | 表面/内部标记 | 对实例 mask 做 3×3×3 erosion；边界为 surface，腐蚀后剩余区域为 interior |

点采样分成约 1,024 个 surface 点和 1,024 个 interior 点，分别在物理坐标下用 farthest-point sampling 选取；候选点不足时允许重复采样。之后每个样本用 FPS 选 64 个 patch 中心，每个中心取最近的 32 个点。不同 patch 可以重叠。

## 3. 四组共用的 Point-MAE 主干

### 3.1 主干结构

```text
2048 个带属性的点
  → FPS 选 64 个 patch 中心
  → 每个中心取最近 32 点
  → 局部坐标 + 4 个属性，经 point MLP 后对 32 点 max-pool
  → 加入中心坐标的位置编码
  → 384 维、10 层、8-head Transformer Encoder
  → [MAE 训练时接 Decoder 重建；表征提取时对 64 个 token 做 mean pooling]
```

本次 checkpoint 的 `large` preset 参数如下：

| 模块 | 结构 |
|---|---|
| Patch 数量/大小 | 64 groups × 32 points |
| Point tokenizer | 每点 7 维输入，经 `7→384→384` MLP；对组内 32 点做 max pooling |
| Position encoder | 中心坐标经 `3→384→384` MLP，加到局部 token 上 |
| Encoder | 384 维，10 层 Transformer，8 个 attention heads，FFN 宽度 1536 |
| MAE Decoder | 192 维，4 层 Transformer，8 heads，FFN 宽度 768 |
| Reconstruction head | 每个 masked patch 预测 `32 × 7` 个坐标与属性值 |

### 3.2 MAE 重建路径

预训练 checkpoint 的 mask ratio 是 0.6。每个实例有 64 个 patch，训练时随机 mask 约 38 个，Encoder 只处理剩余可见 patch；Decoder 接收可见 patch 的 Encoder 特征、mask token 和位置编码，预测所有 patch，再只在被 mask 的 patch 上计算重建损失。

损失包括：

- 几何：预测点集与目标点集之间的双向 Chamfer distance；
- intensity、gradient、distance：按预测坐标最近邻匹配目标属性，再计算 Smooth L1；
- surface：同样按最近邻匹配后，用 binary cross-entropy with logits。

这些项的基础权重均为 1。A/B/C/D 都保留这条 MAE 重建路径；MAE reconstruction 是四组共享的训练目标。

### 3.3 本次下游表征到底取哪一层

Encoder 输出的是 `B × 64 × 384` 的 contextualized patch tokens。对 64 个 token 做 mean pooling 得到：

```text
h = mean(Encoder tokens) ∈ R^384
```

这次实验把 `h` 作为线粒体的全局表征：DEC 使用它，最终聚类分析也使用它。提取时不做 MAE masking，模型处于 eval 模式。

原 Point-MAE 类中还定义了 `384→384→128` 的 projection MLP。这里要区分“预训练是否训练过它”和“A/B/C/D 微调时是否使用它”：初始化 checkpoint 的预训练目标包含两视图 NT-Xent 对比损失，`model.encode()` 会经过该 projection，因此 128 维头当时确实接受过梯度并被训练；但 MAE 重建分支本身不经过它。A/B/C/D 脚本加载并保留这组预训练 projection 权重，随后自定义训练路径直接从 Encoder token pooling 取得 `h`，不再对旧 128 维头反向传播，也不将它用于 VICReg 或 DEC。B/D 另有一个新建的 VICReg projector：`384→384→32`，只服务于 VICReg loss。因此本实验涉及三个量：预训练过的 128D projection 输出、A/B/C/D 实际使用的 Encoder 表征 `h384`、B/D 新增的 VICReg 专用 `p32`；聚类看的是 `h384`。

## 4. A/B/C/D 的模型差异

四组**没有更换 Point-MAE Encoder/Decoder 架构**。差异在于训练时是否增加 VICReg projector、DEC 聚类中心，以及相应损失。

### A：Point-MAE baseline

```text
点云 → Point-MAE Encoder/Decoder → MAE reconstruction loss
```

- 只用 MAE 损失微调 Point-MAE。
- 训练结束后提取 `h384`，在训练集的 `h` 上拟合 KMeans，再给全部样本分配 K=5/10/20 标签。
- KMeans 是训练后的分析步骤，不是 A 的训练损失。

### B：Point-MAE + VICReg

```text
两种同实例 view → 共享 Point-MAE Encoder → h384 → VICReg projector p32
                                                   ↘ VICReg loss
同一模型的 masked reconstruction → MAE loss
```

- 一个 batch 中的每个线粒体独立生成两个 view，共享同一个 Encoder。
- 每个 view 对 surface/interior 两组点分别保留约 90%，再补采样回原点数；随后只对 Y-X 平面做随机旋转，点的属性随点同步重排。这里没有加入尺度变化或强随机噪声。
- `h384` 经独立 `384→384→32` MLP（含 BatchNorm/GELU）得到 `p32`，VICReg loss 只作用于 `p32`。
- 训练后仍在 `h384` 上做 KMeans；不是用 `p32` 做最终聚类。

### C：Point-MAE + DEC

```text
点云 → Point-MAE Encoder → h384 → Student-t assignments q
                                      ↕ KL(q || target distribution p)
                           可训练的 K 个聚类中心
MAE reconstruction loss 与 DEC loss 联合更新 Encoder
```

- 先在初始 Encoder 的训练集 embedding 上做 KMeans，以其中心初始化 DEC 聚类中心。
- Student-t kernel 根据 `h384` 与聚类中心的距离计算软分配 `q`。
- 每个 epoch 用 `p ∝ q² / cluster_frequency` 更新 target distribution，再以 KL divergence 训练聚类分配。
- 训练期间可训练 Encoder、MAE Decoder 和 DEC centers；没有 VICReg head。
- K=5、10、20 分别训练独立的 C 模型，不是同一模型输出三个 K。

### D：Point-MAE + VICReg + DEC

```text
两种 view → 共享 Encoder → h384 → VICReg 专用 p32 → VICReg loss
                        └──────→ DEC centers / KL loss
Point-MAE masked reconstruction ─────────────────→ MAE loss
```

- 同时包含 B 的 VICReg 分支和 C 的 DEC 分支。
- VICReg 作用于 `p32`，DEC 作用于 `h384`，MAE 作用于重建输出。
- K=5、10、20 也分别训练独立模型。

## 5. 损失、初始化与训练设置

总损失为：

```text
L = L_MAE + 0.05 × L_VICReg + 1.0 × warmup(epoch) × L_DEC
```

不存在的分支损失在对应组中省略：A 只有 `L_MAE`；B 有 `L_MAE + 0.05 L_VICReg`；C 有 `L_MAE + warmup·L_DEC`；D 三项都有。

VICReg 内部为：

```text
L_VICReg = 25 × invariance + 25 × variance + 1 × covariance
```

其中 invariance 是两个 view 的投影向量均方误差；variance 惩罚每个维度标准差低于 `gamma=1`；covariance 惩罚非对角协方差。外层 `λ=0.05` 再缩放整个 VICReg loss。

DEC 的外层 `λ=1`，前 5 个 epoch 线性 warm-up，之后保持完整权重；Student-t 参数 `alpha=1`。没有均衡 cluster 数量的约束。

训练设置：

| 项目 | 设置 |
|---|---|
| 初始化 | 同一个 Point-MAE checkpoint；其训练实验为 50 epoch，使用 epoch 42 的 best checkpoint。预训练包含 MAE 重建与两视图 NT-Xent（权重 0.1）；因此初始 checkpoint 中的 128 维 projection 已训练 |
| 随机种子 | 42、123、2026 |
| 每次训练轮数 | 20 epoch（命令行覆盖 YAML 中的默认 10） |
| Batch size / optimizer | 32 / AdamW |
| Learning rate / weight decay | `1e-5` / `0.01` |
| Gradient clipping | 1.0 |
| Train/validation/test | 1969 / 330 / 259 个实例，按 neuron ID 划分，三个随机种子使用同一 split |
| 聚类损失使用数据 | 仅 train split；validation 用于 MAE loss 监控 |

每个 seed 训练 8 个模型：A、B 各一个；C 和 D 各针对 K=5/10/20 训练一个。三种子合计 24 次模型训练。正式运行中的 A/B/C/D 都从同一个预训练 checkpoint 独立初始化，不是顺序接着上一个方法训练。A/B/C/D 新阶段的 MAE 重建不会更新旧 128 维头；B/D 的 VICReg 更新的是新建 32 维 projector，DEC 更新的是直接作用于 `h384` 的聚类中心。

### 关于预训练阶段的 128 维 projection（重要澄清）

预训练训练脚本中，每个 batch 先计算 MAE 重建损失，再对同一批线粒体生成第二种采样/旋转视图：两种视图分别调用 `model.encode()`，而该函数返回 `projection(mean(Encoder tokens))`，即 128 维向量。两视图向量经过对称 in-batch NT-Xent，温度为 0.1；总损失为 `L_MAE + 0.1 × L_NT-Xent`。所以 projection 不依赖 MAE reconstruction 获得梯度，但会依赖 NT-Xent 获得梯度。训练日志记录了非零 `contrastive_loss`，模型代码也显示 NT-Xent 分支确实经过 projection。

A/B/C/D 是加载这份已进行对比预训练的 checkpoint 后开展的新对照阶段。新阶段为了回答“辅助损失是否改善 Encoder 本身的聚类表征”，统一在投影前的 `h384` 上计算 DEC 和评价聚类；这是实验设计选择，并非因为此前的 128 维头从未训练。

## 6. 正式实验结果

### 6.1 跨随机种子一致性

每一格是三组 seed pair（42–123、42–2026、123–2026）的 ARI/NMI 均值 ± 标准差；比较时按 `cache_index` 对齐全部 2,558 个样本。ARI/NMI 对聚类编号置换不敏感。

| 方法 | K | ARI 均值 ± SD | NMI 均值 ± SD |
|---|---:|---:|---:|
| A：MAE | 5 | **0.916 ± 0.020** | **0.893 ± 0.022** |
| A：MAE | 10 | 0.654 ± 0.101 | 0.741 ± 0.063 |
| A：MAE | 20 | 0.574 ± 0.037 | 0.727 ± 0.018 |
| B：MAE+VICReg | 5 | 0.429 ± 0.029 | 0.470 ± 0.021 |
| B：MAE+VICReg | 10 | 0.484 ± 0.048 | 0.598 ± 0.025 |
| B：MAE+VICReg | 20 | 0.425 ± 0.044 | 0.638 ± 0.026 |
| C：MAE+DEC | 5 | 0.735 ± 0.110 | 0.748 ± 0.084 |
| C：MAE+DEC | 10 | 0.593 ± 0.123 | 0.660 ± 0.070 |
| C：MAE+DEC | 20 | 0.484 ± 0.037 | 0.663 ± 0.024 |
| D：联合 | 5 | 0.751 ± 0.060 | 0.738 ± 0.048 |
| D：联合 | 10 | 0.569 ± 0.116 | 0.661 ± 0.073 |
| D：联合 | 20 | 0.486 ± 0.027 | 0.671 ± 0.017 |

### 6.2 Silhouette 与有效秩

Silhouette 是在归一化后的 Encoder `h384` 上，以 cosine 距离抽样 500 个实例计算。有效秩从全部 2,558 个 `h384` 的中心化协方差特征值计算。A/B 的有效秩对 K 不变，因为 KMeans 不会回写 Encoder；C/D 的有效秩随各自 K 的独立训练而变化。

| 方法 | Encoder 有效秩（K=5/10/20） | Silhouette（K=5/10/20） |
|---|---|---|
| A：MAE | 5.50 / 5.50 / 5.50 | 0.329 / 0.299 / 0.290 |
| B：MAE+VICReg | 6.05 / 6.05 / 6.05 | 0.346 / 0.342 / 0.306 |
| C：MAE+DEC | 2.96 / 3.43 / 3.96 | **0.753** / 0.526 / 0.379 |
| D：联合 | 3.38 / 3.90 / 4.24 | 0.519 / 0.421 / 0.341 |

Silhouette 是辅助指标。C 的 DEC loss 明确训练聚类分配，因此它的高 silhouette 不能单独证明发现了独立存在的生物学亚型。所有实验都没有空 cluster 或单一 cluster 占超过 90% 的严重坍塌；但 DEC 在 K=10/20 时偶尔会形成较小簇。

### 6.3 可以和不可以得出的结论

**可以得出：**

- A/K=5 对这三个训练随机种子最稳定；K 从 5 增到 10/20 后一致性降低。
- 本次 VICReg 的实现没有让聚类更稳定，B 的跨种子 ARI/NMI 低于 A。
- DEC 提高了 embedding 上的内部 cluster separation 指标，但伴随更低的 Encoder 有效秩；D 没有稳定超过 C 或 A。
- 在有已知细胞类型标签的预测实例子集上，不同方法的细胞类型 ARI 约 0.02–0.04，当前分群不明显对应已知细胞类型。

**尚不能得出：**

- A/K=5 的五个簇就是五种真实线粒体亚型；稳定性不等于生物学正确。
- C 的 silhouette 更高，因此 C 的生物学分群就更好；该指标受 DEC 优化目标影响。
- 当前模型已经充分利用所有 morphology/texture 属性。需要进一步比较 cluster 与体积、表面积、长宽比、球形度等连续指标的关系，并检查不同 neuron 上是否复现。

## 7. 主要实现位置与产物

服务器项目代码：

- `pointmae.py`：点采样、Point-MAE 主干、MAE 重建损失，以及 VICReg view augmentation 函数。
- `train_deep_clustering.py`：A/B/C/D 训练入口、VICReg projector、DEC head、损失组合和评估。
- `configs/mitochondria_vicreg_dec.yaml`：VICReg/DEC 参数与训练默认配置。

三种子汇总：

```text
/path/to/project_data/EM/point-MAE/runs/pointmae_vicreg_dec_24neurons_formal20ep_3seeds_20261011/
```

其中 `seed_stability_summary.csv` 是跨种子 ARI/NMI 汇总，`seed_stability_pairs.csv` 是逐 pair 结果，`metrics_summary.csv` 是跨 seed 指标均值/标准差，`metrics_by_seed.csv` 是逐 seed 明细。

单个 seed 的输出目录保存每组的 `last.pt`、`metrics.jsonl`、`training_summary.json`、`embeddings_encoder.npz`、`assignments.csv`、`assignment_probabilities.npz`、`cluster_morphology_summary.csv` 和 K=10 的 PCA/UMAP 与代表性点云图。例如 seed42 位于：

```text
/path/to/project_data/EM/point-MAE/runs/pointmae_vicreg_dec_24neurons_formal20ep_seed42_20261011/
```

## 8. 下一步建议

1. 先用 A/K=5 作为重复性基线，用 C/K=5 作为“DEC 增强分离度”的对照。
2. 比较这些簇的体积、表面积、elongation、sphericity、ellipsoid agreement 等形态分布，并按 neuron 分层检查，避免结果被少数 neuron 驱动。
3. 对 B/D 做 VICReg 权重与增强强度的小范围消融；当前结果不支持直接把 VICReg 当作稳定性改进。
4. 在形态学关联明确前，不把 UMAP 分开的区域或 DEC 的 silhouette 直接命名为线粒体类别。
