# 线粒体 Scale-aware Voxel MAE：完整模型与训练说明

更新时间：2026-10-14（北京时间）  
代码位置：`/path/to/JaneliaEM/point-MAE/mito_voxel_mae/`  
本文件整合了现有代码、数据准备结果、sanity check 结果以及
`runs/voxel_mae_native_v1_20261014/mito_training_report_20261014.md`。

> 说明：原始训练报告仍然保留在运行目录中。本文件是面向后续阅读和改进的总说明，不替换原始逐轮训练报告。

---

## 1. 这一次模型到底解决什么问题

这套代码是在原来 Point-MAE 点云模型基础上重新设计的一个独立版本，目标是：

1. 直接使用线粒体原分辨率的 EM 体素和实例 mask；
2. 不再把一个线粒体压缩成固定数量的 2048 个带属性点；
3. 让模型显式感知线粒体的物理尺度和各向异性体素间距；
4. 使用 masked reconstruction 学习线粒体的三维外形、EM 灰度和尺寸；
5. 最终输出一个可以用于后续 PCA、UMAP、聚类或线性 probe 的 256 维 Encoder latent。

因此，它不是对旧 Point-MAE 的一两个参数修改，而是一个“原生体素上下文 + 稀疏几何支持点 + Transformer 编码 + 查询式体素重建”的新模型。

当前版本没有加入 VICReg、NT-Xent、DEC 或其他聚类损失。它目前首先验证体素重建和表征学习链路是否正确。

---

## 2. 与原 Point-MAE 的核心区别

| 内容 | 原 Point-MAE | 当前 `mito_voxel_mae` |
|---|---|---|
| 输入 | 固定数量的点，约 `2048 × 7` | 原分辨率 EM crop、mask、valid mask |
| 点的来源 | surface/interior 等点采样 | 规则网格支持点 + 一层邻域 halo |
| 强度 | 每个点携带局部归一化强度等属性 | U-Net 从原生 EM 体素提取上下文特征 |
| 尺度 | 容易受 instance-level 坐标归一化影响 | 显式输入物理 bbox 尺寸，单位为微米的 log-size |
| Encoder | 点 tokenizer + Transformer | 各向异性 3D U-Net + 支持点 tokenizer + Transformer |
| Masking | 点/patch 层面的遮挡 | 在 U-Net 前对 `4×4×4=64` 个空间 block 遮挡 38 个 |
| 重建方式 | 重建点或 patch | 对随机查询位置预测 occupancy 和 normalized intensity |
| 主表征 | 旧流程曾使用 128D projection | 当前用于下游分析的是 256D Encoder latent |
| 训练目标 | Point-MAE reconstruction | occupancy + Dice + intensity + physical size |
| 训练数据 | 固定点数，容易重复采样 | 可变大小的 native-resolution crop，batch size=1 |

当前模型保留了 MAE 的基本思想：输入被遮挡，Encoder 根据可见上下文生成 latent，再用 Decoder 恢复被遮挡区域；但输入单元和 Decoder 形式已经变成体素/查询式重建。

---

## 3. 代码文件总览

代码根目录：

```text
/path/to/JaneliaEM/point-MAE/mito_voxel_mae/
```

主要文件如下：

| 文件 | 作用 |
|---|---|
| `prepare.py` | 从原始实例和神经元 raw EM block 生成 native-resolution context cache |
| `merge_prepared.py` | 审核 cache 是否与原始 mask/前景 EM 一致，并生成最终训练 manifest |
| `data.py` | 数据集、增强、支持点生成、masking、query 采样和 batch 整理 |
| `model.py` | U-Net、支持点分组、Transformer Encoder、latent、两个 Decoder 和 loss |
| `train.py` | 训练、验证、checkpoint、学习率调度和 JSONL 日志 |
| `checks.py` | 坐标、增强、masking、梯度、padding、query chunk 等测试，以及单实例 overfit |
| `audit.py` | 再次逐实例核对 cache 与原始数据 |
| `export.py` | 导出无掩码 embedding，或对一个实例进行密集 voxel 查询 |
| `launch_prepare.sh` | 启动数据准备 |
| `launch_audit.sh` | 启动 cache 审核与 manifest 合并 |
| `launch_train.sh` | 启动正式训练 |
| `README.md` | 简要英文设计说明 |
| `shorten_training_to_10.py` | 将原计划 100 轮的运行安全地切换为总计 10 轮 |

当前模型的关键实现集中在 `data.py` 和 `model.py`，训练行为集中在 `train.py`。

---

## 4. 数据来源与数据准备

### 4.1 原始数据

本次训练使用：

```text
实例数据：/mnt/data21T_3/cyf/EM/neuron_mito_dataset/
raw EM：  /mnt/data21T_3/cyf/EM/neuron/
```

实例 metadata 位于：

```text
/mnt/data21T_3/cyf/EM/neuron_mito_dataset/metadata/instances.csv
```

每个实例至少关联：

- `record_id`
- `body_id`
- `cell_type`
- 实例 mask 路径
- 实例 EM 路径
- 原始 bbox
- 原始 voxel count
- raw source neuron 的状态

### 4.2 物理坐标和体素间距

当前代码统一使用 ZYX 数组顺序，体素间距为：

```text
Z = 30 nm
Y = 8 nm
X = 8 nm
```

也就是：

```python
spacing_zyx_nm = [30.0, 8.0, 8.0]
```

这是一个明显的各向异性体数据。代码中的卷积下采样只发生在 XY 平面，不在 Z 方向做 pooling，以避免把 Z 方向的 30 nm 间距错误地当作 8 nm 处理。

### 4.3 native-resolution context crop

对每个实例，首先使用原始 mask 的 tight bbox。然后为它增加上下文 margin：

```python
margin = ceil(native_shape * (2 / 17)) + 2
context_shape = native_shape + 2 * margin
```

这样做的原因是支持点使用大约 17 个网格单元；边界附近需要额外的 raw EM 上下文和邻域支持。

重要设计：

- 不对实例 resize；
- 不把不同大小的实例强行拉成相同体素尺寸；
- 不把大实例静默裁剪成较小的实例；
- context crop 的总大小必须不超过 1,500,000 voxels；
- 超过预算的实例暂时排除，未来可以使用 tiled training 扩展。

### 4.4 raw EM 和 valid mask 的拼接

raw neuron 由多个 block 组成。`prepare.py` 根据 block 目录名中的坐标，把与实例 context crop 相交的 raw block 拼接为一个 ZYX 数组：

- `em.npy`：拼接得到的 uint8 raw EM；
- `mask.npy`：context 内的实例 occupancy mask；
- `valid.npy`：该位置是否确实有 raw EM 来源。

如果 foreground mask 落在 raw EM 缺失区域，实例会被拒绝，因为这会导致模型面对“有目标但没有输入”的矛盾监督。

对于背景区域，如果 raw EM 缺失，则保留在 `valid=False` 状态，并在训练 loss 中排除对应 query。这样不会把缺失的 raw EM 错误地当作真实黑色背景。

### 4.5 强度归一化

当前的强度处理分两步：

#### 第一步：source-level percentile clipping

对于每个 body，代码从 raw source block 中使用确定性子采样估计：

- 下限：p0.5；
- 上限：p99.8；
- 超出范围的值 clip 到 `[0, 1]`。

为了避免扫描一个 body 的所有 raw block，估计使用：

- 随机种子 42；
- 最多抽取 4 个 raw block；
- 每个 block 使用 stride 4；
- 每个 block 最多保留 32768 个采样值。

因此，这些百分位是可复现的近似估计，不是整卷精确百分位常数。

#### 第二步：每个实例 tight crop 内的 valid z-score

实例读取后，在 valid 的 tight crop 区域统计均值和标准差：

```python
image = (image - mean_valid_crop) / std_valid_crop
```

最后，所有 `valid=False` 位置的输入强制设为 0。

这意味着当前模型不是直接以原始 uint8 EM 数值学习，而是学习经过 source clipping 和 instance crop z-score 后的强度。`export.py` 导出的 intensity 也是这种 normalized intensity，不是可直接解释为原始灰度百分比的数值。

### 4.6 数据数量和划分

本次 cache 和训练 manifest 的统计为：

| 数据 | 实例数 | 神经元数 |
|---|---:|---:|
| 原始实例 | 9,846 | 98 |
| 成功准备并进入 cohort | 8,822 | 98 |
| train | 7,126 | 78 |
| validation | 847 | 10 |
| test | 849 | 10 |

划分是 body-level disjoint：同一个神经元的实例不会同时出现在 train、val、test 中。

另有：

- 1,024 个实例因为 context crop 超过 1,500,000 voxels 被排除；
- 3,140 个可用实例的 context 中存在部分 raw EM 缺失；
- 所有 8,822 个进入 cohort 的实例都经过了 cache 与原始 mask/前景 EM 的一致性检查；
- `merge_prepared.py` 报告 `rejected_cache_records=0`。

---

## 5. 单个实例从读取到训练样本的完整流程

下面按实际执行顺序描述。

### 步骤 1：读取缓存

`MitoDataset.__getitem__` 读取：

```text
em.npy
mask.npy
valid.npy
prepared.json
```

然后根据 metadata 中保存的 `lo`、`hi` 做 source clipping 和 `[0,1]` 映射。

### 步骤 2：训练增强

训练集每个实例每轮产生 4 个在线 view。四种 view 大致是：

| view | 操作 |
|---|---|
| 0 | 原始几何，可能叠加输入噪声 |
| 1 | 小幅旋转、缩放、剪切和平移 |
| 2 | 主要在 XY 平面做更明显但仍然较小的旋转和平移 |
| 3 | 平滑的轻微非线性形变 |

增强时 `em`、`mask`、`valid` 同步变换。增强后重新计算 tight bbox 和物理 size。

如果增强使实例碰到边界、出现截断、foreground 落入无效 raw 区域或实例完全消失，则退回 clean geometry。随机噪声只加到模型输入 image，不加到 target intensity。

这 4 个 view 是同一个生物学实例的不同训练观察，不等于把数据集的独立线粒体数扩大了 4 倍。

### 步骤 3：生成规则支持点

当前版本没有旧 Point-MAE 中“1024 个 surface 点 + 1024 个 interior 点”的采样逻辑。

`point_support()` 做的是：

1. 在 tight bbox 中建立 17×17×17 的规则候选网格；
2. 在网格外侧加入一层 halo；
3. 用 mask 判断哪些候选网格属于实例或邻近区域；
4. 对 halo 进行 26 邻域膨胀；
5. 去除重复 voxel 坐标；
6. 只保留有效的支持点。

实现上会枚举包含边界 halo 的扩展网格，但只有 tight bbox 内部网格作为 seed，halo 点只能作为邻域支持点。最终支持点数可变，取决于线粒体形状、大小和边界。

每个支持点保存三类信息：

#### `points_t`：轴向归一化坐标

```text
t = (voxel_center - tight_bbox_origin) / tight_bbox_shape
```

它描述点在当前实例 bbox 中的相对位置。

#### `points_r`：保持物理比例的坐标

```text
r = t * (size * spacing) / max(size * spacing)
```

它保留 Z/Y/X 物理尺寸之间的比例，同时将最大物理轴归一化到约 1。

#### `points_grid`：给 3D U-Net 采样用的网格坐标

它转换为 `grid_sample` 需要的 XYZ 顺序和 `[-1,1]` 坐标。

### 步骤 4：空间 block masking

每个 tight crop 被划分为：

```text
4 × 4 × 4 = 64 个空间 block
```

其中随机隐藏 38 个 block，保留 26 个可见 block：

```text
mask ratio = 38 / 64 ≈ 59.4%
visible ratio = 26 / 64 ≈ 40.6%
```

因此当前模型不是 Point-MAE 论文中常说的 80% mask ratio；当前实现使用的是大约 59.4% 的空间遮挡。mask ratio 后续仍然可以作为实验变量，但本次训练没有修改它。

mask 在 U-Net 之前应用：

```text
input_image[~observed] = 0
```

送入 U-Net 的三个输入通道是：

```text
channel 0: masked / normalized EM image
channel 1: observed mask，当前 block 是否可见
channel 2: image_valid，raw EM 是否真实存在
```

模型没有通过 raw skip connection 直接把被遮挡的原图传给 Decoder。

### 步骤 5：随机采样 Decoder 查询位置

每个样本随机采样 8192 个 tight bbox 内的 voxel 位置：

```text
q: 整数 voxel 坐标
q_t: bbox 归一化坐标
q_r: 保持物理形状比例的坐标
```

每个 query 具有：

- `occupancy`：该位置是否属于线粒体 mask；
- `intensity`：归一化后的目标 EM 强度；
- `query_valid`：该位置的 raw EM 是否真实存在；
- `query_hidden`：该位置是否落入隐藏 block。

隐藏区域 query 在 loss 中权重更高，以避免模型只依赖容易的可见位置。

---

## 6. 模型架构：每一级 input/output 和维度

假设 batch size 为 1。由于不同实例的 native crop 大小不同，当前训练固定使用 batch size=1，然后梯度累积 8 次。

### 6.1 输入到 U-Net

输入张量：

```text
B × 3 × Z × Y × X
```

其中：

- `B=1`；
- `Z,Y,X` 随实例不同而变化；
- 3 个通道分别是 image、observed、image_valid。

### 6.2 各向异性 3D U-Net

`AnisotropicUNet(channels=24)` 的通道变化为：

```text
输入             : B × 3  × Z × Y × X
ConvBlock a      : B × 24 × Z × Y × X
XY avg_pool      : B × 24 × Z × Y/2 × X/2
ConvBlock b      : B × 48 × Z × Y/2 × X/2
XY avg_pool      : B × 48 × Z × Y/4 × X/4
ConvBlock c      : B × 96 × Z × Y/4 × X/4
上采样 + skip b  : B × 48 × Z × Y/2 × X/2
上采样 + skip a  : B × 24 × Z × Y × X
```

每个 `ConvBlock` 包含：

```text
Conv3d(kernel=3) -> GroupNorm -> GELU
Conv3d(kernel=3) -> GroupNorm -> GELU
```

关键点：pooling kernel 是 `(1,2,2)`，即不在 Z 方向下采样。这样做是为了保留各向异性 Z 信息。

### 6.3 从 U-Net feature map 采样支持点 feature

每个支持点用 `grid_sample` 从 U-Net 输出的 24 通道 feature map 取特征：

```text
U-Net feature map : B × 24 × Z × Y × X
每个支持点 feature: B × N_support × 24
```

这里 `N_support` 随实例变化，不固定为 2048。

### 6.4 FPS 分组和局部支持点特征

从有效支持点中使用 FPS 选择 64 个中心：

```text
center_xyz: B × 64 × 3
```

对于每个中心，选择最多 32 个最近支持点：

```text
groups: B × 64 × 32
```

每个局部邻域中的点拼接：

```text
local physical coordinate : 3
local normalized coordinate: 3
U-Net feature             : 24
总输入                    : 30
```

经过：

```text
30 -> 256 -> 256
```

并在最多 32 个邻域点上进行 masked max pooling，得到每个中心一个 256D token：

```text
local token: B × 64 × 256
```

注意：这里的 max pooling 只是在每个局部邻域内聚合支持点，不是最终的 64 个 token 的全局 pooling。

### 6.5 几何位置和物理尺度编码

对每个中心 token 加上两类信息：

```text
pos(centers): 3 -> 256 -> 256
scale(log_size): 3 -> 256 -> 256
```

其中 `log_size` 是：

```python
log(size * spacing / 1000)
```

单位为微米。最终 Transformer 输入仍为：

```text
B × 64 × 256
```

### 6.6 Transformer Encoder

当前配置：

```text
Transformer blocks: 6
embedding dim      : 256
attention heads    : 8
head dimension     : 32
FFN hidden dim     : 1024
dropout            : 0.1
norm_first         : True
activation         : GELU
```

输入输出形状不变：

```text
B × 64 × 256 -> B × 64 × 256
```

### 6.7 最终 Encoder latent

Transformer 输出先对 64 个 token 做 mean pooling：

```text
h_mean = MeanPool(H_encoder)
形状   = B × 256
```

然后通过 bottleneck：

```text
LayerNorm(256)
Linear 256 -> 256
GELU
Linear 256 -> 256
```

得到：

```text
z: B × 256
```

这个 `z` 是当前模型真正用于两个 Decoder 的共享 latent，也是后续推荐导出的 embedding。当前版本不存在旧实验中的 128D projection head。

### 6.8 Size decoder

尺寸预测头：

```text
z: 256
256 -> 128 -> 3
```

输出三个轴向的 log physical length：

```text
predicted_log_size: B × 3
```

训练目标是 tight bbox 三个轴的物理长度，单位先转换为微米再取 log。

### 6.9 Query voxel decoder

每个 query 输入：

```text
z      : 256
q_t    : 3
q_r    : 3
Fourier(q_t): 36
总输入 : 298
```

`q_t` 使用 6 个频率的 sin/cos Fourier features：

```text
6 frequencies × 3 coordinates × 2 = 36
```

Decoder：

```text
298 -> 256 -> 256 -> 128 -> 2
```

两个输出是：

```text
output[..., 0]: occupancy logit
output[..., 1]: normalized intensity prediction
```

因此，一个实例的整个 reconstruction 不是直接输出固定大小的体素块，而是：

1. Encoder 只运行一次得到 `z`；
2. Decoder 输入任意数量 query 坐标；
3. 分块查询所有目标 voxel，可以恢复 dense occupancy 和 intensity。

---

## 7. Loss 的组成和含义

设：

- `p = sigmoid(occupancy_logit)`；
- `y` 是 occupancy target；
- `h` 是 query 是否位于 hidden block；
- `v` 是 query 是否有真实 raw EM；
- `w = (0.25 + 0.75h) × v`。

因此可见 query 的基础权重为 0.25，隐藏 query 的基础权重为 1.0；无效 raw EM query 的权重为 0。

### 7.1 Occupancy BCE

代码把 foreground 和 background 分开归一化：

```text
fg_loss = weighted BCE 在 foreground 上的平均
bg_loss = weighted BCE 在 background 上的平均
occupancy_loss = 0.5 × (fg_loss + bg_loss)
```

这样避免 query 大多数落在 background 时，模型只输出全背景就获得很低 loss。

### 7.2 Dice loss

使用带平滑项的 soft Dice：

```text
dice_loss = 1 - (2 Σ(w p y) + 1)
                / (Σ(w p) + Σ(w y) + 1)
```

它主要帮助 occupancy 的整体形状和前景覆盖率。

### 7.3 Intensity loss

强度使用 Smooth L1，并分别统计 foreground/background，再平均：

```text
intensity_loss = 0.5 × (foreground intensity loss
                       + background intensity loss)
```

这会让模型学习 normalized EM texture，但它不是在做 raw uint8 的无损恢复。

### 7.4 Size loss

尺寸预测使用 log physical size 的 Smooth L1：

```text
size_loss = SmoothL1(predicted_log_size, target_log_size)
```

### 7.5 总 loss

代码实际实现为：

```text
L_total = 0.5 × occupancy_loss
        + 0.5 × dice_loss
        + 1.0 × intensity_loss
        + 0.2 × size_loss
```

这里需要注意一个容易混淆的地方：`occupancy_loss` 本身已经是 foreground/background 两项的平均，代码又在总 loss 中乘了 0.5。

当前没有：

- VICReg invariance/variance/covariance loss；
- NT-Xent loss；
- DEC/KL clustering loss；
- class label loss；
- 直接的 morphology regression loss。

---

## 8. 正式训练配置

实际运行目录：

```text
/path/to/project_data/EM/point-MAE/runs/voxel_mae_native_v1_20261014/
```

核心配置：

| 参数 | 实际值 |
|---|---:|
| epochs | 10 |
| queries / instance | 8192 |
| online views | 4 |
| batch size | 1 |
| gradient accumulation | 8 |
| workers | 4 |
| learning rate | 1e-4 |
| optimizer | AdamW |
| weight decay | 0.05 |
| AMP | BF16 |
| Transformer dim | 256 |
| Transformer depth | 6 |
| Transformer heads | 8 |
| latent dim | 256 |
| U-Net base channels | 24 |
| 参数量 | 6,083,325 |
| seed | 42 |
| GPU | 物理 GPU 6，A100-SXM4 80GB |

学习率使用 warmup + cosine 形式：

```text
warmup = min(500, max(32, len(train_loader)//2))
```

本次训练的 `len(train_loader)` 为 28,504，因此 warmup 为 500 个 optimizer/global step。随后使用 cosine 形式逐渐降低学习率，但保留最低约 0.1 倍的比例。

由于每个 native crop 大小不同，代码不允许在一个 batch 中拼接不同尺寸的样本，所以使用：

```text
batch=1 + gradient accumulation=8
```

这不是传统意义上的 batch size 8；8 个样本只是连续累积梯度后做一次 optimizer step。

### 训练轮次调整的历史

最初的启动计划是 100 epochs。训练到第 6 轮后，用户要求缩短为总共 10 轮。`shorten_training_to_10.py` 等待第 6 轮的完整 checkpoint，然后终止旧进程，再从 `last.pt` 恢复到第 10 轮。

所以当前 10 轮结果不是“从第一轮开始就把配置设成 10 epochs”得到的完全独立运行，而是：

```text
前 6 轮：原计划 100 轮的运行
后 4 轮：恢复后完成总计 10 轮
```

这一点在比较不同训练计划时需要保留。

总训练与验证日志耗时约 10.52 小时。日志中 PyTorch 观测到的峰值显存约 3.55 GB，这只是进程峰值分配，不等于 GPU 总显存使用量。

---

## 9. 正式训练结果

以下结果来自整轮 validation 的平均，不是某一个 batch 的数值。验证集包含全部 847 个实例，使用固定 masking 和固定 query 随机位置。

| epoch | train loss | val loss | val occupancy IoU | val intensity loss | val size relative error |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.1848 | 1.1716 | 0.4542 | 0.4009 | 8.43% |
| 2 | 1.1119 | 1.0993 | 0.4938 | 0.3977 | 7.55% |
| 3 | 1.0392 | 1.0360 | 0.5269 | 0.3915 | 7.63% |
| 4 | 0.9559 | 0.9402 | 0.5763 | 0.3853 | 7.44% |
| 5 | 0.8866 | 0.8899 | 0.6089 | 0.3805 | 6.19% |
| 6 | 0.8535 | 0.8699 | 0.6191 | 0.3786 | 6.28% |
| 7 | 0.8150 | 0.8348 | 0.6434 | 0.3749 | 4.76% |
| 8 | 0.8017 | 0.8279 | 0.6456 | 0.3744 | 4.52% |
| 9 | 0.7933 | 0.8217 | 0.6486 | 0.3741 | 4.30% |
| 10 | 0.7885 | 0.8175 | 0.6539 | 0.3735 | 4.28% |

整体变化：

- validation total loss：`1.1716 -> 0.8175`，下降约 30.2%；
- validation occupancy IoU：`0.4542 -> 0.6539`；
- physical size relative error：`8.43% -> 4.28%`；
- intensity loss：`0.4009 -> 0.3735`。

这些结果说明模型已经能够学习一定的外形、灰度和物理尺寸信息。训练和验证曲线没有明显分离，暂时没有明显的训练集过拟合迹象。

但是，当前结果还不能直接说明：

- 256D embedding 是否适合聚类；
- embedding 是否能区分不同形态的线粒体；
- embedding 是否能预测细胞类型或突触环境；
- 训练出来的 representation 是否优于旧 Point-MAE 的 128D representation。

### 重要指标解释

当前 IoU 是在随机 query 位置上计算的采样 IoU，不是对全部原始 voxel 中心做密集预测后计算的 IoU。

当前 intensity loss 是 normalized intensity 的 Smooth L1，不是原始 EM 灰度的百分比误差。

size relative error 是 log-size prediction 转回物理长度之后的平均相对误差。

---

## 10. Checkpoint 和已有产物

运行目录：

```text
/path/to/project_data/EM/point-MAE/runs/voxel_mae_native_v1_20261014/
```

文件含义：

| 文件 | 含义 |
|---|---|
| `best.pt` | 按整轮 validation total loss 选择的最佳 checkpoint；当前是第 10 轮 |
| `last.pt` | 第 10 轮结束的模型和 optimizer 状态 |
| `recovery.pt` | 每 500 个 global step 保存的中途恢复点，可能早于最终模型 |
| `config.json` | 本次运行参数、模型配置和参数量 |
| `metrics.jsonl` | 每轮 train/validation 汇总指标 |
| `steps.jsonl` | 训练过程中的 step 级日志 |
| `mito_training_metrics_20261014.csv` | 便于表格分析的逐轮结果 |
| `mito_training_curves_20261014.png` | 训练/验证曲线 |
| `mito_training_report_20261014.md` | 原始 10 轮训练报告，本总文档已整合其内容 |
| `completed.json` | 训练完成状态和 best validation loss |
| `train.log` | 原始训练 stdout/stderr |

推荐后续默认使用：

```text
best.pt
```

因为它是按 validation total loss 选出的；如果需要恢复训练，则使用 `last.pt`，因为它包含 optimizer 状态。

---

## 11. Sanity check 结果

`checks.py` 已经完成以下检查，结果均为 true：

| 检查 | 目的 |
|---|---|
| `coordinate_ramp` | 确认 ZYX -> XYZ 的 `grid_sample` 坐标转换正确 |
| `neighbor_deduplication` | 确认支持点没有重复 voxel 坐标 |
| `augment_alignment` | 确认 EM、mask、valid 增强后仍然对齐 |
| `input_masking` | 确认隐藏 block 的输入 image 被置零 |
| `all_parameter_gradients` | 确认所有模型参数有有限梯度 |
| `latent_and_scale_gradients` | 确认 latent bottleneck 和 size head 都参与反向传播 |
| `padding_invariance` | 确认无效 padding 支持点不会改变 embedding |
| `query_chunk_equivalence` | 确认一次 query 和分块 query 的 Decoder 输出一致 |
| `scale_sensitivity` | 确认模型确实使用了物理尺度输入 |
| `z1_unet` | 确认 Z=1 的薄 crop 不会在 U-Net 中失败 |

### 单实例 fixed-view overfit

在一个真实实例上关闭增强，用较小模型做 100 steps 的固定视图拟合：

```text
loss:                1.55898 -> 0.29537
occupancy IoU:       0.62891 -> 0.96841
size relative error: 2.7418   -> 0.01232
```

这说明：

- 数据读取和坐标链路可以学习；
- occupancy、intensity、size 的梯度可以进入模型；
- Decoder 不是完全无法拟合目标。

但这不是泛化结果，也不能当作 validation/test 性能。

---

## 12. 推理和导出流程

### 12.1 导出每个实例的 256D embedding

使用：

```bash
cd /path/to/JaneliaEM/point-MAE

python -m mito_voxel_mae.export \
  --cache /path/to/project_data/EM/mito_voxel_mae_training_v1 \
  --checkpoint /path/to/project_data/EM/point-MAE/runs/voxel_mae_native_v1_20261014/best.pt \
  --out /path/to/export \
  --embeddings-only
```

这个模式会：

1. 加载 `best.pt`；
2. 构建 `MitoDataset(split='all', masking=False)`；
3. 对每个实例使用完整未遮挡输入；
4. 调用 `model.encode()`；
5. 保存 `embeddings.npz`。

输出包括：

```text
embedding: N × 256
record_id: N
body_id: N
```

用于 UMAP、PCA、线性 probe 或 clustering 的默认向量应该是这里的 `embedding`，也就是 Encoder 的 256D latent，而不是任何 reconstruction output。

### 12.2 单实例 dense reconstruction

可以使用：

```bash
python -m mito_voxel_mae.export \
  --cache /path/to/project_data/EM/mito_voxel_mae_training_v1 \
  --checkpoint /path/to/project_data/EM/point-MAE/runs/voxel_mae_native_v1_20261014/best.pt \
  --out /path/to/dense_export \
  --record-id RECORD_ID \
  --chunk 8192
```

流程是：

1. 使用完整未遮挡输入得到一次 `z`；
2. 枚举实例 tight bbox 内所有 voxel center；
3. 分块调用 query Decoder；
4. 保存：
   - `occupancy_probability.npy`
   - `intensity_normalized.npy`
   - `embedding.npy`
   - `metadata.json`

当前正式模型尚未对全部 test 集做 dense export 和 dense voxel metric 汇总。已有的 dense export 主要来自单实例 overfit check，因此不能当作正式模型的泛化重建结果。

---

## 13. 当前模型已经验证了什么

目前可以确认：

1. 原始 mask、foreground EM 与 cache 一致；
2. raw context 的 valid mask 能区分真实 raw 和缺失 raw；
3. ZYX/XYZ 坐标转换通过了 analytic ramp test；
4. anisotropic U-Net 可以处理 Z=1 的薄 crop；
5. hidden block 会在 U-Net 前被遮挡；
6. Encoder、bottleneck、size head、voxel decoder 都参与反向传播；
7. padding、query chunking 不会改变输出；
8. latent 对物理尺度变化敏感；
9. 正式 validation loss、occupancy IoU、size error 都随训练改善；
10. 单实例 fixed-view 可以被明显拟合。

---

## 14. 当前还不能下的结论

以下结论目前还没有实验证据：

### 14.1 不能直接说 256D embedding 已经适合聚类

当前训练只优化 reconstruction 和 size loss，没有直接优化 cluster separation。必须先导出所有实例的无掩码 256D embedding，再进行：

- PCA/UMAP；
- effective rank；
- pairwise cosine/Euclidean distance；
- same-instance augmentation stability；
- morphology feature linear probe；
- neuron-level split 下的聚类/分类评价。

### 14.2 validation IoU 不是 dense IoU

它基于 8192 个随机 query。需要在测试集上对全部 tight bbox voxel 做密集分块推理后，重新计算：

- dense occupancy IoU；
- Dice；
- foreground precision/recall；
- normalized intensity MAE；
- size error。

### 14.3 当前未覆盖被排除的大实例

1,024 个 context 超过 1.5M voxels 的实例不在当前模型训练 cohort 中。因此当前结论不适用于最大线粒体。未来需要 tiled native-resolution context 或更高效的稀疏表示。

### 14.4 当前增强还不是严格的全 3D 旋转

当前主要是物理 spacing 感知的轻微仿射、XY 旋转/平移和轻微非线性形变。它不是任意 3D SO(3) 旋转，也没有证明强旋转不变性。

### 14.5 occupancy 不代表 cristae membrane label

`mask.npy` 是线粒体外轮廓 occupancy。内部嵴结构只可能通过 EM intensity texture 间接体现，并没有单独的 cristae occupancy target。

---

## 15. 建议的下一步验证顺序

建议按以下顺序推进，先确认模型是否真的学到了可用表征，再考虑加复杂目标：

### 第一步：正式 test dense reconstruction

固定 `best.pt`，对 849 个 test 实例进行 dense query，报告：

- occupancy IoU/Dice；
- 前景和背景的 precision/recall；
- intensity MAE/PSNR；
- 三轴 physical size relative error；
- 按实例大小分层的结果。

### 第二步：导出无掩码 256D embedding

对 train/val/test 分别导出，且严格使用 body-level split。检查：

- embedding norm；
- 每个维度方差；
- PCA explained variance；
- effective rank；
- UMAP/PCA；
- 同一实例不同 view 的 cosine similarity。

### 第三步：检查表征是否包含基础形态

从 mask 和 metadata 计算：

- log volume；
- physical bbox length；
- elongation；
- surface area proxy；
- sphericity；
- intensity contrast；
- occupancy fraction。

然后做 neuron-level train/test split 的 frozen linear probe，报告 R²，而不是只看 UMAP 图。

### 第四步：与旧 Point-MAE 128D 结果严格对比

必须固定：

- 相同实例集合；
- 相同 body-level split；
- 相同 metadata；
- 相同 UMAP/PCA 参数；
- 相同稳定性和 probe 指标。

比较对象应明确为：

```text
旧 Point-MAE 的最终可用表征
vs.
当前 Voxel MAE 的 256D Encoder latent
```

### 第五步：再决定是否加入 VICReg 或 clustering objective

只有当 reconstruction、embedding 稳定性和 morphology probe 都正常后，才适合在此基础上加入 VICReg、NT-Xent 或 DEC。否则会把输入、Encoder、pooling 和 clustering loss 的问题混在一起。

---

## 16. 一句话总结

当前模型的实际逻辑是：

```text
原分辨率 EM + 外轮廓 mask + valid mask
    -> 轻微几何增强
    -> 4×4×4 空间遮挡
    -> 各向异性 3D U-Net
    -> 规则网格/halo 支持点 feature
    -> 64 个局部 token
    -> 6 层 256D Transformer
    -> mean pooling + bottleneck
    -> 256D Encoder latent z
    -> occupancy/intensity query reconstruction + physical size prediction
```

本次 10 轮训练已经证明这条链路可以稳定学习重建任务；但是否形成了具有生物学区分度的线粒体 representation，还需要 dense test、embedding stability、形态学 probe 和与旧 Point-MAE 的公平对照来确认。

