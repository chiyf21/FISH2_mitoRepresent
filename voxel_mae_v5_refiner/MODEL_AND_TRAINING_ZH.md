# Mitochondria Voxel MAE v5：模型与训练复现说明

文档日期：2026-10-15  
代码目录：/path/to/JaneliaEM/point-MAE/mito_voxel_mae_v5/  
配套运行目录：/path/to/project_data/EM/point-MAE/runs/voxel_mae_v5_strict333_3x2_20261015/  
数据缓存：/path/to/project_data/EM/mito_voxel_mae_prepared_v3_square_20261015

本文记录 v5 当前代码实现的算法和双目标训练协议，目标是让另一位工程人员结合本文、代码目录和缓存复现过程。它描述的是实现事实，不把尚未完成的训练称作最终验证权重。动态训练状态以运行目录内实际日志和 checkpoint 为准。

## 1. 一句话定义

v5 是一个原分辨率、物理坐标感知的线粒体 masked reconstruction 预训练模型。图像分支从被遮挡的 EM 体数据提取特征；点分支在实例周围构造稀疏支持点，将图像特征聚合成 64 个局部 token；Transformer 将其压缩成 256 维实例向量。decoder 只对“整个 3×3×3 原始体素块都未被输入看见”的 query 预测目标实例占用与灰度。

它不是标准 Point-MAE 论文网络的逐字复现，也不是 Refiner、VICReg 或聚类模型。它沿用本项目图像—点组—Transformer 主干，并用两种可切换的 MAE 灰度目标做对照。

## 2. 版本沿革：v5 与前几版的关系

| 版本 | 主要内容 | 与 v5 的关系 |
|---|---|---|
| v1 | 原分辨率体数据、遮挡、U-Net、稀疏点分组、Transformer 和 occupancy/intensity/size 重建 | 最早的 voxel-MAE 路线；不是更早的 128D Point-MAE projection 路线 |
| v2 | 扩大至完整 19,947 个实例；padding-aware 处理、分桶 batch、大实例 tiled 特征提取；约 700/300 前景/背景 query | v5 复用其 FPS 分组、Transformer/bottleneck、尺寸 head 和 relative 17-grid 支持点构造 |
| v3 | 改为按切片批处理的 2D U-Net，并追加不下采样 Z 的残差融合 | 提供 v4/v5 使用的 SliceUNet 与 ZResidual |
| v4 | 原生分辨率 tile：XY 512 tile、256 core、128 halo，Z 16 core、3 halo；固定物理网格 query；保留尺寸 scale 输入 | v5 继承图像 tile 路径，但修改支持点分布、移除直接尺寸输入，并重做 query 和 decoder 目标 |
| v5 | 3×3×3 整块严格隐藏 query；27 voxel occupancy/intensity 或 3D 低频 DCT；query 不重复 | 本文版本 |

从 v4 到 v5，图像主干和 Transformer 的总体结构没有推倒重来，但算法也不只是换 loss。变化包括：

1. Encoder 支持点从 v4 固定物理间距网格改成以 tight bbox 为参照的 17³ 相对网格及其 26 邻接支持点。输入网络的坐标仍是实例中心为原点的真实微米坐标。
2. 移除每个 token 上直接相加的 bbox 尺寸 scale embedding；物理尺寸只作为辅助回归目标。
3. decoder 从预测单个查询点改成预测 query 周围的 27 个原始体素。
4. query 必须满足 27 个体素全部有效、全部隐藏；不再进行 700/300 前景/背景配额，也不通过重复采样补足数量。
5. 灰度目标增加完整小块和物理低频小块两种对照。

## 3. 数据来源、轴顺序与缓存接口

### 3.1 当前缓存

v5 不负责从原始 EM 制作缓存；它读取已准备好的 v3 square cache。当前缓存统计：

- 19,947 个线粒体实例、184 个神经元 body。
- train 15,508、validation 1,953、test 2,486 个实例。
- body 级划分互斥：train/val/test 分别覆盖 148/18/18 个 body。
- 原始数组轴顺序 ZYX，体素间距为 [30, 8, 8] nm；等价 XYZ 为 [8, 8, 30] nm。
- 每个实例目录含 em.npy、mask.npy、valid.npy。数组形状均为上下文体积 [Z,Y,X]。mask 是当前目标线粒体的二值 mask；valid 表示该位置是否有可用原始 EM。
- manifest.json 的 record 至少含 record_id、body_id、relative_dir、shape_zyx、tight_lo_zyx、tight_shape_zyx、spacing_zyx_nm、强度截断界限 lo/hi、bbox_xyz 和 split。
- audit.json 记录缓存与原始来源的一致性。本轮缓存审核记录为 19,947 条、0 errors，前景 EM 与 mask 和来源一致。

缓存中的方形 XY 是为方便组织上下文而准备的，**不代表把每个线粒体缩放到相同尺寸**。训练没有把整条线粒体 resize 成统一的 Z×Y×X。

### 3.2 三种 mask/valid 概念

- mask.npy：监督使用的目标 instance mask。
- valid.npy：原始 EM 是否存在；缺失上下文不参与 query 或有效像素统计。
- observed：当前随机遮挡后，哪些位置允许 encoder 看见；它是每个 view 的动态输入遮挡图，不等同于 valid。

目标实例的可见 mask 输入为 mask 与 observed 的逻辑交集。隐藏位置的目标 mask 不会送入 encoder。其他结构的 EM 强度仍可能出现在可见图像中，但其他结构不属于当前目标实例，occupancy 标签记为 0。

### 3.3 可选的缓存复核

v5 启动器要求 cache 中存在 audit.json，且实例数为 19,947、errors=0、一致性标志为 true。若要针对自己的原始数据重新生成审计结果，v2 audit 脚本需要一个包含 metadata/instances.csv 和 CSV 所引用原始 mask/EM 文件的 source root：

    cd /path/to/JaneliaEM/point-MAE
    PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
    CACHE=/path/to/project_data/EM/mito_voxel_mae_prepared_v3_square_20261015
    ORIGINAL_DATA_ROOT=<包含 metadata/instances.csv 的原始数据根目录>
    PYTHONPATH=$PWD "$PY" -m mito_voxel_mae_v2.audit --cache "$CACHE" --original "$ORIGINAL_DATA_ROOT" --workers 4

不要在没有确认 source root 的情况下猜测或替换原始数据路径。v5 训练只消费现有缓存，不自动构建它。

## 4. 单个训练样本/view 的构造

入口是 mito_voxel_mae_v5/data.py 中的 MitoDataset。每个训练实例每 epoch 产生 4 个在线 view。第 index 个样本使用的 numpy 随机种子为：

    seed_view = seed + index * 100003 + epoch * 10000019

训练循环会更新 epoch；validation 固定 epoch 0、1 view、无几何增强，因此每轮 validation 使用固定遮挡和 query 随机流。训练 view 编号由 index 除以实例数的整数商决定，对应 4 种增强类型。

### 4.1 灰度预处理与增强顺序

单个实例按以下次序处理：

1. 以 manifest 中该实例自己的 lo/hi 对 raw EM 做线性截断映射：clip((raw-lo)/max(hi-lo,1e-6), 0, 1)。
2. 对 EM、mask、valid 同步施加几何变换。强度用线性插值，mask 用最近邻插值；valid 变换后以插值结果 >0.999 判为有效。
3. 如果变换后 mask 为空、碰到 crop 边界，或 mask 前景落入 invalid 区域，则拒绝这次变换，退回原始 EM/mask/valid。
4. 重新计算变换后 mask 的 tight bbox。
5. 在 tight bbox 内选择 valid 且当前可见的灰度像素，计算均值和标准差；标准差下限 1e-6。用同一组统计量 z-score 变换可见输入与干净 query 灰度 target。
6. 仅输入灰度会添加 Gaussian noise，sigma 从 [0,0.035] 均匀采样；target 不加噪声。最后将所有不可见或 invalid 的输入灰度设为 0。

四种几何 view 的实现位于 mito_voxel_mae_v4/native_data.py 的 augment 函数：

- kind 0：不作几何变换。
- kind 1：小幅物理仿射，约 ±3° 旋转、每轴 0.97–1.03 缩放、小剪切和小位移。
- kind 2：XY 平面较大旋转，约 ±10°，带小位移。
- kind 3：小仿射/位移加平滑弹性形变。
- 位移量级约为各数组轴尺寸的 1.5%；弹性位移由 4³ 粗网格平滑插值得到，量级约为对应轴尺寸的 0.8%。

这些增强保持原体数据的各向异性物理间距，不是任意交换 Z/Y/X 轴。

### 4.2 输入空间遮挡

在变换后的 tight bbox 中均匀划分 4×4×4 共 64 个空间块，随机隐藏 38 个，保留 26 个可见块。名义遮挡率 38/64 ≈ 59.4%，不是 Point-MAE 论文中的 80%。

每个支持点按相对 tight bbox 坐标映射到 4³ block。若所有支持点都因 block 隐藏或 raw invalid 而不可见，代码会随机选一个有效支持点并重新打开其所在 block；这是薄实例的防空输入回退，可能使实际隐藏块数少于 38。随后 observed = visible_block_map 与 valid 的逻辑交集。

输入图像分支的四个通道依次为：

1. masked、z-scored EM intensity；
2. 仅可见部分的目标 instance mask；
3. observed 可见性；
4. raw valid 图。

valid 表示原始数据可用性，不携带目标实例标签。隐藏区域灰度与隐藏 mask 都不进入 encoder。

## 5. Encoder 支持点与坐标

### 5.1 Relative 17³ grid + 一圈邻居

v5 调用 mito_voxel_mae_v2/data.py 中的 point_support(mask, lo, size, spacing, grid=17)：

1. tight bbox 每轴放置 17 个相对位置；实现额外构造索引 -1 与 17 的一圈 halo 候选，因此初始候选格是 19×19×19。
2. 中间 17³ 位置中落在目标 mask 内的格点作为 seed；halo 本身不能成为 seed。
3. 对 seed 做 3×3×3、全 1 结构元二值膨胀，保留 seed 周围的 26 邻居，并裁掉真实体积边界之外的点。
4. 如果极薄实例一个 seed 都没有，则从真实前景中至多取 64 个均匀索引位置作为 seed，并加入按 bbox 尺度计算的 3×3×3 邻居，再去重和裁界。此时是 fallback，分布与常规规则网格略有不同。

所以 19³ 是候选位置数，不是送入网络的固定点数。保留的支持点是目标内部规则 seed 及其局部邻居；之后还要筛选可见且 raw-valid 的支持点。点数随实例形状、遮挡和有效数据变化。

### 5.2 坐标、可见性与图像特征采样

数组与 spacing 均以 ZYX 顺序保存。tight bbox 左下角为 lo、形状为 size、几何中心为 centre = lo + size/2。voxel 坐标 vox 对应的物理坐标为：

    points_um = (vox + 0.5 - centre) * spacing_zyx_nm / 1000

即每个实例以中心为坐标原点，单位微米；保留 Z 与 XY 的真实物理比例。候选点有效性为：候选点所属遮挡块可见，且该点对应 raw voxel 有效。

Dataset 仍打包 points_t、points_r 等兼容/诊断字段，但 v5 的 encode() 实际使用 points_um、points_valid、points_vox。v5 不把归一化 bbox 坐标或 bbox 尺寸显式输入 encoder。

## 6. Decoder query：严格隐藏的 3×3×3 target

### 6.1 候选中心格点

query 中心在物理间距 XYZ=32×32×30 nm 的格点上取样。按数组 ZYX 表示，步幅是 (1,4,4)：每个 Z 层取候选中心，Y/X 每 4 个原始 voxel 取一个。格点 phase 由记录的全局 bbox 起点和 tight bbox 起点确定，避免不同 crop 的格点任意漂移。

query 中心必须离数组边界至少一个 voxel，以保证周围 27 个位置都在体积内。一个 query 的 offsets 为：

    (dz,dy,dx)，其中 dz、dy、dx 各自属于 {-1,0,1}

### 6.2 一个 query 被接纳的全部条件

mito_voxel_mae_v5/targets.py 中的 safe_query_centres() 仅在以下条件同时满足时保留中心：

1. 中心属于上述固定物理网格。
2. 周围 27 个 raw voxel 全部 valid=True。
3. 这 27 个 voxel 映射到的 4³ 遮挡块全部不可见；其中任何一个 voxel 落在可见块，该 query 都拒绝。
4. 27 个 voxel 中至少 2 个属于当前目标线粒体 mask。

其他 voxel 可以是当前 instance 外部、背景或另一个 instance；它们对当前目标 occupancy 的标签均为 0。query 最多取 1000 个；候选不足时取全部，不放回采样；没有前景/背景配额，也不重复补足。候选数为 0 时跳过本实例/view 并记入 skipped log，不会删除原始 cache record。

一个 query patch 覆盖 3×3×3 个原始 voxel，物理尺寸为 XYZ=24×24×90 nm（ZYX=90×24×24 nm）。query patch 之间允许重叠。

### 6.3 query label 与防泄漏

每个 query 产生：

- occupancy[q,27]：27 个 voxel 是否属于当前目标 instance。
- intensity[q,27]：几何增强后的干净灰度，使用仅由可见图像得到的同一 mu/std 标准化。
- query_um[q,3]：中心坐标相对于实例中心的物理微米坐标。
- query_valid[q]：真实 query 为 true；batch padding 位置为 false。

每个目标 patch 生成后会再次断言 27 个 voxel 全未被 observed 覆盖。正确性测试还会改变所有隐藏 EM 灰度，验证 encoder 输入不变而 target 灰度改变。因此隐藏目标只用于 loss，不通过图像或归一化统计泄漏到 encoder。

## 7. 模型结构与张量形状

生产配置默认为 channels=24、groups=64、k=32、dim=256、depth=6、heads=8、latent=256。A 参数量 5,656,313；B（含 occupancy）5,651,944。实现分散在 v5/model.py、v4/model.py、v3/model.py 与 v2/model.py。

### 7.1 原分辨率 tile → 2D U-Net → Z 融合

每个支持点依据 voxel 坐标被分配到 ZYX core [16,256,256]。生成图像 tile 时每侧加 Z halo=3、XY halo=128，通常最大 tile 为 [22,512,512]；volume 边界处截取后补零到 tile 尺寸。不 resize 原图。

多个 tile 的切片一起送进 SliceUNet，输入形状 [切片数,4,512,512]，输出 [切片数,24,64,64]（XY stride 8）。SliceUNet 每级先做 2×2 平均池化，用 2D 卷积块编码，并在 1/8 分辨率 skip 上拼接更深层特征。之后恢复每个 tile 的 Z 顺序，经过 3 个 ZResidual 块：

- XY 卷积核 (1,3,3)；
- Z 卷积核 (3,1,1)；
- 1×1×1 通道混合和残差；
- 不沿 Z 下采样；LayerNorm 只在通道维归一化，避免 padding 改变空间统计。

最后用三维 grid_sample 在每个 points_vox 位置读取 24D 图像特征。特征图 XY stride8 对应原图约 64 nm；支持/query 网格按 32 nm 取样并通过插值读取，不意味着 U-Net 保留了原始 32 nm 的 XY 特征分辨率。

tile core/halo、SliceUNet 和 ZResidual 的实现入口分别见 v4/model.py 与 v3/model.py。训练中 tile feature extraction 使用 non-reentrant activation checkpoint，降低保存激活的显存开销。

### 7.2 点局部组与 Transformer

对每条实例的有效物理支持点：

1. 在 points_um 上选 64 个 FPS 中心。FPS 每次选距已选集合最远的有效点；候选少于 64 时中心允许重复，padding 点不参与。
2. 对每个中心按物理欧氏距离稳定排序，选最多 32 个有效最近邻。
3. 对邻居拼接 3D 相对物理坐标 neighbor_um-center_um 与 24D U-Net feature，共 27D。
4. 经共享 MLP 27→256→256，对 32 个邻居做 masked max pooling，成为一个 256D 局部 token。
5. 将中心物理位置（相对于实例中心，单位 µm）经 3→256→256 位置 MLP 后加到 token。

结果为 [B,64,256]。Transformer 为 6 层、8 heads、256 hidden、1024 feed-forward、dropout 0.1、GELU、pre-norm。沿 token 维 mean pooling 得 [B,256]，经 LayerNorm→Linear(256,256)→GELU→Linear(256,256) bottleneck，得到实例向量 z。

最终表征是 encoder 路径输出的 256D z，不是一个未参与训练的 projection head 输出。v5 没有 128D projection head。

### 7.3 尺寸 head 与 query decoder

尺寸 head 为 256→128→3，回归三个轴的 log physical bbox 尺寸，目标为：

    size_target = log(size_zyx * spacing_zyx_nm / 1000)

它只作为辅助监督，尺寸数值不显式拼进 encoder token。

query decoder 对每个 query 使用：

- 全局实例向量 z：256D；
- query 中心物理坐标 q：3D，单位 µm；
- Fourier 特征：q 除以 6 个波长 [0.024,0.048,0.096,0.192,0.384,0.768] µm 后取 sin/cos，共 36D。

拼接维度为 295，送入 MLP 295→256→256→output_dim。每个 query 独立预测。

## 8. 两种预训练目标与精确 loss

张量中 B=batch size，Q=batch 内 query padding 后最大数量。occupancy/intensity 目标形状 [B,Q,27]；query padding 的 query_valid=0，不参与统计。loss 先在每个 instance 内对有效 query 求平均，再按 sample_weight 对 instance 加权平均。

定义 p=sigmoid(occupancy_logits)，y 为二值 occupancy：

- 平衡 BCE：实例内部前景 voxel BCE 均值和背景 voxel BCE 均值各占 0.5。
- Dice loss：1-(2*sum(p*y)+1)/(sum(p)+sum(y)+1)，只对有效 query 的 27 个 voxel 汇总。
- size loss：三轴 log size 上的 SmoothL1 均值。
- 尺寸项权重 0.2；Dice 项权重 0.5。

### A：27 voxel occupancy + 27 voxel intensity

最后一层输出 54，reshape 为 [B,Q,27,2]，拆为 occupancy logits 与灰度预测。灰度 SmoothL1 在每个实例内部按 GT 前景/背景分别平均，再按前景 0.7、背景 0.3 加权。

    L_A = balanced_BCE + 0.5 * Dice + intensity_0.7fg_0.3bg + 0.2 * size

A 的低频指标由预测的 27 灰度再做 DCT 得到，仅是诊断指标，不加入 A 的总 loss。

### B：10 个物理低频 DCT 系数 + 27 voxel occupancy

最后一层输出 37（10 个系数 + 27 个 occupancy logits）；生产配置必须启用 --lf-with-occ。在 3×3×3 block 上做正交 3D DCT-II。每个轴 n=3，物理 spacing 使用 ZYX [30,8,8] nm。候选 mode k 的频率为：

    frequency(k) = sqrt(sum_axis((k_axis / (2 * 3 * spacing_axis))^2))

按物理频率由低到高选 10 个 mode，而不是只按 mode index 排序。basis 实现在 mito_voxel_mae_v5/targets.py 的 low_frequency_basis()。

目标系数为 target27 @ basis.T / sqrt(27)；预测系数经 pred10 @ basis * sqrt(27) 变回低通图像。B 灰度 SmoothL1 比较的是两幅 27-voxel 低通图像，而不是 10 个系数的简单均值。B 不对频域像素套用前景/背景配额，因为低频基函数覆盖整个 patch。

    L_B = lowpass_image_SmoothL1 + balanced_BCE + 0.5 * Dice + 0.2 * size

B 只监督低频灰度，不能用它单独的 full-intensity MAE 宣称细节恢复与 A 等价。两实验可共同比较 occupancy、低频误差、尺寸误差和后续 embedding 指标。

### 8.1 训练与 validation 指标

训练/validation 记录 loss、lowfreq、size、size_relative_error、intensity_mae、dc_mae、ac_mae；A 另记录 intensity loss，A/B（生产 B 启用 occupancy）都记录 occupancy BCE、Dice loss、IoU。size_relative_error 由预测与目标 log-size 差取指数并计算相对误差。

validation IoU 是安全 query patch 指标。相邻 patch 可能重叠，同一个原始 voxel 因而可能重复计数；它不是完整 volume 上的无偏 dense IoU。

## 9. Batch、DDP 与训练更新

### 9.1 不同实例尺寸如何组成 batch

图像 volume 不在 collate 阶段强行 pad 成相同尺寸；每条实例的四个 volume channel 保持为 list。支持点和 query pad 到本 batch 的最大长度，并以 points_valid、query_valid 标记真实位置。

BucketBatches 先按 Z 长度和 XY tile 工作量分桶，再决定每 rank 的 batch 数量。当前压力校准参数为：

- tile_batch=24；
- slice_microbatch=528，即每个 Z core 16 层加两侧 halo 后约 22 层，24 tile 对应至多 528 切片；
- 每 rank 最大 instance batch=128；
- workload budget=2048；
- 每进程显存分配上限 0.72。

batch 会按实例体积/workload 缩小；这些是当前机器和代表性实例的校准值，不保证适用于任意 GPU 或极端形状。

### 9.2 多卡与梯度累积

A 使用 GPU 0/1/2 三卡 DDP；B 使用 GPU 3/4/5 三卡 DDP。每组 world size=3。当前参数：

- epochs=15，queries=1000，train views=4，seed=42；
  即训练集每 epoch 最多产生 15,508×4=62,032 个 view，实际有效数还要扣除没有安全 query 的跳过项。
- AdamW，初始 learning rate=1e-4，weight decay=0.05；
- AMP bfloat16，TF32 开启；
- 每 rank 2 个 DataLoader worker，spawn context，prefetch factor=1；
- 全局 effective batch=512，跨多个动态 microbatch 累积梯度；
- 梯度按累计有效样本数归一化，global norm clip=1，再执行 optimizer step；
- NCCL process group timeout=60 分钟。

学习率每个训练 batch 更新：warmup steps 为 min(200, max(20, steps_per_epoch//4))；之后按全训练进度 cosine 衰减，最低系数为 0.1。代码等价表达为：

    progress = (epoch * steps_per_epoch + step) / (steps_per_epoch * epochs)
    warm = min(1, (global_step + 1) / warmup_steps)
    lr = base_lr * warm * (0.1 + 0.45 * (1 + cos(pi * progress)))

各 rank 的空 batch 使用 sample_weight=0 的 dummy，保持 DDP collective/梯度图连通但不贡献 loss。为凑齐 rank step 数而重复的尾部样本也设置 weight=0。

### 9.3 初始化、checkpoint 和恢复

全新运行默认随机初始化模型。训练器没有 v4 checkpoint 的 init_from 加载参数；只有显式提供 --resume 才恢复 model 与 optimizer。因此当前 v5 训练不是对某个已训 v4 checkpoint 微调。

每个 epoch 后 rank 0 保存 last.pt；validation loss 改善时保存 best.pt；每 5 epoch 保存 epoch_05.pt、epoch_10.pt、epoch_15.pt。训练中每累计 250 个 batch 且梯度累积窗口清空时，会写 recovery.pt。checkpoint 包括 model、optimizer、epoch、global_step、best、config 和 model_config。

恢复时载入模型/optimizer，并按 runner 逻辑从记录 epoch 的下一个 epoch 开始；OOM retry 可从最近 last.pt 或 recovery.pt 接续。checkpoint 没有记录完整 Python/numpy/CUDA RNG state，因此不能承诺 bitwise-exact resume；数据增强随机流由 seed/index/epoch 公式生成，dropout 等随机序列恢复后仍可能不同。

## 10. 复现命令

### 10.1 先运行实现检查

checks.py 包含 synthetic correctness assertions；加 --cache 后还会取少量真实缓存实例，对 A/B 分别做 forward、backward 和一次 optimizer step。它需要可用 CUDA，建议先在独立空闲 GPU 上运行，避免和正式 DDP 抢显存。

    cd /path/to/JaneliaEM/point-MAE
    PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
    CACHE=/path/to/project_data/EM/mito_voxel_mae_prepared_v3_square_20261015
    CHECKS=/path/to/project_data/EM/point-MAE/runs/v5_preflight_YYYYMMDD/checks
    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=<单张空闲GPU编号> "$PY" -m mito_voxel_mae_v5.checks --cache "$CACHE" --out "$CHECKS"

检查覆盖：安全 query 快速算法与暴力枚举相同；27 voxel 全 hidden/valid 且至少 2 个前景；无重复 query；相对 17-grid；无安全 query 时跳过；隐藏灰度不泄漏到输入统计；物理坐标单位；DCT 正交、DC 与物理频率顺序；batch 与单实例 latent 一致；target 不影响 encoder；A/B 和 DDP dummy 的梯度有限且连通。

### 10.2 推荐的成对正式运行入口

launch_two.py 会自动执行三卡压力探测、生成本次代码 snapshot、启动 A/B 两个三卡实验，并在 OOM 时逐步缩小 workload 参数。必须使用一个全新的 run root：启动器会拒绝覆盖已存在的 code_snapshot/。

    cd /path/to/JaneliaEM/point-MAE
    PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
    CACHE=/path/to/project_data/EM/mito_voxel_mae_prepared_v3_square_20261015
    RUN_ROOT=/path/to/project_data/EM/point-MAE/runs/voxel_mae_v5_reproduction_YYYYMMDD
    PYTHONPATH=$PWD "$PY" -m mito_voxel_mae_v5.launch_two --cache "$CACHE" --root "$RUN_ROOT"

启动器把 v2/v3/v4/v5 复制到 RUN_ROOT/code_snapshot/；A 绑定 GPU 0–2，B 绑定 GPU 3–5，并自动选择端口。随后逐级压测 tile batch 和 instance batch。当前机器本轮测得 tile probe 32 通过、instance batch 128 通过、256 OOM，故选 tile_batch=24、slice_microbatch=528、max_batch=128、budget=2048。不同 GPU 占用/显存可能选出不同调度参数；严格复核时以各组 tuning.json 与 config.json 的最终值为准。

### 10.3 单独启动一组

例如单独跑 A 时，确认输出目录没有旧 last.pt，再选好三张 GPU：

    cd /path/to/JaneliaEM/point-MAE
    PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
    CACHE=/path/to/prepared_cache
    OUT=/path/to/new_run_A
    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0,1,2 "$PY" -m torch.distributed.run --nproc_per_node=3 --module mito_voxel_mae_v5.train --cache "$CACHE" --out "$OUT" --objective voxel --epochs 15 --queries 1000 --views 4 --workers 2 --effective-batch 512 --save-every 5 --seed 42 --memory-fraction 0.72 --budget 2048 --max-batch 128 --tile-batch 24 --slice-microbatch 528

单独跑 B 使用相同命令和独立 OUT，将 --objective voxel 替换为 --objective lowfreq --lf-with-occ。不要让 A/B 意外重复占用同一组 GPU；单独启动也要确认 cache 已有符合要求的 audit.json。示例 /path/to/... 不是服务器的真实数据路径。

## 11. 运行产物与故障定位

每个实验目录预期包含：

- config.json：最终超参数、模型形状、query 与 DCT 配置；
- command_attempt*.json：启动命令及 GPU 划分；
- pressure_*.json/log、tuning.json：显存和运行校准；
- train_attempt*.log、steps.jsonl：训练过程和每 20 个 batch 的 step 指标；
- metrics.jsonl：每 epoch train/validation 汇总；
- skipped_rank*.jsonl：没有安全 query 等跳过原因；
- last.pt、best.pt、epoch_*.pt、可选 recovery.pt；
- 根目录 completed.json 或 failed.json：结束状态。

训练中的 DDP 日志可能出现 “Grad strides do not match bucket view strides” 警告。它本身不是梯度错误，但 PyTorch 提示可能影响效率。若 step/epoch 长时间不推进，应查看各 rank 日志、DataLoader worker、GPU 进程和 events.jsonl，不要仅凭进程存在判断训练正常。

训练器本身不导出 UMAP、PCA 或 embedding 文件。模型 forward 返回 out['z']，可作为 256D embedding；但独立 embedding extraction/export 脚本，以及推理时采用 mask 还是完整可见输入、如何聚合多 view 的协议，目前没有在 v5 中定义。任何下游使用都应将该推理策略写入独立配置。

## 12. 解读结果的限制

1. v5 遮挡是 tight bbox 上 4³ block 中隐藏 38/64，不是 80%。
2. Encoder 输入支持点是相对 17-grid；decoder query 是固定物理间距格点。前者覆盖实例形状，后者对应可比较的物理局部目标。
3. 相对 17-grid 经 points_um 保留实际物理尺寸；但 tile U-Net feature 的 XY stride 为 64 nm，32 nm 读取通过插值。
4. validation 只使用安全 query patch，patch 可重叠且每块至少含 2 个目标 voxel；IoU 不是完整实例 dense segmentation 指标。
5. B 只重建低频灰度，不应以 full-intensity MAE 单项和 A 判输赢。
6. 用于表征的输出是 256D encoder latent。架构说明不能证明它已有形态学可读性；需在 checkpoint 完成后以 held-out body 做独立 embedding/probe 评估。

## 13. 源码导航

- mito_voxel_mae_v5/model.py：v5 encoder 组 token、query decoder、A/B 输出与 loss。
- mito_voxel_mae_v5/data.py：缓存加载、增强/遮挡、point/query target、collate、DDP dummy。
- mito_voxel_mae_v5/targets.py：strict 3³ safe-query predicate、物理频率 DCT basis。
- mito_voxel_mae_v5/batching.py：空间工作量分桶与 rank 对齐。
- mito_voxel_mae_v5/runner.py、train.py：训练、validation、loss 汇总、checkpoint/resume。
- mito_voxel_mae_v5/launch_two.py：两组实验、压力探测、代码快照、OOM backoff。
- mito_voxel_mae_v5/pressure.py：tile/instance batch 显存与梯度压力试验。
- mito_voxel_mae_v5/checks.py：合成与少量真实数据正确性检查。
- mito_voxel_mae_v4/model.py：512 tile 与 point feature 读取。
- mito_voxel_mae_v3/model.py：SliceUNet、ChannelNorm、ZResidual。
- mito_voxel_mae_v2/model.py：FPS、Transformer、bottleneck、尺寸 head 的基础结构。
- mito_voxel_mae_v2/data.py：v5 复用的 point_support()。
- mito_voxel_mae_v4/native_data.py：v5 复用的几何增强与 tight bbox。

复现实验时先读运行目录的 config.json 和 command_attempt*.json，再对照该次 code_snapshot/。源代码后续若改变，snapshot 才是那次实验实际使用的代码版本。

## 14. 额外的实现复核注记

### 14.1 DCT 归一化约定

为避免不同库的 DCT normalization convention 不一致，v5 明确构造正交的一维长度 3 基：n,k∈{0,1,2}，phi_0(n)=1/sqrt(3)，k>0 时 phi_k(n)=sqrt(2/3)*cos(pi*(2*n+1)*k/6)。3D basis 是 Z、Y、X 三个一维 basis 的外积，展平顺序为 ZYX。目标系数再除以 sqrt(27)，预测逆变换乘回 sqrt(27)。实现采用显式余弦矩阵，不依赖 SciPy/PyTorch DCT 的默认归一化约定。

### 14.2 正确性测试与压力测试不是同一件事

checks.py 定义了独立的 synthetic/real-data correctness suite，但 launch_two.py 的正式启动流程不会自动调用它。压力测试主要验证代表性 tile/instance batch 能否完成前向、反向和 optimizer step；这不等价于运行完整 checks.py。本次检查运行目录时没有找到 checks.json，因此仅凭存在 checks.py 或 pressure_*.json，不能声称完整正确性测试已经执行并全部通过。复现实验建议先按第 10.1 节单独运行并保存 checks.json，再启动成对训练。
