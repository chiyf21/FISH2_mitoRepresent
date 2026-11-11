# 线粒体 MAE v5：严格隐藏的 3×3×3 重建

本次两实验共用同一数据划分、随机种子42、四种在线视图、输入点与安全query规则，训练15 epoch。

- A：27体素 occupancy +27体素 intensity +0.2倍 log物理尺寸回归。
- B：27体素 occupancy +3D低频灰度重建 +0.2倍 log物理尺寸回归。B必须启用`--lf-with-occ`。
- occupancy为前景/背景平衡BCE +0.5倍Dice。A灰度SmoothL1按前景0.7、背景0.3加权。B在低通逆变换图像上计算SmoothL1，与A保持灰度单位；其频域模式跨越前后景，未再套用前景/背景像素配额。

## 图像与输入点

原始采集分辨率XYZ=8×8×30 nm。实例XY扩成正方形的缓存保留原生分辨率，不把整个实例缩放到512。大实例以512×512原生图像tile提特征，核心256×256，XY halo各128；Z核心16层、halo各3层。切片送入共享2D U-Net批处理，再经三层不下采样Z的轻量残差融合。输入通道为隐藏后的灰度、目标instance mask、可见性和有效原始图像区域。

输入采样恢复17×17×17相对点阵，只取目标instance内部及26邻接点阵位置的支持点，再筛选可见且原始数据有效的点。17³是候选点阵，实际输入点数可变。编码器以中心为原点的物理坐标(µm)及图像特征形成64个局部分组token，经6层Transformer和256维瓶颈产生实例embedding。删除直接bbox边长输入；尺寸仍可从物理几何推断，尺寸回归仅是辅助监督。

## 严格query规则

MAE按4×4×4分块，随机隐藏38块。“重建crop完全在mask内”解释为完全位于MAE隐藏区域，不是27个体素全在目标GT内部。

1. query候选为XYZ=32×32×30 nm格点；Z每层都取。
2. 以query为中心的27个原始体素必须全部有效、全部隐藏，不能有任一体素与可见输入区域重叠。
3. 27个体素中至少2个属于目标instance GT，其余可以是其他实例或背景，occupancy始终只标记目标实例。
4. 每实例/视图最多1000个不同query，不足不重复补齐。新规则不再执行旧700/300 query中心配额。
5. 没有安全query就跳过本实例/本视图并记录；不删除整条原始instance。DDP某rank局部为空时使用零权重且梯度连通的填充，避免多卡死锁。
6. crop实际物理覆盖XYZ=24×24×90 nm。query之间可以重叠，但所有目标体素都必须隐藏。
7. 灰度归一化的均值与标准差仅用可见像素计算，隐藏强度不会通过归一化统计影响输入。

## 3D低频目标

在3³小块上计算正交3D DCT-II，按物理频率`sum((k/(2*N*spacing))²)`排序，保留最低10个模式。XYZ分辨率不同，因此不是简单按DCT索引大小选频率。系数按sqrt(27)归一化，训练时逆变换为低通灰度块后计算损失，避免系数尺度使B灰度损失天然偏小。

## 全量缓存与运行

缓存19947个实例，来自184个神经元：train15508、val1953、test2486，按神经元互斥。四个在线视图对应每epoch62032个潜在训练视图，实际数扣除无安全query的视图。增强沿用轻微噪声、尺度/剪切、旋转/位移和弹性形变。

113缓存：`/path/to/project_data/EM/mito_voxel_mae_prepared_v3_square_20261015`。

双实验根目录：`/path/to/project_data/EM/point-MAE/runs/voxel_mae_v5_strict333_3x2_20261015`。

- `expA_occ_intensity`：GPU0、1、2，三卡DDP。
- `expB_occ_lowfreq`：GPU3、4、5，三卡DDP。
- `code_snapshot/`：两实验共用本次代码副本。
- `checkpoint_policy.json`：最新保存策略`save_every=5`。独立checkpoint在epoch5、10、15保存；last每epoch覆盖，best在验证改善时更新。
- 每实验的`pressure_*.json/log`记录逐级增大tile和实例batch的三卡前向、反向和optimizer测试；`tuning.json`记录选择值。实例batch按Z和空间工作量分桶，最大batch是代表性实例测得的上限，不代表任意形状都可用同样batch。
- `train_attempt*.log`、`steps.jsonl`、`metrics.jsonl`和`skipped_rank*.jsonl`分别记录过程、步数、epoch指标与跳过原因。
- 若全量训练遇到不同形状导致的OOM，启动器降低batch工作预算和tile微批量，从最近checkpoint重试；其他异常会写入根目录`failed.json`。

## 验证及指标限制

检查包括3³安全规则与暴力计算一致、改变全部隐藏灰度不影响编码器输入、17³点阵、可变query padding、DCT物理选频、A/B有限梯度、零权重DDP空rank和单实例/批量embedding一致性。全缓存审计检查所有mask及前景原始灰度与源数据一致。

日志中的occupancy IoU是经安全query选择的重建小块指标，重叠体素可能重复计数，不是完整实例的无偏体素IoU。B仅输出低频灰度，因此其完整灰度MAE不能单独用于公平判定细节重建质量；应共同比较occupancy、低频误差、尺寸误差与后续embedding指标。
