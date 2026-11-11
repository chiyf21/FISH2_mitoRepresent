# v5 + MIM-Refiner R0：代码修订说明（2026-10-16）

工作目录：`/path/to/JaneliaEM/point-MAE/mito_voxel_mae_v5_refiner_r0/`。
修订前备份：同级 `mito_voxel_mae_v5_refiner_r0_backup_20261016/`。
本轮为代码修正和 CPU 无优化器更新的接口检查；未执行任何训练入口、带optimizer
更新的单元检查、真实数据 Refiner 训练、显存压测或效果比较。等待用户明确通知。

## 1. 与 v5 保持一致的部分

保留原生 EM 分辨率和可变实例尺寸；完整实例 mask 构造 tight bbox 和 17³
候选支持点，包含内部及 26 邻接 halo。支持点通过 mask/valid/observed 筛选。
图像来自 512² tile（XY 核心 256²、halo128；Z核心16、halo3），共享切片
U-Net + 三个 Z 融合模块输出点特征。FPS形成64组，每组最多32点，局部
相对物理坐标与图像特征经 MLP/max pooling，再添加中心物理位置编码。
六层 Transformer 输出 `[N,64,256]`，MeanPool `[N,256]`，bottleneck 输出
256D。权重字段与原始 v5 一致，不修改原 v5 的代码、checkpoint 或缓存。

Refiner 副本的 `image_backbone.py` 固定了本次使用的 v4 图像实现；只改变
activation checkpoint 的控制条件。仍需要同一项目中的 `mito_voxel_mae_v2`
（基础网络/FPS/支持点）、`mito_voxel_mae_v3`（SliceUNet/ZResidual）和
`mito_voxel_mae_v4.native_data`（tight_box），迁移时需一起提供这些依赖。

## 2. Encoder-only 数据路径

`MitoDataset(..., encoder_only=True)` 不创建 decoder query 或重建标签，不因
“没有隐藏 query”跳过全可见样本。默认 `encoder_only=False` 保留原 MAE
数据行为，单元检查会与原始 v5 逐字段比较。

`mito_refiner/data_utils.py` 是唯一数据适配实现，旧 `v5_data_utils.py` 仅重导出。
Refiner 使用完整可见 EM 和实例 mask，输入归一化仍用 observed/valid 范围。
两个 Global Views 使用 kind0（原形+轻微输入噪声）与 kind2（轻微物理XY旋转/
平移+独立噪声）；两者来自同一实例。当前不使用 scale/shear/elastic 或 local views。
增强失败仍沿用 v5 的回退规则，可能回退到原始形状。

完整 mask 的 conditioning 是设计的一部分：mask 通道、支持点选择和 bbox
坐标均依赖实例分割。因此这里不声称隐藏形态完全不可见，也不是 raw-EM-only
的无条件表征模型。当前 Refiner 没有空间 masking；MAE 预训练仍按自身规则执行。

## 3. Head、梯度和 NNA

默认三个独立 Head：B4 MeanPool、B5 MeanPool、B6 MeanPool。每个包含
三层 Projector、两层 Predictor 和真实 BatchNorm。三个 Head loss 求和；
每个 Head 内两个 cross-view CE terms 求平均。仅 NNA 用于独立 refinement，
此阶段不联合回传 MAE reconstruction loss。

Projector 的 preswap Key detach 后用于 Queue 查询和 off-diagonal negatives；
Student Predictor 保持梯度，沿 Predictor→Projector→Encoder 回传。
每个 Head 拥有独立 FIFO Queue，排除同 record_id；Stage A Top-1，Stage B
从 Top-20 均匀随机取单个近邻作为 positive。Queue 提供 positive，负样本来自
当前实例 batch。全部 loss/梯度计算完成后，optimizer 更新一次、可选 EMA
更新一次，再将更新前的 view0 Key 入队一次。

默认 B4/B5/B6 不回传 bottleneck；代码显式冻结 bottleneck 和未使用的
reconstruction/size head，并保存 `bottleneck_refined=false`。Block6 MeanPool
为主要诊断表征，最终256D也保留并单独评价（其输入会变化，bottleneck参数不变）。
如在 Stage A 使用 `--final-head-input bottleneck`，三个 Head 变成 B4/B5/final，
bottleneck 参与梯度；Stage B 必须沿用同一 Stage A Head 配置。
最终用于分析的向量均来自 Encoder/bottleneck，不使用 ID Projector 输出。

## 4. 真实 batch 与 Replay

默认 `--batch-mode true`。B个实例各取两view，按 `[i0v0,i0v2,i1v0,i1v2,...]`
排列，v5 collate 保留各原生 volume，只 padding 支持点。Encoder 真实输入
`[2B,...]`，Head 一次接收 `[2B,D]`，再还原 `[B,2,D]`。B=8时每个方向
logits 为8×8，各行7个negatives；BN使用16行特征。

`--batch-mode replay` 保留低显存路径：no_grad逐view提取，多层特征成为
完整batch的叶节点；Head统一反传后逐view重算 Encoder，用所有层梯度联合
VJP回传。与真实batch共用 `training_step.cross_view_heads()`，queue与head
的执行次数/顺序一致。该方法不同于把NNA拆成多个小batch后梯度累积。
不自动捕获OOM并切换策略；需要显式选择 replay。大实例是否能组成目标真实
batch，仍要等用户通知后做真实GPU压测。

模型使用 eval() 关闭 Transformer Dropout，但训练参数仍有梯度。
activation checkpoint 使用独立开关 `--activation-checkpointing`（默认开启），
按 `torch.is_grad_enabled()` 判断；eval 下的反向仍可checkpoint，no_grad缓存/
推理不checkpoint。关闭方式：`--no-activation-checkpointing`。
两种策略使用相同的 `--precision`（CUDA默认bf16，CPU检查必须fp32），Head/NNA
统一FP32。TF32关闭，Replay漂移超阈值时在optimizer更新前失败。

`--slice-microbatch 8 --tile-batch 2` 为独立运行默认值，可调；不会直接继承
预训练checkpoint里可能很大的528/24工作批量。tile microbatch不会改变NNA
的实例batch。当前训练入口为单设备，显式拒绝 WORLD_SIZE>1；尚未实现跨卡
feature gather/全局BN，不能直接套用原MAE的DDP启动命令。

失败view成对剔除并记录，剩余实例少于2时跳过；不向BN/NNA加入MAE零权重
填充。尾部仅1实例时沿用旧逻辑并入前一组，实际B可能为配置值+1；日志记录
实际实例/view数。Encoder和Head分别采用 `--encoder-lr`/`--head-lr`。

## 5. Checkpoint 与评价

周期保存 `last_init.pt` / `last_refine.pt`（`--save-every`，默认50个已处理batch），
阶段结束保存 `head_init.pt` / `refined.pt`。包含模型/Head/Queue、optimizer、
Python/NumPy/Torch/CUDA RNG、epoch/batch游标、源checkpoint SHA256和样本顺序。
`--resume` 恢复中间位置，不自动覆盖已有运行；恢复时检查来源、cohort及关键
数值配置。Stage B可选Encoder EMA（默认0.999，设0关闭），它不是Momentum Projector。

主要embedding定义为：完整可见、无增强、无噪声、eval/no_grad的单个256D向量。
两增强view的稳定性单独计算，不把带噪view0充当主评估输入。Block6 MeanPool和
final均导出；按神经元互斥的train/test split做linear probe，沿用当前manifest的
全部test实例，不写死旧的849数量。纹理proxy直接从原始缓存EM计算；表面积
用物理体素暴露面估计（已修正uint8差分溢出），不是平滑mesh表面积。
多遮挡view聚合、跨卡训练、Soft NN、共识loss均尚未加入本版。

## 6. 代码检查命令（无需GPU/真实训练）

```bash
cd /path/to/JaneliaEM/point-MAE
PY=/home/cyf/miniconda3/envs/wbi_cuda124/bin/python
CUDA_VISIBLE_DEVICES='' "$PY" -m mito_voxel_mae_v5_refiner_r0.mito_refiner.tests.test_v5_integration \
  --interfaces-only --out /tmp/v5_refiner_code_checks.json
```

interfaces-only覆盖全可见无query、可变形状collate、双view配对及成对跳过、原MAE
数据回归、缩小通道模型的批量/逐个/原v5前向一致性、eval下checkpoint前向及评价接口。
不去掉该开关就不会执行backward/optimizer检查。

另外已准备但本轮未执行的检查：原test_core、True/Replay loss/梯度/更新、默认/
可选bottleneck梯度、合成fixture上Stage A/B训练器的游标/Queue/EMA恢复。
待用户允许开展实验后再运行，不能把脚本已经写好表述为这些检查已经通过。

## 7. 后续实验命令模板（等用户通知后执行）

```bash
cd /path/to/JaneliaEM/point-MAE
PY=/home/cyf/miniconda3/envs/wbi_cuda124/bin/python
CACHE=/path/to/project_data/EM/mito_voxel_mae_prepared_v3_square_20261015
CHECKPOINT=/path/to/selected/v5/best.pt
OUT=/path/to/new/refiner_run

CUDA_VISIBLE_DEVICES=7 "$PY" -m mito_voxel_mae_v5_refiner_r0.mito_refiner.train_v5 \
  --phase init --cache "$CACHE" --checkpoint "$CHECKPOINT" --out "$OUT" \
  --limit 128 --batch-size 8 --epochs 10 --max-steps 50 --seed 42

CUDA_VISIBLE_DEVICES=7 "$PY" -m mito_voxel_mae_v5_refiner_r0.mito_refiner.train_v5 \
  --phase refine --cache "$CACHE" --checkpoint "$CHECKPOINT" --out "$OUT" \
  --head-init "$OUT/head_init.pt" --limit 128 --batch-size 8 --epochs 10 \
  --max-steps 20 --batch-mode true --precision bf16 --seed 42

# 低显存对照使用新的输出目录、相同Stage-A文件及配置，改成 --batch-mode replay。
# 中断恢复追加 --resume "$OUT/last_refine.pt"；max-steps为累计处理batch上限。

CUDA_VISIBLE_DEVICES=7 "$PY" -m mito_voxel_mae_v5_refiner_r0.mito_refiner.evaluate \
  --cache "$CACHE" --base-checkpoint "$CHECKPOINT" \
  --refined-checkpoint "$OUT/refined.pt" --out "$OUT/evaluation" \
  --train-limit 1024 --eval-limit 0 --representations block6_mean,final
```

未来替换MAE时，主要替换adapter的load/extract和数据适配；Head、NNA、Queue
以及完整特征batch的loss逻辑无需跟随主干重写。上述命令仅为模板，本轮未执行。

## 8. 本轮已执行的检查记录

2026-10-16：全目录语法检查和train/evaluate参数导入检查通过；interfaces-only
检查通过。合成输入中，批量/逐个前向最大误差约3.5e-6；与原v5相同权重的
前向差异为0。检查结果保存在 `code_checks_20261016/interface_checks.json`。
本轮没有backward或optimizer更新，也未运行Stage A/B训练入口。
梯度/参数更新对照、训练器恢复、完整模型GPU显存测试均仍待执行。
