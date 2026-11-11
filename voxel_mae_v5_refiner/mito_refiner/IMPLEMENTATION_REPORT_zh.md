# 历史 v1 报告（不是本次 v5 训练结果）

本文随旧代码复制保留。当前 v5 的修订和已执行检查范围请阅读上级
`REFINER_ADDON_ZH.md`；本轮未启动训练，旧报告中的通过记录不能视为本轮验证。

# Mito-Refiner（R0）实现与小规模验证记录

日期：2026-10-14（北京时间）

## 结论先说

已在服务器快照目录新增独立 `mito_refiner/`，没有改动原 Voxel-MAE 代码、原 checkpoint 或缓存。Hard NNA 的两阶段流程、虚拟 batch、队列和评估脚本已通过核心单测及端到端小样本验证。当前只是一轮 smoke test，不足以判断生物学聚类质量是否改善。

代码：`/path/to/JaneliaEM/mito_voxel_mae_cyf/mito_refiner/`

小型结果：`/path/to/JaneliaEM/mito_voxel_mae_cyf/runs/mito_refiner/r0_hard_nna_smoke/`

Encoder 梯度复核：`/path/to/JaneliaEM/mito_voxel_mae_cyf/runs/mito_refiner/r0_hard_nna_gradcheck/`

## 接入的模型表征

原模型接口 `VoxelMAE.encode(batch)` 只返回 bottleneck；refiner 没有改这个接口，而是对 Transformer 的 `layers[3]`、`layers[4]`、`layers[5]` 加临时 hook，分别对 64 个 token 做 MeanPool，得到 Block 4/5/6 MeanPool。原 `encode()` 返回的 256D bottleneck 也单独保留。

R0 默认三个 Head 接：

1. Block 4 MeanPool（256D）
2. Block 5 MeanPool（256D）
3. 原始 Encoder bottleneck（256D）

每个 head 独立训练。`--final-head-input block6_mean` 可把第三个 Head 改接 Block 6 MeanPool；这只是消融选项，不把 Block 6 pooling 和 bottleneck 混称。下游评估始终使用未经过 ID Head 的 256D Encoder bottleneck。

## R0 的 Head、Queue 和损失

每层 ID Head 是三层 Projector、两层 Predictor。官方论文的隐藏宽度为 2048/4096；考虑我们的 256D backbone，当前默认各缩为 512，仍保留 GELU/BatchNorm 和层数。Projector 输出作为队列特征，Predictor 输出作为 anchor；每层使用独立 FIFO queue。每个队列项保存 projector feature、record ID、body ID 和 step。

R0 默认 Hard Top-1：从旧 queue 中取最近邻作为正样本，当前 virtual batch 中其它 record 的 projector 输出作为 stop-gradient negatives。相同 record ID 的历史队列项和 batch negative 会排除；同 body 的其它线粒体保留，并统计最近邻 same-body 比例。Queue 只在完整 virtual batch 结束后统一写入。空队列 warm-up 批次不计算无效 loss，但写入当前 batch，避免 NaN。

原论文的 `Top-K` 不是 Soft Positive：官方 queue 在 Top-K 候选中随机抽一个；本代码保留为 `--positive-mode official_random_topk --topk 20`。R1 的 `soft_temperature`、固定/线性/cosine 温度、entropy target 都有独立接口和极限单测，但本轮没有启动 R1 训练。R2 只分析未精修模型的邻域，没有执行 consensus refinement。

方法依据： [MIM-Refiner 论文](https://arxiv.org/abs/2402.10093)；[官方 NNA loss](https://github.com/ml-jku/MIM-Refiner/blob/main/src/losses/ssl/nnclr_loss.py)；[官方 queue](https://github.com/ml-jku/MIM-Refiner/blob/main/src/modules/ssl/nnclr_queue.py)；[官方 ID Head](https://github.com/ml-jku/MIM-Refiner/blob/main/src/models/ssl/nnclr_head.py)。

## 两阶段训练与虚拟 batch

Stage A 冻结 U-Net、tokenizer、Transformer 和 bottleneck。用固定无 mask view 预缓存 train split 的 Block 4/5/6 与 bottleneck 特征，只更新 ID Heads。Stage B 默认冻结 U-Net/tokenizer，仅解冻 Transformer+bottleneck；可用 `--unfreeze-frontend` 单独做完整 Encoder 解冻消融。两阶段只用 NNA，不叠加 MAE reconstruction、VICReg、NT-Xent 或 I-Con。

由于 U-Net 输入 crop 大小不同，保持 microbatch=1，但对 8 个不同 record ID 做 Virtual Contrastive Batch：Pass 1 在无梯度模式缓存当前参数下的 features/projector keys；Pass 2 固定同一组 CPU 输入，逐个重跑 anchor forward、累计平均梯度；最后一次 optimizer step，再统一入队和更新 EMA。两遍均设 `eval()`，关闭 Transformer dropout，冻结 Head BatchNorm running stats（仍允许其仿射参数反传）；Voxel-MAE 前端使用 GroupNorm。这样每个 anchor 的网络计算是确定的，避免把随机 view/dropout/BN 变化混成梯度不一致。

单测在确定性合成 head 上逐参数比较 True Batch 与 Virtual Batch，loss 和所有梯度一致；真实 Voxel-MAE 的单步日志进一步确认 Encoder/Head 均有非零梯度。该结论不外推到训练态 BatchNorm/Dropout 网络。

## 本轮实际运行

- 预训练权重：快照中的 `results/voxel_mae_native_v1_20261014/best.pt`；校验值仍为 `42bc219b0df0e061c93f8ee900a43707ef060a9d14fbeab7b6b533bd5e431ebe`。
- 数据使用原来的 body-level split 与 cache；train queue 只读 train。Smoke 只抽了 16 个 train 实例和 24 个 held-out test 实例，没有把 validation/test 放进 queue。
- GPU：113 号服务器物理卡 6；没有 OOM。
- Head Init：第一组 queue 为空，0 个 anchor 参与 NNA（正常 warm-up），随后写入 8 条/层。第二组每层 8/8 个 anchor 有效，三层总 loss `6.4780`；结束时每层 queue 有 16 个实例。
- Encoder refinement：virtual batch 为 8，Block 4/5/final 各有 8/8 个有效 anchor，合计 loss `6.2091`。独立复核的 pre-clip grad norm：Transformer+bottleneck `1.8995`，ID Heads `15.2109`，两者均有限且非零。
- 原始与精修后 24 个 test bottleneck 的平均 cosine 为 `0.999983`，15NN Jaccard 为 `0.9948`。只更新一步且 encoder LR 为 `1e-5`，这种极小变化符合 smoke 的目的；不能据此说 NNA 已改善表示。
- 24 个 test 样本上的 Effective Rank `10.56`、Participation Ratio `7.77`。该估计样本数只有 24（中心化后最大秩 23），不应与 8,822 样本的正式 embedding rank 直接比较。
- body-disjoint ridge probe（16 train / 24 test）R²：log-volume `0.573`，physical elongation `0.128`，log-surface-area `0.645`，internal intensity contrast `0.066`。样本极小，只说明评估链路能跑。
- 8 个实例的 4-view 平均 cosine：base `0.9842`，refined `0.9840`。仍是小样本流程检查。
- 未精修模型 Block4/Block5/final 的 Top-1 邻居一致率依次：`0.833`、`0.583`、`0.750`。Top-20 因为只有 24 个 test 实例，覆盖 23 个候选里的 20 个，交集/覆盖率明显受候选集太小影响，不能据此得出 consensus 很强。

结果文件包括：`head_init.pt`、`refined.pt`、train 特征缓存、`evaluation/metrics.json`、`cross_layer_nn_diagnostics.json`、`feature_recovery.csv`、`anchor_neighbors.csv` 和 embedding NPZ。核心训练行在 `training_metrics.jsonl`。

## 代码文件

| 文件 | 功能 |
|---|---|
| `features.py` | 临时 hook 提取 Transformer 中间层与 bottleneck |
| `heads.py`, `models.py` | MIM-Refiner 风格 ID heads |
| `nna.py`, `queue.py` | NNA hard/soft loss、Top-K 正样本选择和带元数据的 FIFO |
| `virtual_batch.py` | True-batch loss及两遍虚拟 batch 的 anchor loss |
| `train.py` | 冻结 head 初始化与 Encoder refinement 两阶段训练 |
| `evaluate.py` | Effective Rank、probe、增强稳定性、15NN、hubness、固定锚点输出 |
| `neighbor_consensus.py` | R2 跨层 Top-1/Top-K、Mutual KNN、共邻居覆盖和 proxy 差异 |
| `tests/test_core.py` | NNA/queue/虚拟梯度、Soft 温度极限、R2 空交集等检查 |
| `run_smoke.sh` | 固定服务器路径的小样本端到端命令 |

运行单测：

```bash
cd /path/to/JaneliaEM/mito_voxel_mae_cyf
export PYTHONPATH="$PWD:$PWD/code"
/home/cyf/miniconda3/envs/cellmap_organelle/bin/python -m mito_refiner.tests.test_core
bash mito_refiner/run_smoke.sh
```

正式训练前应把样本数扩大，并跨随机种子重复，查看队列正样本 same-body 率和邻域稳定性；再比较 Base/Head-only/Encoder-refined 的相同 held-out 评估。当前 16/24 样本结果仅是 smoke baseline。
# 历史说明

下文为复制保留的旧 v1 Refiner 报告，不是本次 v5 的训练或验证结果。
当前实现与本轮检查范围请阅读上级目录 `REFINER_ADDON_ZH.md`。

