# Mitochondria Point-MAE

第一版只处理 mitochondria，目标是把每个 3D instance 转成固定大小的 attributed point cloud，进行 masked point modeling，并输出 128D embedding。

## 设计约定

- 输入数组顺序：`ZYX`。
- 物理坐标：`coord_nm = voxel_zyx * spacing_zyx`，只减去 instance centroid，再除以全局固定 `coord_scale_nm`；不做 organelle-specific normalization。
- 每个 instance 默认 `1024 surface + 1024 interior = 2048` points。
- cache 永久保存四个完整属性：`intensity, gradient, distance, surface`。模型通过 YAML 的 `data.feature_names` 选择 ablation，因此不必重复预处理。
- raw intensity 使用 local crop 的 median/MAD normalization；gradient 使用真实 spacing；distance 使用物理单位 distance transform 后除以同一个全局 `coord_scale_nm`。
- train/validation/test 按 `neuron_id` 划分，不按 organelle instance 随机划分。
- 当前训练入口是 single-GPU；先完成 sampling 和小数据 overfit，再接 DDP。

## 1. Build cache

```bash
python preprocess.py \
  --instance /path/mitochondria_instances.zarr \
  --raw /path/raw.ome.tif \
  --out /path/mitochondria_cache.zarr \
  --spacing-zyx 30 8 8 \
  --neuron-map /path/instance_to_neuron.json
```

`instance_to_neuron.json` 是简单 JSON object，例如：

```json
{"1": "neuron_A", "2": "neuron_A", "3": "neuron_B"}
```

不提供 neuron map 仍可生成 cache，但 split 只能作为 smoke test，不能用于正式泛化评估。

## 2. Inspect sampling

```bash
python visualize_sampling.py --cache /path/mitochondria_cache.zarr --index 0 --out sampling_0000.png
```

## 3. Model smoke test

```bash
python smoke_test.py
```

## 4. Train

```bash
python train.py \
  --config configs/mitochondria_medium.yaml \
  --cache /path/mitochondria_cache.zarr \
  --output /path/pointmae_runs/mitochondria_medium
```

支持 `--resume /path/last.pt`。训练会写 `last.pt`、`best.pt` 和 `metrics.jsonl`，并在开始时打印总参数量和可训练参数量。

三个 preset：

```text
mitochondria_small.yaml   # 约 2M 级别
mitochondria_medium.yaml  # 约 5–8M 级别附近
mitochondria_large.yaml   # 约 15–25M 级别附近
```

## 5. Extract embeddings

```bash
python extract_embeddings.py \
  --config configs/mitochondria_medium.yaml \
  --cache /path/mitochondria_cache.zarr \
  --checkpoint /path/pointmae_runs/mitochondria_medium/best.pt \
  --out /path/mitochondria_embeddings.npz
```

输出包含 `embedding[N,128]`、`instance_id`、`neuron_id` 和 `organelle_type`。提取时不使用 masking。

## Feature ablation

只修改 YAML：

```yaml
feature_names: [intensity, gradient, distance, surface]
```

可改为：

```yaml
feature_names: []
feature_names: [intensity]
feature_names: [intensity, gradient]
```

坐标始终输入；22D handcrafted features 不进入 SSL encoder。
