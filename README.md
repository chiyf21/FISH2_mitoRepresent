# Fusion50 mitochondrial representation learning

This branch implements native 768D Point and Image CLS readouts. Historical
384D Fusion50 remains available on its original branch.

Block 9-11 use parameter-free LayerNorm on the exported CLS. Block 12 takes the
native final LayerNorm output. Fusion directly concatenates both 768D CLS without
a modality projection, PairBalance, or input LayerNorm. The Project384 head retains
its output LayerNorm. Point-only NNA consumes raw 768D block CLS, as in the historical objective.
Monitoring and fusion use the normalized CLS readouts. Fusion uses VICReg and filtered InfoNCE on Project384, plus variance
and covariance regularizers on the normalized per-modality CLS.

## Run

Use the project PyTorch environment or install the dependencies in `fusion50/requirements.txt`. Run these commands from the repository root. In the commands below, `$RUN` is a new absolute output directory outside Git. For CUDA training, set the visible GPUs to match the requested process count.

The input manifest is JSON, either a list or an object with a `records` list. Each row contains `body_id`, an instance identifier (`record_id` or `global_instance`), a body-disjoint `split` (`train`, `val`, or `test`), and paths to aligned ZYX `em.npy` and binary `mask.npy` arrays. Use `em_path` and `mask_path`, or `relative_dir` with those filenames. Paths can be absolute or resolved via `--source-root`. Provide consistent physical voxel spacing in the Point-MAE config. The preparation command samples two independent sets of surface points and writes EM blocks that intersect train masks only. Supply the masks you want the model to use, including any boundary repair or hole filling required for your dataset.

```sh
python fusion50/prepare.py --manifest /data/instances.json --output "$RUN/cache" --image-blocks
```

Copy `fusion50/point/config.example.json` to `$RUN/point.json`. Set `input.path` to the cache, `output` to `$RUN/point` and `spacing_xyz_nm` to the data spacing (8,8,8 for LC21). Set mae.trans_dim=768, mae.heads=12 and mae.decoder_heads=12 for this refine model.

```sh
python fusion50/point/pipeline.py --config "$RUN/point.json" --prepare-only
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  fusion50/point/train_ddp.py --config "$RUN/point/config.resolved.json"

CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 \
  fusion50/image/main_pretrain.py --config fusion50/image/config.json \
  --data_path "$RUN/cache/image_blocks" --npy_axes zyx \
  --output_dir "$RUN/image" --log_dir "$RUN/image"
```

The defaults represent the historical 100-epoch Point-MAE and 120-epoch image checkpoint. Choose an image epoch budget using measured throughput if needed. FOCUS3D batch size is per GPU. Point-MAE batch size is global. A previously prepared FOCUS3D block manifest can also be passed directly to `--data_path`; use its actual axis order.

Copy `fusion50/refine/config.example.json` to `$RUN/refine.json` and set the point-run directory and both MAE checkpoint paths. Set source_run_id to the recorded source for a legacy manifest lacking this field.
The example stops after 20 Point-only NNA epochs. To start fusion in a new output
directory, set fusion_epochs to the cosine schedule length (historically 50),
set fusion_stop_epoch to the desired initial stop, and pass
--init-point-cls with the final Point-only checkpoint. To resume that fusion
job, use its resolved config, the same rank count and --resume.
The trainer uses global batch 64 and the original Point body split.

```sh
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 \
  fusion50/refine/train.py --config "$RUN/refine.json" --output "$RUN/refine"

CUDA_VISIBLE_DEVICES=2 python fusion50/refine/export.py \
  --config "$RUN/refine.json" --checkpoint "$RUN/refine/last.pt" \
  --output "$RUN/embeddings"
```

Export creates `point_12.npy`, `image_12.npy`, `concat_12.npy`, `fused_12.npy` and `instance_map.csv` in the same row order. Input data, run outputs and weights stay outside the checkout. Load only trusted checkpoints. [Source provenance](SOURCE_MAP.md).
