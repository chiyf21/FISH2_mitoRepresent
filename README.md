# Fusion50 mitochondrial representation learning

Fusion50 trains two masked autoencoders separately, applies 20 epochs of point-only nearest-neighbor alignment (NNA), then trains the two encoders and a shared fusion head for 50 epochs. The standard output is the 384-dimensional L12 embedding.

| Stage | Entry point | Model |
| --- | --- | --- |
| Surface points | `fusion50/point/train_ddp.py` | Point-MAE, 2,048 surface points, 64 groups, 70% masking, 384-dimensional encoder |
| EM image | `fusion50/image/main_pretrain.py` | FOCUS3D, 32×128×128 ZYX crops, 2×8×8 patches, 50% masking, 768-dimensional encoder |
| NNA and fusion | `fusion50/refine/train.py` | Point CLS and last four blocks for NNA, then both CLS tokens, both encoders' last four blocks and a shared MLP |
| Embeddings | `fusion50/refine/export.py` | Canonical L12 fused vectors plus an instance map |

Fusion uses the original per-layer CLS readouts at L9–L12, modality variance balancing, LayerNorm, image projection and concatenation before the MLP. The objective combines VICReg, filtered InfoNCE and modality CLS regularizers. Positive pairs are two views of the same instance. The 20 nearest other instances are excluded from the negatives for each anchor when the batch permits.

## Run

Use the project PyTorch environment or install the dependencies in `fusion50/requirements.txt`. Run these commands from the repository root. In the commands below, `$RUN` is a new absolute output directory outside Git. For CUDA training, set the visible GPUs to match the requested process count.

The input manifest is JSON, either a list or an object with a `records` list. Each row contains `body_id`, an instance identifier (`record_id` or `global_instance`), a body-disjoint `split` (`train`, `val`, or `test`), and paths to aligned ZYX `em.npy` and binary `mask.npy` arrays. Use `em_path` and `mask_path`, or `relative_dir` with those filenames. Paths can be absolute or resolved via `--source-root`. Provide consistent physical voxel spacing in the Point-MAE config. The preparation command samples two independent sets of surface points and writes EM blocks that intersect train masks only. Supply the masks you want the model to use, including any boundary repair or hole filling required for your dataset.

```sh
python fusion50/prepare.py --manifest /data/instances.json --output "$RUN/cache" --image-blocks
```

Copy `fusion50/point/config.example.json` to `$RUN/point.json`. Set `input.path` to the cache, `output` to `$RUN/point` and `spacing_xyz_nm` to the data spacing (8,8,8 for LC21). Both MAEs can train in parallel.

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

Copy `fusion50/refine/config.example.json` to `$RUN/refine.json` and set the point-run directory and both MAE checkpoint paths. The refine trainer runs NNA then fusion. It uses a global batch of 64 for the historical 4-GPU configuration and shares the point run's body split.

```sh
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 \
  fusion50/refine/train.py --config "$RUN/refine.json" --output "$RUN/refine"

CUDA_VISIBLE_DEVICES=2 python fusion50/refine/export.py \
  --config "$RUN/refine.json" --checkpoint "$RUN/refine/checkpoints/refine_epoch050.pt" \
  --output "$RUN/embeddings"
```

Export creates `fused_12.npy` and `instance_map.csv` in the same row order. Input data, run outputs and weights stay outside the checkout. Load only trusted checkpoints. [Source provenance](SOURCE_MAP.md).

## Balanced1152D without a projection head

The entry point fusion50/refine/train_no_projection.py starts directly from a
trusted, completed NNA20 checkpoint. It preserves the baseline cohort, point
cache, physical spacing, four-rank sampler, RNG states, PairBalance, L9–L12
losses, both CLS regularizers and the original 50-epoch optimizer schedule.
Only both CLS tokens and both encoders' last four blocks are trainable.
The entire Fusion MLP is frozen, excluded from AdamW and guarded against execution.
Losses receive PairBalance's 1152-dimensional output directly.

Create an external JSON configuration from the baseline configuration, adding
without_fusion_head=true, baseline_run, init_pretrain, point_checkpoint,
image_checkpoint, spacing_xyz_nm, and smoke_result (the path to the smoke
output's complete.json). Retain the baseline's pointclouds, seed, batch size,
world size and learning rates. The trainer reads the baseline's exact cohort and
NNA20 calibration probe; it does not rerun either MAE or NNA.

    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=4 \
      fusion50/refine/train_no_projection.py --config /runs/nohead/config.json \
      --output /runs/nohead/smoke --smoke
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=4 \
      fusion50/refine/train_no_projection.py --config /runs/nohead/config.json \
      --output /runs/nohead

The two-batch smoke checks all four gradient groups, frozen parameters, actual
updates in each trainable block and CLS, unchanged unused MLP weights, finite
loss/gradients, GPU memory and full checkpoint/optimizer/RNG restoration. A
separate frozen reference MLP measures old and new losses on the same first batch
without contributing to training. Production requires a passing smoke result.
Every epoch saves a full atomic checkpoint and checksum manifest; last.pt
supports --resume with the same four ranks. STOP_AFTER_EPOCH pauses at a saved
epoch boundary. Keep all outputs outside Git.

The existing fusion50/refine/export.py reads the no-head configuration and
exports 1152-dimensional L12 vectors to fused_12.npy with the usual
instance_map.csv for the existing downstream LR workflow.

The calibrated no-head configuration may set main_covariance_scale=0.002,
changing only the main VICReg covariance coefficient from 0.2 to 0.0004.
Raw Point/Image CLS coefficients remain 1.25 for variance and 0.05 for covariance.
calibrate_no_projection.py compares both representations from the same NNA20
checkpoint and two fixed batches without optimizer updates, including encoder
gradient norms and directions. Do not initialize formal training from smoke
checkpoints or any checkpoint that has received Fusion updates.

A selected smoke can include original batch numbers 1,2,41,42 to cover an input
that previously stalled a rank. Per-rank phase files distinguish preprocessing,
forward, backward and optimizer progress. distributed_timeout_seconds configures
the collective timeout without changing data, augmentation or batch assignment.
Production still starts from NNA20, writes an untouched refinement epoch000,
then runs all 50 epochs with the original optimizer schedule.

For the known large input on this host, NUMPY_MADVISE_HUGEPAGE=0 disables NumPy's
process-local transparent-huge-page advice. It changes allocation behavior only,
and leaves all arrays, RNG draws and augmentation mathematics unchanged. The
calibrated configuration records and checks this environment setting.
