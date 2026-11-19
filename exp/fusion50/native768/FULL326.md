# Full326 native 768D fusion

The Full326 baseline uses a Phase 1 Point CLS checkpoint followed by per-view
VICReg refinement. Block 12 Concat (1536D) is the default downstream readout.
Projection (384D), Point CLS and Image CLS are available for separate evaluation.

## Point-MAE and Point CLS

| Stage | Settings |
| --- | --- |
| Point-MAE reconstruction | Unit-sphere normalization after physical XYZ conversion at 8, 8 and 30 nm. 2,048 points, 64 FPS groups of 32, 70% group masking, 768D tokens, 12 encoder blocks and four decoder blocks, each with 12 heads. Symmetric squared Chamfer L2. |
| Point-MAE optimizer | Four GPUs, global batch 512, seed 20261105, AdamW with betas (0.9, 0.95), weight decay 0.05 and peak LR 0.0005. The first 100 epochs use 10 warmup epochs and cosine decay. Epochs 101–300 resume E100 and hold the last scheduled LR. |
| Point CLS seed | A shared 768D vector from the mean of the pre-Transformer group content tokens on 256 fixed, training-only instances. Use one canonical point view and omit positional embeddings from the mean. |
| Point CLS adaptation | Start from Point-MAE E300. Train the shared CLS and Point blocks 9–12 for 20 epochs on two views. VICReg uses per-view statistics and weights 25:25:1. Use a separate projector for each of blocks 9–12. Global batch 512, seed 20261118, encoder LR 1e-5 and CLS/projector LR 2e-4. |

Use [the Full326 Point-MAE config template](full326_point_mae.example.json)
with prepared inputs. Run [the distributed trainer](../../../fusion50/point/train_ddp.py)
with `mae.epochs` set to 100 for the first invocation. For continuation restore the
300-epoch target from the template,
`mae.schedule_epochs` to 100, and supply the E100 checkpoint with `--checkpoint`.
Use the same data split and four ranks to preserve the checkpoint's optimizer and
per-rank RNG state.

Copy [the Point CLS template](full326_point_cls.example.json) into a new run
directory and resolve its three asset paths. Then run from the repository root:

```sh
torchrun --standalone --nproc_per_node=4 fusion50/point/train_cls_vicreg.py \
  --config /path/to/point-cls-run/config.json
```

The selected artifact is `checkpoints/point_cls_epoch020.pt`. Fusion takes its
trained CLS and four Point-Transformer blocks. Phase 1's projector heads are
discarded. The cached points, seed vector and checkpoint files are external
inputs.

## Multimodal refinement

- Start Image-MAE from checkpoint 119. Each modality emits a native 768D CLS.
  Read blocks 9–12 and normalize each CLS before concatenating. Apply the
  pretrained LayerNorm at block 12 and functional LayerNorm at blocks 9–11.
- Use direct 1536D concatenation and a shared projection
  `Linear(1536,384) → GELU → Linear(384,384) → LayerNorm(384)`.
  PairBalance is off. Train the two CLS tokens, both encoders' blocks 9–12 and the
  projection. The earlier blocks stay frozen.
- Compute the loss on each of blocks 9–12 and average across the four losses.
  VICReg acts on the projection with weights **25:25:1**. Invariance compares
  the paired views of each instance. Across the global batch, compute variance
  and off-diagonal covariance separately per view, average the two variance
  penalties and sum the two covariance penalties.
- Add a negative-only BCE term with weight 2, cosine margin 0.86 and temperature
  0.20. For each anchor, exclude its own instance and the 20 nearest other
  instances, using the normalized two-view means to find neighbors. Add the
  original two normalized-CLS regularizers, each with variance weight 1.25 and
  covariance weight 0.05.
- Global batch 512 on four GPUs, 128 instances per rank, microbatch 4.
  AdamW uses betas (0.9, 0.95) and weight decay 1e-4. The peak learning rates are
  1e-5 for encoder blocks and 2e-5 for the CLS tokens and projection. The
  published configuration uses seed 20261118 and a one-epoch warmup.

Copy [the Fusion template](full326_refine.example.json) into a separate run
directory and resolve its paths. Supply the fixed `split.json` with the original
Point-MAE train and validation IDs plus the fixed `validation_batches`, and
supply `validation_probe_ids.json` for the validated embedding probe. For the
matched baseline, `max_train_crop_voxels` excludes training instances whose raw
EM crop arrays contain over 200,000,000 voxels. The mask foreground count is
a different quantity. A saved `exclude_train_indices` list can be supplied to
verify the exact selection. The reference run excluded eight instances and
performed 41 updates per epoch. The validation split is unaffected.

Start or resume with the same config and four ranks:

```sh
torchrun --standalone --nproc_per_node=4 exp/fusion50/native768/train_full326_refine_bce.py \
  --config /path/to/fusion-run/config.json --epochs 5

torchrun --standalone --nproc_per_node=4 exp/fusion50/native768/train_full326_refine_bce.py \
  --config /path/to/fusion-run/config.json --epochs 55 --resume
```

The second example assumes the checkpoint already contains the versioned
schedule extension to E55. The default initial horizon is E30. To extend a
run with an unfinished original schedule, add `--extend-schedule-to 55` to a
resume invocation. This stretches the remaining cosine decay continuously
from that checkpoint's epoch. Subsequent resumes read the extension from the
checkpoint. Optionally specify `--stop-at-unix` for a wall-time budget. The
trainer writes a full checkpoint each epoch and exits at a complete epoch
boundary. The **evaluation epoch** and the **latest resumable epoch** are
separate artifacts.

## Evaluation

Use [the primary-panel exporter](../../../fish2_downstream/export_primary_native.py)
with a prepared panel directory containing `inputs.json`, `pointclouds.npy`
and `instance_map.csv`. Add the target checkpoint path on the command line.
The exporter uses one canonical view without augmentation and emits the four
Block 12 readouts for the same record IDs. The [primary-panel evaluator](../../../fish2_downstream/evaluate_primary_panel.py)
provides the reusable `task_specs` and `evaluate_readout` functions. Use the
same body-disjoint five folds, LR settings and training-fold standardization for
each readout. Keep record-level and aggregated metrics alongside the checkpoint
revision. For comparable UMAPs use PCA30, cosine, 30 neighbors, min_dist 0.1, and
seed 20261116. Fit each projection using the same panel IDs and apply metadata
only as an overlay.
