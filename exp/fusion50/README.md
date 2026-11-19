# Fusion50 model variants

| Model | Entrypoints | Outputs |
| --- | --- | --- |
| Original Fusion50 (384D Point CLS + 768D Image CLS) | [Trainer](../../fusion50/refine/train.py), [exporter](../../fusion50/refine/export.py) | 384D fused embedding |
| Native 768D CLS Fusion50 | [Trainer](native768/train.py), [exporter](native768/export.py) | 768D Point and Image CLS, 1536D concat and 384D projected embedding |

For the Full326 per-view VICReg training recipe, see [Full326 native fusion](native768/FULL326.md).

The two trainers have different checkpoint layouts. Use the exporter that matches the
checkpoint's model. Both variants share the Point-MAE and image MAE implementations.

For native 768D, start from [the config template](native768/config.example.json).
Set the input paths to the native 768D Point-MAE, its prepared data and the image
MAE checkpoint. For Point-only CLS training, keep `fusion_epochs: 0`. To start the
next stage from a completed Point CLS checkpoint, set `fusion_epochs` and pass
`--init-point-cls`. Use `--resume` with the same run directory, schedule and world
size to resume. PairBalance is optional and defaults to off.

The native model code comes from the training revisions `0291cd7` (no Balance)
and `bfc3102` (PairBalance). Checkpoint files and evaluation assets are managed
outside Git.
