# Scale-aware mitochondrial voxel MAE

This independent module extends the old Point-MAE design; historical source and checkpoints remain usable.

- Native-resolution EM: ZYX, 30/8/8nm. Initial cohort keeps context crops <=1.5M voxels; excluded large crops are explicitly logged for future tiled training.
- Source raw blocks recover real context, with an explicit valid-EM volume. Missing values never supervise reconstruction.
- Reference intensity algorithm: source p0.5/p99.8 clip to [0,1], then valid tight-crop z-score. Source percentile estimates use a deterministic capped block sample, recorded in statistics/*.json; this is not the historical volume's exact percentile constants.
- Tight bbox remains the scale reference; a separate native context crop adds two 17-grid steps plus 2 voxels per axis.
- 17^3 candidate grid + deduplicated 26 grid neighbors, with one-cell support halo. Variable support count is masked; invalid and hidden support points do not enter grouping/pooling.
- Two coordinates: t for per-axis normalized layout, r for shape-preserving geometry; log physical ZYX size in micrometers enters encoder.
- Raw spatial masking occurs BEFORE a trainable anisotropic U-Net. No raw skip reaches the decoder. Fixed 4^3 mask blocks hide 38/64 regions.
- Final 256D latent directly supervises both decoders. Voxel decoder predicts occupancy and normalized intensity at target-independent uniformly drawn queries; size decoder predicts log physical lengths.
- Four online views per train instance/epoch: noise, mild affine/shear, physical XY rotation/translation, smooth nonlinear warp. EM, occupancy and validity warp together. Augmented tight bbox/size is recomputed. Truncating/invalid augmentations fall back to the clean geometry; noise is input-only. This does not increase the number of independent biological instances.
- Neuron/body train/val/test splits are disjoint and reproducible; validation queries and masking are fixed.
- Varying native crops use batch1 and gradient accumulation8. Larger-context tiled support is a future extension; this run does not silently resize them.
- Training: 100 epochs, 8192 queries/instance, AdamW, BF16, 6-layer 256D Transformer. Formal launch validates all held-out validation instances; test remains untouched by training.
- `python -m mito_voxel_mae.checks --cache CACHE --out CHECKS`: coordinate/augmentation/padding/gradient/chunk/Z1 checks and real fixed-view overfit.
- `python -m mito_voxel_mae.export --cache CACHE --checkpoint RUN/best.pt --out EXPORT --index 0`: encode once, chunk-query all original voxel centers.
- `--embeddings-only`: export unmasked per-instance latents and body/record IDs.

Input occupancy is the segmented outer-instance volume, not cristae membrane occupancy. The voxel targets may preserve cristae texture through intensity. Low-dimensional latents do not guarantee lossless native EM reconstruction.
