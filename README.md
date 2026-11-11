# Mitochondria self-supervised pretraining and contrastive learning

Code-only research repository for point-based and voxel-based mitochondrial representation learning. It collects masked autoencoding and contrastive/self-supervised objectives implemented in the source snapshots. It does not include training volumes, instance annotations, caches, embeddings, checkpoints, logs, or exported results.

## Contents

| Directory | Method family | Main entry points |
|---|---|---|
| `point_mae/` | Attributed point-cloud masked modeling with optional two-view NT-Xent in the training objective. | `preprocess.py`, `train.py`, `train_deep_clustering.py`, `extract_embeddings.py` |
| `point_mae/configs/` | Small, medium, large, and VICReg/DEC experiment configurations. | `mitochondria_*.yaml` |
| `voxel_mae_baseline/mito_voxel_mae/` | Voxel masked-autoencoding baseline and preparation/export tools. | `prepare.py`, `train.py`, `export.py` |
| `voxel_mae_v5_perfopt/` | Optimized v5 source snapshot and profiling helpers. | See its README and model/training files. |
| `voxel_mae_v5_refiner/mito_refiner/` | Voxel-MAE feature refinement with projector/predictor heads, a queue-based nearest-neighbor objective, and evaluation tools. | `train.py`, `train_v5.py`, `evaluate.py` |

Point-MAE and voxel-MAE are separate research tracks collected in one repository for convenient version control. They do not share one input format or one universal training command.

## Point-MAE data conventions

The included Point-MAE README is the authoritative source for preprocessing, split rules, and configuration. Inputs use ZYX order. Coordinates use physical voxel spacing and a fixed global scale. The cache stores intensity, gradient, distance, and surface attributes; the YAML selects which features are used. The documented default point set has 1,024 surface plus 1,024 interior points per instance. When neuron metadata is supplied, train/validation/test splitting is grouped by neuron ID rather than by instance.

## Point-MAE quick start

Prepare a cache using paths and spacing for your dataset:

```bash
cd point_mae
python preprocess.py --instance /path/to/mitochondria_instances.zarr --raw /path/to/raw.ome.tif --out /path/to/mitochondria_cache.zarr --spacing-zyx 30 8 8 --neuron-map /path/to/instance_to_neuron.json
```

Train and export embeddings:

```bash
python train.py --config configs/mitochondria_medium.yaml --cache /path/to/mitochondria_cache.zarr --output /path/to/runs/mitochondria_medium
python extract_embeddings.py --config configs/mitochondria_medium.yaml --cache /path/to/mitochondria_cache.zarr --checkpoint /path/to/runs/mitochondria_medium/best.pt --out /path/to/mitochondria_embeddings.npz
```

`train.py` combines masked reconstruction with the configured two-view NT-Xent loss when the contrastive weight is nonzero. `train_deep_clustering.py` contains separate MAE/VICReg/DEC mode experiments. Read the selected YAML and script before comparing runs; these objectives and modes are not interchangeable.

## Voxel-MAE and refinement

The voxel baseline and v5 refiner have their own READMEs and model documentation. The refiner uses learned heads and a nearest-neighbor/queue-style self-supervised objective over voxel-MAE features; some paths use virtual batches or replay to manage memory. Consult the bundled docs for exact training stages, batch semantics, checkpoint format, and supported interfaces. A v5-refiner checkpoint is not assumed to load in the baseline trainer.

## Environment and reproducibility

There is no unified, pinned requirements file for all method families. Use separate environments if their PyTorch/CUDA or data-stack requirements differ. Check each subproject's imports, launcher, and README; record compatible library versions, the resolved config, source revision, environment, and split manifest with each new run.

Recommended workflow:

1. Prepare a cache outside the Git checkout and verify sampled instances.
2. Confirm that train/validation/test groups do not leak across the intended split.
3. Run the supplied smoke or correctness check for the selected model.
4. Train to a new output directory and never overwrite a historical run.
5. Save the resolved config, source revision, environment, and split manifest beside the checkpoint.

Smoke scripts and tests are included, but were not run while preparing these repositories.

## Large files and GitHub

`.gitignore` excludes common data, checkpoint, cache, and run-output files. Keep raw volumes, annotation masks, embeddings, model weights, logs, and credentials in approved storage rather than Git. Review any new file type before committing.

This repository is initialized on the 113 server on branch `main`. Source files are staged for the initial commit. No Git remote is configured and nothing has been pushed. Set your Git author identity, review `git status`, make the initial commit, add your GitHub remote, and push when you choose.