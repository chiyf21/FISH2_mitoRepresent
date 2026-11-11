# Mitochondrial representation probes

This experiment reads the actual 2026-11-05 filled-mask cohort and runtime model without modifying it. All raw inputs, weights, arrays, predictions and regenerable results stay outside Git. Runtime source fingerprints, checkpoint identity, stable keys, body splits, preprocessing, tuning traces and per-instance held-out predictions are saved in the chosen output directory.

Use the existing server environment, not a new dependency installation:

```sh
cd /home/cyf/github_repos/mitochondria-pretraining-contrastive_worktrees/representation_probe_20261111
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
# Choose a NEW output directory. Never point this at another experiment.
OUT=/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/point-MAE/runs/YOUR_UNIQUE_PROBE_DIRECTORY
mkdir -m 700 "$OUT"
"$PY" representation_probe/prepare.py --output "$OUT"
"$PY" representation_probe/morphometry.py --output "$OUT" --limit 8
"$PY" representation_probe/evaluate.py --output "$OUT" --smoke
"$PY" representation_probe/evaluate.py --output "$OUT" --phase phase1
# Inspect GPU availability and memory before choosing ONE unused GPU UUID.
export CUDA_VISIBLE_DEVICES=GPU_UUID_YOU_CHECKED
"$PY" representation_probe/extract.py --output "$OUT" --stage fusion50 --gpu "$CUDA_VISIBLE_DEVICES" --smoke
"$PY" representation_probe/extract.py --output "$OUT" --stage fusion50 --gpu "$CUDA_VISIBLE_DEVICES"
"$PY" representation_probe/extract.py --output "$OUT" --stage pointmae100 --gpu "$CUDA_VISIBLE_DEVICES"
"$PY" representation_probe/extract.py --output "$OUT" --stage nna20 --gpu "$CUDA_VISIBLE_DEVICES"
"$PY" representation_probe/extract.py --output "$OUT" --stage em_base --gpu "$CUDA_VISIBLE_DEVICES"
"$PY" representation_probe/morphometry.py --output "$OUT"
"$PY" representation_probe/evaluate.py --output "$OUT" --phase final
```

prepare.py verifies the dashboard feature-key derivation, hard-linked checkpoint, reference arrays, finite shapes, full cohort/point-cache stable identities and balanced concatenation. It does not assume global instance IDs are unique across runs/bodies. The 808 unchanged masks intentionally retain their original paths, audited as zero added voxels.

extract.py invokes existing core.prepare and System forwards. Native forward hooks pool block outputs without replacing block computations. A TorchFunctionMode observes the functional patch projection because the runtime bypasses its Conv3d module. It returns the original operator output unchanged. Only pooled arrays are saved. EM means use the runtime valid mask, excluding CLS and padding. Point means use 64 actual groups. EM L12 raw and final-norm reads are distinct. Point-MAE100 has no synthetic CLS: its native TransformerEncoder wrapper receives all unmasked group tokens. The NNA/Fusion Point context includes a CLS, a remaining stage-comparison caveat.

morphometry.py incrementally computes voxel volume, physical voxel-center PCA elongation and closed Lewiner-mesh area at ZYX=(30,8,8) nm. It uses a 0.5 isosurface, step1, one exterior-zero voxel pad and removes degenerate triangles. Surface area is not exposed voxel-face counting. The padded surface describes the observed segmentation; physical truncation beyond the crop cannot be recovered. Sphericity and surface/volume are derived from volume/area and are not independent discoveries.

evaluate.py freezes feature arrays and uses L2 Logistic/Ridge, train-only scaling, fixed validation tuning and shared body-held-out folds. Cell type uses six existing labels with >=5 body support; in this cohort the minimum is seven. Its outer five folds and inner holdouts are stratified at body level; training weights give equal total influence to each body and balance class-body frequency. Compartment uses balanced class weights. Neither supervised probe nor scaler is refitted on validation/test. Test-body bootstrap describes conditional uncertainty, not encoder retraining uncertainty.

Labels need caution: the dashboard compartments are graph/synapse-derived weak labels, not the manually approved skeleton labels. Six cohort bodies have a manual-skeleton coverage index but the referenced 112 files are unavailable on 113. Existing compartment code uses different physical-scale assumptions from mask morphometry, which is recorded rather than silently corrected. Cell types are propagated body metadata, not independently relabelled here. All EM base-pretraining validation/test-body exposure remains a limit on whole-system generalization claims.

The first float32 Ridge diagnostic and a stringent Logistic solver attempt are preserved only in the experiment directory. Formal probes use float64 and demand convergence. Hyperparameter candidates were retained; only solver numerical precision/budget changed, before the formal first table.

All shared training code, dashboard assets, model files, masks and primary main worktree remain read-only. New output directories are required. Feature chunks and morphology records can resume inside their own directory after matching provenance; do not combine chunks from different configurations.
