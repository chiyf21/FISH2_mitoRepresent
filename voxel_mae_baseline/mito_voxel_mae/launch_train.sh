#!/usr/bin/env bash
set -eu
cd /path/to/JaneliaEM/point-MAE
CACHE=/path/to/project_data/EM/mito_voxel_mae_training_v1
OUT=/path/to/project_data/EM/point-MAE/runs/voxel_mae_native_v1_20261014
mkdir -p "$OUT"
if [[ -f "$OUT/train.pid" ]] && kill -0 "$(cat "$OUT/train.pid")" 2>/dev/null; then
  echo "Training already running"; exit 1
fi
if [[ -f "$OUT/last.pt" ]]; then echo "Existing run requires an explicit resume"; exit 1; fi
nohup env CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/cyf/miniconda3/envs/cellmap_organelle/bin/python -u -m mito_voxel_mae.train \
  --cache "$CACHE" --out "$OUT" --epochs 10 --queries 8192 --views 4 --accumulate 8 \
  --workers 4 --dim 256 --depth 6 --latent 256 --channels 24 --val-limit 0 \
  > "$OUT/train.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/train.pid"
cat "$OUT/train.pid"
