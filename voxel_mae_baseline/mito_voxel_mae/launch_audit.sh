#!/usr/bin/env bash
set -eu
cd /path/to/JaneliaEM/point-MAE
BASE=/path/to/project_data/EM
OUT=$BASE/mito_voxel_mae_training_v1
mkdir -p "$OUT"
if [[ -f "$OUT/manifest.json" ]]; then echo "Manifest already finalized"; exit 1; fi
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/cyf/miniconda3/envs/cellmap_organelle/bin/python -u -m mito_voxel_mae.merge_prepared \
  --inputs "$BASE/mito_voxel_mae_prepared_v1" --original "$BASE/neuron_mito_dataset" \
  --out "$OUT" > "$OUT/audit.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/audit.pid"
cat "$OUT/audit.pid"
