#!/usr/bin/env bash
set -eu
cd /path/to/JaneliaEM/Point-MAE
OUT=/mnt/data21T_3/cyf/EM/mito_voxel_mae_prepared_v1
mkdir -p "$OUT"
if [[ -f "$OUT/preparation.pid" ]] && kill -0 "$(cat "$OUT/preparation.pid")" 2>/dev/null; then
  echo "Preparation already running"; exit 1
fi
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/cyf/miniconda3/envs/tacco_env/bin/python -u -m mito_voxel_mae.prepare \
  --dataset /mnt/data21T_3/cyf/EM/neuron_mito_dataset --raw-root /mnt/data21T_3/cyf/EM/neuron \
  --out "$OUT" --workers 6 > "$OUT/preparation.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/preparation.pid"
cat "$OUT/preparation.pid"
