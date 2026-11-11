#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' '当前v5 Refiner等待用户通知后再训练。命令模板见 ../REFINER_ADDON_ZH.md。' \
  '仅检查接口：python -m mito_voxel_mae_v5_refiner_r0.mito_refiner.tests.test_v5_integration --interfaces-only'
