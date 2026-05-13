#!/usr/bin/env bash
# Render + metrics + plot for 0513 setup2 (weight mode sweep, progressive packing).
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0513_setup2_progressive}" \
PLOT_GROUP_BY="weight_mode" \
TITLE_SUFFIX="8x8x8 progressive — weight mode sweep" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
bash "$ROOT/dlapisgs-utility/experiments/run_render_metrics.sh"
