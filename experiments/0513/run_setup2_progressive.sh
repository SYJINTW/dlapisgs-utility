#!/usr/bin/env bash
# Setup 2: per-Gaussian progressive packing with three weight modes.
# Also runs tile_strict and tile_partial for comparison under the same weight.
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
PLY="${PLY:-$ROOT/exp-dataset/bicycle/point_cloud.ply}"
TRACE="${TRACE:-$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json}"
OUT_BASE="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0513_setup2_progressive}"
CONDA_ENV="${CONDA_ENV:-gsquic}"

BUDGET_LIST="${BUDGET_LIST:-20 60 100 200 500}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
SCHEMES=("vd_lod_w_c")

# Each cell is "<packing_mode> <weight_mode>".
COMBOS=(
    "progressive volume"
    "progressive volume_over_d2"
    "progressive screen_area"
    "tile_strict  volume_over_d2"
    "tile_partial volume_over_d2"
)

echo "=========================================="
echo "Setup 2 — progressive / tile_strict (0513)"
echo "=========================================="
echo "Output root  : $OUT_BASE"
echo "Grid shape   : $GRID_SHAPE"
echo "Budgets MB   : $BUDGET_LIST"
echo "Schemes      : ${SCHEMES[*]}"
echo "Combos       : ${#COMBOS[@]} (packing_mode, weight_mode)"
echo "=========================================="

mkdir -p "$OUT_BASE"

for combo in "${COMBOS[@]}"; do
    read -r PACKING_MODE WEIGHT_MODE <<< "$combo"
    TAG="${PACKING_MODE}__${WEIGHT_MODE}"
    OUT_DIR="$OUT_BASE/$TAG"
    echo ""
    echo "--> $TAG  ->  $OUT_DIR"
    mkdir -p "$OUT_DIR"

    conda run -n "$CONDA_ENV" python "$ROOT/dlapisgs-utility/test_utility.py" \
        --ply "$PLY" \
        --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" \
        --grid-shape $GRID_SHAPE \
        --budgets-mb $BUDGET_LIST \
        --schemes "${SCHEMES[@]}" \
        --num-lod "$NUM_LOD" \
        --camera-index "$CAMERA_INDEX" \
        --packing-mode "$PACKING_MODE" \
        --weight-mode "$WEIGHT_MODE" \
        --w-norm max \
        --c-norm max
done

echo ""
echo "Done. Per-combo outputs under: $OUT_BASE/<packing>__<weight>/"
