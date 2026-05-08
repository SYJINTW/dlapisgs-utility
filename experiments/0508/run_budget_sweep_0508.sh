#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
PLY="$ROOT/exp-dataset/bicycle/point_cloud.ply"
TRACE="$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json"
OUT_DIR="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0508_budget_sweep}"
CONDA_ENV="${CONDA_ENV:-gsquic}"

BUDGET_LIST="${BUDGET_LIST:-10 20 40 60 80 100 200 340 500 700}"
SCHEMES=("vd_lod" "vd_lod_w" "vd_lod_c" "vd_lod_w_c")
NUM_LOD="${NUM_LOD:-1}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"

echo ""
echo "=========================================="
echo "Budget Sweep 0508 (8x8x8 grid)"
echo "=========================================="
echo "Output root: $OUT_DIR"
echo "Grid shape : $GRID_SHAPE"
echo "Budgets MB : $BUDGET_LIST"
echo "Schemes    : ${SCHEMES[*]}"
echo "Camera mode: $CAMERA_INDEX"
echo "=========================================="
echo ""

mkdir -p "$OUT_DIR"

# Single invocation: all budgets and schemes share PLY load, tiling, visibility, and utility
# computation. Per-scheme greedy order is computed once; budgets are free prefix slices.
conda run -n "$CONDA_ENV" python "$ROOT/dlapisgs-utility/test_utility.py" \
    --ply "$PLY" \
    --output-root "$OUT_DIR" \
    --camera-trace "$TRACE" \
    --grid-shape $GRID_SHAPE \
    --budgets-mb $BUDGET_LIST \
    --schemes "${SCHEMES[@]}" \
    --num-lod "$NUM_LOD" \
    --camera-index "$CAMERA_INDEX"

echo "Done. Selection outputs at: $OUT_DIR"
