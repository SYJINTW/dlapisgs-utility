#!/usr/bin/env bash
# Setup 1: tile-level selection with W_k / C_k normalization sweep.
# Scheme fixed to vd_lod_w_c (full model) so we can see the W and C levers act.
# weight_mode=volume_over_d2 (sigmoid(o)*vol/d^2); packing=tile_strict for clean tile attribution.
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
PLY="${PLY:-$ROOT/exp-dataset/bicycle/point_cloud.ply}"
TRACE="${TRACE:-$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json}"
OUT_BASE="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0513_setup1_norm_sweep}"
CONDA_ENV="${CONDA_ENV:-gsquic}"

BUDGET_LIST="${BUDGET_LIST:-20 60 100 200 500}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
SCHEMES=("vd_lod_w_c")
PACKING_MODE="${PACKING_MODE:-tile_strict}"
WEIGHT_MODE="${WEIGHT_MODE:-volume_over_d2}"

# Symmetric pairs only: both terms in [0,1] so neither dominates by scale.
# none/max = 0508 baseline (W unbounded); none/none = fully unbounded reference.
PAIRS=(
    "none max"      # 0508 baseline: W unbounded, C in [0,1]
    "none none"     # unbounded reference
    "max max"       # [0,1] / [0,1]
    "minmax minmax" # [0,1] / [0,1], zero-shifted
    "log1p log1p"   # [0,1] / [0,1], log-compressed (fixed: log1p(x)/log1p(max))
    "sum sum"       # probability mass per tile, sums to 1
)

echo "=========================================="
echo "Setup 1 — W/C normalization sweep (0513)"
echo "=========================================="
echo "Output root  : $OUT_BASE"
echo "Grid shape   : $GRID_SHAPE"
echo "Budgets MB   : $BUDGET_LIST"
echo "Schemes      : ${SCHEMES[*]}"
echo "Packing mode : $PACKING_MODE"
echo "Weight mode  : $WEIGHT_MODE"
echo "Pairs        : ${#PAIRS[@]} (w_norm, c_norm) combos"
echo "=========================================="

mkdir -p "$OUT_BASE"
TILING_CACHE="$OUT_BASE/shared_tiling_cache.npz"

for pair in "${PAIRS[@]}"; do
    read -r W_NORM C_NORM <<< "$pair"
    TAG="w-${W_NORM}_c-${C_NORM}"
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
        --tiling-cache "$TILING_CACHE" \
        --w-norm "$W_NORM" \
        --c-norm "$C_NORM"
done

echo ""
echo "Done. Per-combo outputs under: $OUT_BASE/<w-...>_<c-...>/"
