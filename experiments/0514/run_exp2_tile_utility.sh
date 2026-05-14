#!/usr/bin/env bash
# Experiment 2 (0514): Tile utility ablation on bicycle, tile_strict packing.
# Two sub-sweeps so the conclusion is robust to GS-weight choice:
#   sub-A: weight-mode = volume_over_d2
#   sub-B: weight-mode = screen_area
# Each runs the 4 schemes {vd_lod, vd_lod_w, vd_lod_c, vd_lod_w_c} across 7 budgets.
#
# Env overrides:
#   BUDGET_PCTS="10 25 40 55 70 85 100"
#   WEIGHT_MODES="volume_over_d2 screen_area"   # the two sub-sweeps
#   GRID_SHAPE="8 8 8"
#   CAMERA_INDEX=-1
#   CUDA_VISIBLE_DEVICES=3
#   DRY_RUN=1, SKIP_RENDER=1
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"

CONDA_ENV="${CONDA_ENV:-gsquic}"
RENDER_ENV="${RENDER_ENV:-gaussian_splatting}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

SCENE="${SCENE:-bicycle}"
PLY="${PLY:-$ROOT/exp-dataset/bicycle/point_cloud.ply}"
FULL_MB="${FULL_MB:-1418.8}"
TRACE="${TRACE:-$(dirname "$PLY")/sparse_views_100.json}"

BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 100}"
WEIGHT_MODES="${WEIGHT_MODES:-volume_over_d2 screen_area}"
SCHEMES_LIST="${SCHEMES_LIST:-vd_lod vd_lod_w vd_lod_c vd_lod_w_c}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0514/exp2_tile_utility}"

DRY_RUN_FLAG=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_FLAG="--dry-run"

# Convert percentages to absolute MB
BUDGETS=""
for p in $BUDGET_PCTS; do
    mb=$(python -c "print(f'{$FULL_MB * $p / 100:.1f}')")
    BUDGETS="$BUDGETS $mb"
done

echo "=========================================="
echo "0514 Exp 2 — Tile utility sweep (bicycle, tile_strict)"
echo "=========================================="
echo "OUTPUT_ROOT   : $OUT_BASE"
echo "SCENE         : $SCENE"
echo "PLY           : $PLY"
echo "TRACE         : $TRACE"
echo "BUDGETS_MB    :$BUDGETS"
echo "SCHEMES       : $SCHEMES_LIST"
echo "WEIGHT_MODES  : $WEIGHT_MODES"
echo "GRID_SHAPE    : $GRID_SHAPE"
echo "CAMERA_INDEX  : $CAMERA_INDEX"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DRY_RUN       : ${DRY_RUN:-0}"
echo "=========================================="

if [[ ! -f "$PLY" ]]; then echo "[fatal] PLY not found: $PLY"; exit 1; fi
if [[ ! -f "$TRACE" ]]; then echo "[fatal] trace not found: $TRACE  (run gen_sparse_views.py first)"; exit 1; fi

mkdir -p "$OUT_BASE"
TILING_CACHE="$OUT_BASE/.tiling_cache.npz"

for wm in $WEIGHT_MODES; do
    TAG="$wm"
    OUT_DIR="$OUT_BASE/$TAG"
    echo ""
    echo "---- weight_mode=$wm  schemes=[$SCHEMES_LIST]  budgets=[$BUDGETS] ----"
    mkdir -p "$OUT_DIR"

    conda run -n "$CONDA_ENV" python "$UTIL_DIR/test_utility.py" \
        --ply "$PLY" \
        --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" \
        --grid-shape $GRID_SHAPE \
        --budgets-mb $BUDGETS \
        --schemes $SCHEMES_LIST \
        --num-lod "$NUM_LOD" \
        --camera-index "$CAMERA_INDEX" \
        --packing-mode tile_strict \
        --weight-mode "$wm" \
        --w-norm sum \
        --c-norm sum \
        --tiling-cache "$TILING_CACHE" \
        $DRY_RUN_FLAG

    if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_RENDER:-0}" == "1" ]]; then continue; fi

    conda run -n "$RENDER_ENV" python "$UTIL_DIR/experiments/render_metrics.py" \
        --output-root "$OUT_DIR" \
        --gt-ply "$PLY" \
        --trace "$TRACE" \
        --scene "$SCENE" \
        --render-dir "$OUT_DIR/renders"

    conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/aggregate_timings.py" \
        --output-root "$OUT_DIR" || true

    # Quick-look plot grouped by scheme
    conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/plot_metrics.py" \
        --summary-csv "$OUT_DIR/metrics/summary.csv" \
        --out-dir "$OUT_DIR/plots" \
        --group-by scheme \
        --title-suffix "$SCENE (Exp2 tile_strict, w=$wm)"

    conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/0514/pick_representative_views.py" \
        --summary-csv "$OUT_DIR/metrics/summary.csv" \
        --output-root "$OUT_DIR" \
        --group-by scheme || true
done

if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_RENDER:-0}" == "1" ]]; then
    echo ""; echo "Dry-run / skip-render mode — done."; exit 0
fi

# Concatenate both sub-sweeps for cross-weight comparison
SWEEP_CSV="$OUT_BASE/summary_all.csv"
conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

echo ""
echo "Done. Sweep root: $OUT_BASE"
