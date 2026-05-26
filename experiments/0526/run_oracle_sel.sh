#!/usr/bin/env bash
# Oracle-based tile selection ceiling experiment.
# Runs oracle_loo / oracle_aoi / oracle_combined schemes for hotdog + ship,
# using pre-computed oracle_dq.npz scores from 0522/exp4_oracle_dq.
# Follows exp2/wmode-mean structure: selection → render → plot → concat.
#
# Env overrides:
#   SCENES="hotdog ship"
#   BUDGET_PCTS="10 25 40 55 70 85 99 100"
#   SCHEMES="oracle_loo oracle_aoi oracle_combined"
#   CUDA_VISIBLE_DEVICES=0
#   DRY_RUN=1, SKIP_RENDER=1, KEEP_PLY=1
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"

CONDA_ENV="${CONDA_ENV:-gsquic}"
RENDER_ENV="${RENDER_ENV:-gaussian_splatting}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENES="${SCENES:-hotdog ship}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
SCHEMES="${SCHEMES:-oracle_loo oracle_aoi oracle_combined}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
ORACLE_ROOT="$UTIL_DIR/output/0522/exp4_oracle_dq"
TILING_ROOT="$UTIL_DIR/output/0514/exp1_gs_weight_done"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0526/exp_oracle_sel}"

DRY_RUN_FLAG=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_FLAG="--dry-run"

DELETE_PLY_FLAG="--delete-ply"
[[ "${KEEP_PLY:-0}" == "1" ]] && DELETE_PLY_FLAG=""

scene_ply() {
    case "$1" in
        hotdog)    echo "$ROOT/exp-dataset/hotdog/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        ship)      echo "$ROOT/exp-dataset/ship/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        bicycle)   echo "$ROOT/exp-dataset/bicycle/point_cloud.ply" ;;
        chair)     echo "$ROOT/exp-dataset/chair/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        drums)     echo "$ROOT/exp-dataset/drums/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        ficus)     echo "$ROOT/exp-dataset/ficus/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        materials) echo "$ROOT/exp-dataset/materials/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        mic)       echo "$ROOT/exp-dataset/mic/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        *) echo "" ;;
    esac
}
scene_trace() {
    local ply; ply="$(scene_ply "$1")"
    echo "$(dirname "$ply")/sparse_views_100.json"
}

echo "=========================================="
echo "Oracle selection ceiling experiment"
echo "OUT_BASE : $OUT_BASE"
echo "SCENES   : $SCENES"
echo "SCHEMES  : $SCHEMES"
echo "BUDGETS  : $BUDGET_PCTS"
echo "GPU      : $CUDA_VISIBLE_DEVICES"
echo "DRY_RUN  : ${DRY_RUN:-0}"
echo "=========================================="

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$(scene_ply "$scene")"
    TRACE="$(scene_trace "$scene")"
    ORACLE_NPZ="$ORACLE_ROOT/$scene/oracle_dq.npz"
    TILING_CACHE="$TILING_ROOT/$scene/.tiling_cache.npz"
    OUT_DIR="$OUT_BASE/$scene"

    if [[ -z "$PLY" || ! -f "$PLY" ]]; then
        echo "[skip] $scene: PLY not found at $PLY"; continue
    fi
    if [[ ! -f "$TRACE" ]]; then
        echo "[skip] $scene: trace not found at $TRACE"; continue
    fi
    if [[ ! -f "$ORACLE_NPZ" ]]; then
        echo "[skip] $scene: oracle_dq.npz not found at $ORACLE_NPZ"; continue
    fi

    echo ""
    echo "---- [$scene] ----"
    echo "     OUT_DIR=$OUT_DIR"
    mkdir -p "$OUT_DIR"

    conda run -n "$CONDA_ENV" python "$UTIL_DIR/test_utility.py" \
        --ply "$PLY" \
        --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" \
        --grid-shape $GRID_SHAPE \
        --budget-pct $BUDGET_PCTS \
        --schemes $SCHEMES \
        --num-lod "$NUM_LOD" \
        --camera-index "$CAMERA_INDEX" \
        --packing-mode tile_strict \
        --weight-mode screen_area \
        --w-norm sum --c-norm sum \
        --tiling-cache "$TILING_CACHE" \
        --oracle-npz "$ORACLE_NPZ" \
        $DRY_RUN_FLAG

    if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_RENDER:-0}" == "1" ]]; then continue; fi

    conda run -n "$RENDER_ENV" python "$UTIL_DIR/experiments/render_metrics.py" \
        --output-root "$OUT_DIR" \
        --gt-ply "$PLY" \
        --trace "$TRACE" \
        --scene "$scene" \
        --render-dir "$OUT_DIR/renders" \
        $DELETE_PLY_FLAG

    conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/aggregate_timings.py" \
        --output-root "$OUT_DIR" || true

    conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/plot_metrics.py" \
        --summary-csv "$OUT_DIR/metrics/summary.csv" \
        --out-dir "$OUT_DIR/plots" \
        --group-by scheme \
        --title-suffix "$scene (oracle selection ceiling, tile_strict)"

    conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/0514/pick_representative_views.py" \
        --summary-csv "$OUT_DIR/metrics/summary.csv" \
        --output-root "$OUT_DIR" \
        --group-by scheme || true
done

if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_RENDER:-0}" == "1" ]]; then
    echo ""; echo "Dry-run / skip-render mode — done."; exit 0
fi

SWEEP_CSV="$OUT_BASE/summary_all.csv"
conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

for scene in $SCENES; do
    SCENE_CSV="$OUT_BASE/$scene/summary_all.csv"
    conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/concat_summaries.py" \
        --glob "$OUT_BASE/$scene/metrics/summary.csv" \
        --out "$SCENE_CSV" || true
done

echo ""
echo "Done. Sweep root: $OUT_BASE"
