#!/usr/bin/env bash
# Experiment 1 (0514): GS weight design sweep.
# Fixed:   packing-mode=progressive, scheme=vd_lod_w_c, w_norm=sum, c_norm=sum.
# Swept:   weight-mode ∈ {volume, volume_over_d2, screen_area}.
# Scenes:  bicycle, hotdog, ship.  Budgets: 10/25/40/55/70/85/100 % of each PLY size.
#
# Env overrides (all optional):
#   SCENES="bicycle hotdog ship"
#   BUDGET_PCTS="10 25 40 55 70 85 100"
#   WEIGHT_MODES="volume volume_over_d2 screen_area"
#   GRID_SHAPE="8 8 8"
#   CAMERA_INDEX=-1                    # -1 = all cameras
#   CUDA_VISIBLE_DEVICES=2
#   OUTPUT_ROOT=.../output/0514/exp1_gs_weight
#   DRY_RUN=1                          # passes --dry-run to test_utility.py
#   SKIP_RENDER=1                      # skip render_metrics + downstream
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTIL_DIR="$ROOT/dlapisgs-utility"

CONDA_ENV="${CONDA_ENV:-gsquic}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SCENES="${SCENES:-bicycle hotdog ship}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 100}"
WEIGHT_MODES="${WEIGHT_MODES:-volume volume_over_d2 screen_area}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
SCHEME="${SCHEME:-vd_lod_w_c}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0514/exp1_gs_weight}"

DRY_RUN_FLAG=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_FLAG="--dry-run"

# Per-scene config: (PLY path, full-size MB, scene-type for view generator).
scene_ply() {
    case "$1" in
        bicycle) echo "$ROOT/exp-dataset/bicycle/point_cloud.ply" ;;
        hotdog)  echo "$ROOT/exp-dataset/hotdog/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        ship)    echo "$ROOT/exp-dataset/ship/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        *) echo "" ;;
    esac
}
scene_full_mb() {
    case "$1" in
        bicycle) echo 1418.8 ;;
        hotdog)  echo 35.2 ;;
        ship)    echo 76.9 ;;
        *) echo 0 ;;
    esac
}
scene_trace() {
    local ply; ply="$(scene_ply "$1")"
    echo "$(dirname "$ply")/sparse_views_100.json"
}

budget_list_for_scene() {
    local full="$1"
    local pcts="$2"
    local out=""
    for p in $pcts; do
        # bash math is integer only — use python for the multiply
        local mb
        mb=$(python -c "print(f'{$full * $p / 100:.1f}')")
        out="$out $mb"
    done
    echo "$out"
}

echo "=========================================="
echo "0514 Exp 1 — GS weight sweep (progressive packing)"
echo "=========================================="
echo "OUTPUT_ROOT  : $OUT_BASE"
echo "SCENES       : $SCENES"
echo "BUDGET_PCTS  : $BUDGET_PCTS"
echo "WEIGHT_MODES : $WEIGHT_MODES"
echo "GRID_SHAPE   : $GRID_SHAPE"
echo "CAMERA_INDEX : $CAMERA_INDEX"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DRY_RUN      : ${DRY_RUN:-0}"
echo "=========================================="

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$(scene_ply "$scene")"
    FULL_MB="$(scene_full_mb "$scene")"
    TRACE="$(scene_trace "$scene")"

    if [[ -z "$PLY" || ! -f "$PLY" ]]; then
        echo "[skip] $scene: PLY not found at $PLY"; continue
    fi
    if [[ ! -f "$TRACE" ]]; then
        echo "[skip] $scene: trace not found at $TRACE  (run gen_sparse_views.py first)"; continue
    fi

    BUDGETS="$(budget_list_for_scene "$FULL_MB" "$BUDGET_PCTS")"
    TILING_CACHE="$OUT_BASE/$scene/.tiling_cache.npz"
    mkdir -p "$(dirname "$TILING_CACHE")"

    for wm in $WEIGHT_MODES; do
        TAG="$wm"
        OUT_DIR="$OUT_BASE/$scene/$TAG"
        echo ""
        echo "---- [$scene] weight_mode=$wm  budgets=[$BUDGETS] ----"
        echo "     OUT_DIR=$OUT_DIR"
        mkdir -p "$OUT_DIR"

        conda run -n "$CONDA_ENV" python "$UTIL_DIR/test_utility.py" \
            --ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape $GRID_SHAPE \
            --budgets-mb $BUDGETS \
            --schemes "$SCHEME" \
            --num-lod "$NUM_LOD" \
            --camera-index "$CAMERA_INDEX" \
            --packing-mode progressive \
            --weight-mode "$wm" \
            --w-norm sum \
            --c-norm sum \
            --tiling-cache "$TILING_CACHE" \
            $DRY_RUN_FLAG

        if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_RENDER:-0}" == "1" ]]; then
            continue
        fi

        # Render + metrics for this (scene, weight_mode) cell
        conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/render_metrics.py" \
            --output-root "$OUT_DIR" \
            --gt-ply "$PLY" \
            --trace "$TRACE" \
            --scene "$scene" \
            --render-dir "$OUT_DIR/renders"

        # Timing aggregate
        conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/aggregate_timings.py" \
            --output-root "$OUT_DIR" || true
    done
done

if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_RENDER:-0}" == "1" ]]; then
    echo ""; echo "Dry-run / skip-render mode — done."; exit 0
fi

# Concatenate all per-cell summary.csv files into one sweep-level CSV
SWEEP_CSV="$OUT_BASE/summary_all.csv"
echo ""
echo "Concatenating per-cell summary.csv -> $SWEEP_CSV"
conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

# Quick-look plots grouped by weight_mode (one figure pair per scene)
for scene in $SCENES; do
    SCENE_CSV="$OUT_BASE/$scene/summary_all.csv"
    conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/concat_summaries.py" \
        --glob "$OUT_BASE/$scene/*/metrics/summary.csv" \
        --out "$SCENE_CSV"
    if [[ -f "$SCENE_CSV" ]]; then
        conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/plot_metrics.py" \
            --summary-csv "$SCENE_CSV" \
            --out-dir "$OUT_BASE/$scene/plots" \
            --group-by weight_mode \
            --title-suffix "$scene (Exp1 progressive)"
    fi

    # Representative views
    if [[ -f "$SCENE_CSV" ]]; then
        for wm in $WEIGHT_MODES; do
            CELL_ROOT="$OUT_BASE/$scene/$wm"
            CELL_CSV="$CELL_ROOT/metrics/summary.csv"
            [[ -f "$CELL_CSV" ]] || continue
            conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/0514/pick_representative_views.py" \
                --summary-csv "$CELL_CSV" \
                --output-root "$CELL_ROOT" \
                --group-by weight_mode || true
        done
    fi
done

echo ""
echo "Done. Sweep root: $OUT_BASE"
