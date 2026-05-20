#!/usr/bin/env bash
# Experiment 1 (0514) — incremental-plotting variant.
# Identical sweep to run_exp1_gs_weight.sh; only difference: per-scene plots are
# refreshed after EACH (scene, weight_mode) cell finishes, so you can inspect
# results as they come in instead of waiting for the entire sweep (incl. bicycle).
#
# After cell N of a given scene, the scene plot shows N weight_mode curves.
# When all 3 cells for a scene are done, the plot is final.
#
# Env overrides: same as run_exp1_gs_weight.sh.
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTIL_DIR="$ROOT/dlapisgs-utility"

CONDA_ENV="${CONDA_ENV:-gsquic}"
RENDER_ENV="${RENDER_ENV:-gaussian_splatting}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SCENES="${SCENES:-chair drums ficus hotdog materials mic ship bicycle}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
WEIGHT_MODES="${WEIGHT_MODES:-volume volume_over_d2 screen_area}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
SCHEME="${SCHEME:-vd_lod_w_c}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0518/exp1_gs_weight}"

DRY_RUN_FLAG=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_FLAG="--dry-run"

DELETE_PLY_FLAG="--delete-ply"
[[ "${KEEP_PLY:-0}" == "1" ]] && DELETE_PLY_FLAG=""

scene_ply() {
    case "$1" in
        bicycle)   echo "$ROOT/exp-dataset/bicycle/point_cloud.ply" ;;
        chair)     echo "$ROOT/exp-dataset/chair/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        drums)     echo "$ROOT/exp-dataset/drums/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        ficus)     echo "$ROOT/exp-dataset/ficus/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        hotdog)    echo "$ROOT/exp-dataset/hotdog/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        materials) echo "$ROOT/exp-dataset/materials/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        mic)       echo "$ROOT/exp-dataset/mic/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        ship)      echo "$ROOT/exp-dataset/ship/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        *) echo "" ;;
    esac
}
scene_trace() {
    local ply; ply="$(scene_ply "$1")"
    echo "$(dirname "$ply")/sparse_views_100.json"
}

echo "=========================================="
echo "Exp 1 — GS weight sweep (incremental plotting) — output_root=$OUT_BASE"
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
    TRACE="$(scene_trace "$scene")"

    if [[ -z "$PLY" || ! -f "$PLY" ]]; then
        echo "[skip] $scene: PLY not found at $PLY"; continue
    fi
    if [[ ! -f "$TRACE" ]]; then
        echo "[skip] $scene: trace not found at $TRACE  (run gen_sparse_views.py first)"; continue
    fi

    TILING_CACHE="$OUT_BASE/$scene/.tiling_cache.npz"
    mkdir -p "$(dirname "$TILING_CACHE")"

    for wm in $WEIGHT_MODES; do
        TAG="$wm"
        OUT_DIR="$OUT_BASE/$scene/$TAG"
        echo ""
        echo "---- [$scene] weight_mode=$wm  budget_pcts=[$BUDGET_PCTS] ----"
        echo "     OUT_DIR=$OUT_DIR"
        mkdir -p "$OUT_DIR"

        conda run -n "$CONDA_ENV" python "$UTIL_DIR/test_utility.py" \
            --ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape $GRID_SHAPE \
            --budget-pct $BUDGET_PCTS \
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

        conda run -n "$RENDER_ENV" python "$UTIL_DIR/experiments/render_metrics.py" \
            --output-root "$OUT_DIR" \
            --gt-ply "$PLY" \
            --trace "$TRACE" \
            --scene "$scene" \
            --render-dir "$OUT_DIR/renders" \
            $DELETE_PLY_FLAG

        conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/aggregate_timings.py" \
            --output-root "$OUT_DIR" || true

        # --- Incremental per-scene plot refresh ---
        # Rebuild scene-level summary_all.csv from whatever cells exist so far,
        # then redraw the weight_mode comparison plot. After cell N: N curves.
        SCENE_CSV="$OUT_BASE/$scene/summary_all.csv"
        conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/concat_summaries.py" \
            --glob "$OUT_BASE/$scene/*/metrics/summary.csv" \
            --out "$SCENE_CSV" || true
        if [[ -f "$SCENE_CSV" ]]; then
            conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/plot_metrics.py" \
                --summary-csv "$SCENE_CSV" \
                --out-dir "$OUT_BASE/$scene/plots" \
                --group-by weight_mode \
                --title-suffix "$scene (Exp1 progressive)" || true
        fi

        # Representative views for this cell.
        CELL_CSV="$OUT_DIR/metrics/summary.csv"
        if [[ -f "$CELL_CSV" ]]; then
            conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/0514/pick_representative_views.py" \
                --summary-csv "$CELL_CSV" \
                --output-root "$OUT_DIR" \
                --group-by weight_mode || true
        fi
    done
done

if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_RENDER:-0}" == "1" ]]; then
    echo ""; echo "Dry-run / skip-render mode — done."; exit 0
fi

# Final sweep-level concat (per-scene CSVs and plots are already current).
SWEEP_CSV="$OUT_BASE/summary_all.csv"
echo ""
echo "Concatenating per-cell summary.csv -> $SWEEP_CSV"
conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

echo ""
echo "Done. Sweep root: $OUT_BASE"
