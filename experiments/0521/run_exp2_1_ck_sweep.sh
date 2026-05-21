#!/usr/bin/env bash
# Experiment 2-1 (0521): C_k descriptor ablation with tile_strict packing.
# Tests eigenentropy (and optionally voxel_entropy) as C_k replacements for #GS.
# Packing = tile_strict: cleanly isolates the utility ranking from packing noise.
#
# Run ONLY after diag_ck_spearman.py confirms ρ_resid > 0.3.
#
# Env overrides:
#   SCENES="hotdog ship bicycle"
#   BUDGET_PCTS="10 25 40 55 70 85 99 100"
#   WEIGHT_MODES="screen_area"
#   SCHEMES_LIST="vd_lod vd_lod_w vd_lod_c vd_lod_w_c"
#   CK_KIND="eigenentropy"
#   GRID_SHAPE="8 8 8"
#   CUDA_VISIBLE_DEVICES=3
#   DRY_RUN=1, SKIP_RENDER=1, KEEP_PLY=1
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"

CONDA_ENV="${CONDA_ENV:-gsquic}"
RENDER_ENV="${RENDER_ENV:-gaussian_splatting}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

SCENES="${SCENES:-hotdog ship bicycle}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
WEIGHT_MODES="${WEIGHT_MODES:-screen_area}"
SCHEMES_LIST="${SCHEMES_LIST:-vd_lod vd_lod_w vd_lod_c vd_lod_w_c}"
CK_KIND="${CK_KIND:-eigenentropy}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0521/exp2_1_ck_descriptor}"

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
echo "Exp 2-1 — C_k descriptor sweep (tile_strict) — output_root=$OUT_BASE"
echo "=========================================="
echo "OUTPUT_ROOT   : $OUT_BASE"
echo "SCENES        : $SCENES"
echo "BUDGET_PCTS   : $BUDGET_PCTS"
echo "SCHEMES       : $SCHEMES_LIST"
echo "WEIGHT_MODES  : $WEIGHT_MODES"
echo "CK_KIND       : $CK_KIND"
echo "GRID_SHAPE    : $GRID_SHAPE"
echo "CAMERA_INDEX  : $CAMERA_INDEX"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DRY_RUN       : ${DRY_RUN:-0}"
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
        echo "---- [$scene] weight_mode=$wm  c_kind=$CK_KIND  schemes=[$SCHEMES_LIST]  budget_pcts=[$BUDGET_PCTS] ----"
        echo "     OUT_DIR=$OUT_DIR"
        mkdir -p "$OUT_DIR"

        conda run -n "$CONDA_ENV" python "$UTIL_DIR/test_utility.py" \
            --ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape $GRID_SHAPE \
            --budget-pct $BUDGET_PCTS \
            --schemes $SCHEMES_LIST \
            --num-lod "$NUM_LOD" \
            --camera-index "$CAMERA_INDEX" \
            --packing-mode tile_strict \
            --weight-mode "$wm" \
            --c-kind "$CK_KIND" \
            --w-norm sum \
            --c-norm sum \
            --tiling-cache "$TILING_CACHE" \
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
            --title-suffix "$scene (Exp2-1 tile_strict, c_kind=$CK_KIND, w=$wm)"

        conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/0514/pick_representative_views.py" \
            --summary-csv "$OUT_DIR/metrics/summary.csv" \
            --output-root "$OUT_DIR" \
            --group-by scheme || true
    done
done

if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_RENDER:-0}" == "1" ]]; then
    echo ""; echo "Dry-run / skip-render mode — done."; exit 0
fi

SWEEP_CSV="$OUT_BASE/summary_all.csv"
conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

for scene in $SCENES; do
    SCENE_CSV="$OUT_BASE/$scene/summary_all.csv"
    conda run -n "$CONDA_ENV" python "$UTIL_DIR/experiments/concat_summaries.py" \
        --glob "$OUT_BASE/$scene/*/metrics/summary.csv" \
        --out "$SCENE_CSV" || true
done

echo ""
echo "Done. Sweep root: $OUT_BASE"
