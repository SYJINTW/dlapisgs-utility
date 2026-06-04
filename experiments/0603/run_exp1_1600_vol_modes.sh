#!/usr/bin/env bash
# Experiment 1 @1600² — weight-mode sweep, volume + volume_over_d2 only.
# screen_area@1600² already exists in output/0601/exp3_packing/progressive/.
#
# Fixed:  packing=progressive, scheme=vd_lod, grid=8x8x8, img=1600x1600.
# Swept:  weight-mode ∈ {volume, volume_over_d2} × variant ∈ {ordered, random}.
# Scenes: 8 (7 synthetic + bicycle). Uses test_utility_inmem.py (shuffle merged in).
# Random control (SPEC Del 2): --shuffle-visible-seed within visible partition.
# Trace:  sparse_views_100_eval.json (matched to exp3 for cross-mode comparability).
#
# Env overrides:
#   SCENES="chair drums ficus hotdog materials mic ship"
#   WEIGHT_MODES="volume volume_over_d2"
#   CUDA_VISIBLE_DEVICES=0
#   OUTPUT_ROOT=.../output/0603/exp1_1600_vol_modes
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENES="${SCENES:-bicycle drums ship mic ficus materials chair hotdog}"
WEIGHT_MODES="${WEIGHT_MODES:-screen_area volume volume_over_d2}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
SCHEME="${SCHEME:-vd_lod}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0603/exp1_1600_vol_modes}"

scene_ply() {
    case "$1" in
        chair)     echo "$ROOT/exp-dataset/chair/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        drums)     echo "$ROOT/exp-dataset/drums/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        ficus)     echo "$ROOT/exp-dataset/ficus/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        hotdog)    echo "$ROOT/exp-dataset/hotdog/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        materials) echo "$ROOT/exp-dataset/materials/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        mic)       echo "$ROOT/exp-dataset/mic/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        ship)      echo "$ROOT/exp-dataset/ship/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
        bicycle)   echo "$ROOT/exp-dataset/bicycle/point_cloud.ply" ;;
        *) echo "" ;;
    esac
}
scene_trace() {
    echo "$(dirname "$(scene_ply "$1")")/sparse_views_100_eval.json"
}

echo "=========================================="
echo "Exp 1 @1600² — volume + volume_over_d2 weight modes"
echo "OUTPUT_ROOT  : $OUT_BASE"
echo "SCENES       : $SCENES"
echo "WEIGHT_MODES : $WEIGHT_MODES"
echo "BUDGET_PCTS  : $BUDGET_PCTS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$(scene_ply "$scene")"
    TRACE="$(scene_trace "$scene")"

    if [[ -z "$PLY" || ! -f "$PLY" ]]; then
        echo "[skip] $scene: PLY not found"; continue
    fi
    if [[ ! -f "$TRACE" ]]; then
        echo "[skip] $scene: trace not found at $TRACE"; continue
    fi

    TILING_CACHE="$OUT_BASE/$scene/.tiling_cache.npz"
    GT_CACHE="$OUT_BASE/$scene/gt_renders"
    mkdir -p "$(dirname "$TILING_CACHE")"

    for wm in $WEIGHT_MODES; do
        OUT_DIR="$OUT_BASE/$scene/$wm"
        echo ""
        echo "---- [$scene] weight_mode=$wm ----"
        mkdir -p "$OUT_DIR"

        conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
            --ply "$PLY" \
            --gt-ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape $GRID_SHAPE \
            --budget-pct $BUDGET_PCTS \
            --schemes "$SCHEME" \
            --num-lod "$NUM_LOD" \
            --camera-index "$CAMERA_INDEX" \
            --packing-mode progressive \
            --weight-mode "$wm" \
            --img-w 1600 --img-h 1600 \
            --scene "$scene" \
            --group-by weight_mode \
            --tiling-cache "$TILING_CACHE" \
            --gt-renders-cache "$GT_CACHE"

        conda run -n gsquic python "$UTIL_DIR/experiments/aggregate_timings.py" \
            --output-root "$OUT_DIR" || true
    done

    # Random baseline — one per scene, weight-mode irrelevant
    OUT_DIR="$OUT_BASE/$scene/random"
    echo ""
    echo "---- [$scene] weight_mode=random (baseline) ----"
    mkdir -p "$OUT_DIR"
    conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
        --ply "$PLY" \
        --gt-ply "$PLY" \
        --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" \
        --grid-shape $GRID_SHAPE \
        --budget-pct $BUDGET_PCTS \
        --schemes "$SCHEME" \
        --num-lod "$NUM_LOD" \
        --camera-index "$CAMERA_INDEX" \
        --packing-mode progressive \
        --weight-mode screen_area \
        --shuffle-visible-seed "$SHUFFLE_SEED" \
        --img-w 1600 --img-h 1600 \
        --scene "$scene" \
        --group-by weight_mode \
        --tiling-cache "$TILING_CACHE" \
        --gt-renders-cache "$GT_CACHE"
done

# Concat per-cell CSVs + plot
SWEEP_CSV="$OUT_BASE/summary_all.csv"
echo ""
echo "Concatenating -> $SWEEP_CSV"
conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

for scene in $SCENES; do
    SCENE_CSV="$OUT_BASE/$scene/summary_all.csv"
    conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
        --glob "$OUT_BASE/$scene/*/metrics/summary.csv" \
        --out "$SCENE_CSV" || true
    if [[ -f "$SCENE_CSV" ]]; then
        conda run -n gsquic python "$UTIL_DIR/experiments/plot_metrics.py" \
            --summary-csv "$SCENE_CSV" \
            --out-dir "$OUT_BASE/$scene/plots" \
            --group-by weight_mode \
            --title-suffix "$scene (Exp1 @1600² progressive)" || true
    fi
done

echo ""
echo "Done. Sweep root: $OUT_BASE"
