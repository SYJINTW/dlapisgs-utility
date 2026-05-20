#!/usr/bin/env bash
# Experiment 4 (0519): LOO oracle ΔQ on bicycle + ship.
# For each tile, render the scene with that tile removed and record
# MSE/PSNR/SSIM against the full-scene render. See
# `.claude/exp4_oracle_dq.md` for design.
#
# Env overrides (all optional):
#   SCENES="bicycle ship"
#   NUM_CAMERAS=100                   # 10 for pilot
#   GRID_SHAPE_TAG="exp1_gs_weight"   # tiling cache scene parent
#   TILING_BASE="$UTIL_DIR/output/0514/exp1_gs_weight"
#   OUTPUT_ROOT=.../output/0519/exp4_oracle_dq
#   CUDA_VISIBLE_DEVICES=2
#   SAVE_ABLATION_PNG=1               # set 0 to skip per-tile PNGs (~43 GB)
#   SAVE_PT=1                         # set 0 to skip float16 R_full .pt dump
#   FLUSH_EVERY=10
#   SKIP_EXISTING=1                   # resume from partial NPZ
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"

RENDER_ENV="${RENDER_ENV:-gaussian_splatting}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SCENES="${SCENES:-bicycle ship}"
NUM_CAMERAS="${NUM_CAMERAS:-100}"
TILING_BASE="${TILING_BASE:-$UTIL_DIR/output/0514/exp1_gs_weight}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0519/exp4_oracle_dq}"
FLUSH_EVERY="${FLUSH_EVERY:-10}"

SAVE_ABLATION_FLAG=""
[[ "${SAVE_ABLATION_PNG:-1}" == "1" ]] && SAVE_ABLATION_FLAG="--save-ablation-png"
SAVE_PT_FLAG=""
[[ "${SAVE_PT:-1}" == "1" ]] && SAVE_PT_FLAG="--save-pt"
SKIP_FLAG=""
[[ "${SKIP_EXISTING:-1}" == "1" ]] && SKIP_FLAG="--skip-existing"

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
echo "Exp 4 — LOO oracle ΔQ — output_root=$OUT_BASE"
echo "=========================================="
echo "SCENES        : $SCENES"
echo "NUM_CAMERAS   : $NUM_CAMERAS"
echo "TILING_BASE   : $TILING_BASE"
echo "FLUSH_EVERY   : $FLUSH_EVERY"
echo "SAVE_PT       : ${SAVE_PT:-1}"
echo "SAVE_ABLATION : ${SAVE_ABLATION_PNG:-1}"
echo "SKIP_EXISTING : ${SKIP_EXISTING:-1}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$(scene_ply "$scene")"
    TRACE="$(scene_trace "$scene")"
    TILING_CACHE="$TILING_BASE/$scene/.tiling_cache.npz"

    if [[ -z "$PLY" || ! -f "$PLY" ]]; then
        echo "[skip] $scene: PLY not found at $PLY"; continue
    fi
    if [[ ! -f "$TRACE" ]]; then
        echo "[skip] $scene: trace not found at $TRACE  (run gen_sparse_views.py first)"; continue
    fi
    if [[ ! -f "$TILING_CACHE" ]]; then
        echo "[skip] $scene: tiling cache not found at $TILING_CACHE  (run Exp 1 first)"; continue
    fi

    OUT_DIR="$OUT_BASE/$scene"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "---- [$scene] LOO oracle: $NUM_CAMERAS cams ----"
    echo "     OUT_DIR=$OUT_DIR"

    conda run -n "$RENDER_ENV" python "$UTIL_DIR/experiments/exp4_oracle_dq.py" \
        --ply "$PLY" \
        --trace "$TRACE" \
        --tiling-cache "$TILING_CACHE" \
        --output-root "$OUT_DIR" \
        --num-cameras "$NUM_CAMERAS" \
        --flush-every "$FLUSH_EVERY" \
        $SAVE_PT_FLAG $SAVE_ABLATION_FLAG $SKIP_FLAG
done

echo ""
echo "Done. Oracle root: $OUT_BASE"
