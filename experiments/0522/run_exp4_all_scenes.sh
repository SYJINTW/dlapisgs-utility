#!/usr/bin/env bash
# Exp 4 (0522): LOO + AOI oracle for all 8 scenes.
# Order: hotdog, ship, chair, drums, ficus, mic, materials, bicycle
#
# Env overrides:
#   SCENES="hotdog ship ..."
#   CUDA_VISIBLE_DEVICES=3
#   SKIP_EXISTING=1     (default 1 — safe to resume after crash)
#   COMPUTE_AOI=1       (default 1 — add-one-in alongside LOO)
#   NUM_CAMERAS=0       (0 = all cameras in trace)
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
RENDER_ENV="${RENDER_ENV:-gaussian_splatting}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

SCENES="${SCENES:-hotdog ship chair drums ficus mic materials bicycle}"
NUM_CAMERAS="${NUM_CAMERAS:-0}"
TILING_BASE="$UTIL_DIR/output/0514/exp1_gs_weight_done"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0522/exp4_oracle_dq}"
FLUSH_EVERY="${FLUSH_EVERY:-10}"

SKIP_FLAG=""; [[ "${SKIP_EXISTING:-1}" == "1" ]] && SKIP_FLAG="--skip-existing"
AOI_FLAG="";  [[ "${COMPUTE_AOI:-1}"   == "1" ]] && AOI_FLAG="--compute-aoi"

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
    case "$1" in
        bicycle)   echo "$ROOT/exp-dataset/bicycle/sparse_views_100.json" ;;
        *)         echo "$(dirname "$(scene_ply "$1")")/sparse_views_100.json" ;;
    esac
}

echo "=========================================="
echo "Exp 4 (0522) — LOO+AOI oracle"
echo "  SCENES        : $SCENES"
echo "  NUM_CAMERAS   : $NUM_CAMERAS (0=all)"
echo "  TILING_BASE   : $TILING_BASE"
echo "  COMPUTE_AOI   : ${COMPUTE_AOI:-1}"
echo "  SKIP_EXISTING : ${SKIP_EXISTING:-1}"
echo "  CUDA          : $CUDA_VISIBLE_DEVICES"
echo "  OUT_BASE      : $OUT_BASE"
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
        echo "[skip] $scene: trace not found at $TRACE"; continue
    fi
    if [[ ! -f "$TILING_CACHE" ]]; then
        echo "[skip] $scene: tiling cache not found at $TILING_CACHE"; continue
    fi

    OUT_DIR="$OUT_BASE/$scene"
    echo ""
    echo "---- [$scene] ----"
    conda run -n "$RENDER_ENV" python "$UTIL_DIR/experiments/exp4_oracle_dq.py" \
        --ply "$PLY" \
        --trace "$TRACE" \
        --tiling-cache "$TILING_CACHE" \
        --output-root "$OUT_DIR" \
        --num-cameras "$NUM_CAMERAS" \
        --width "${WIDTH:-1600}" \
        --height "${HEIGHT:-1600}" \
        --flush-every "$FLUSH_EVERY" \
        $AOI_FLAG $SKIP_FLAG
    echo "[$scene] done -> $OUT_DIR"
done

echo ""
echo "All done. Oracle root: $OUT_BASE"
