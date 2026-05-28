#!/usr/bin/env bash
# Compute LOO oracle for all 8 scenes using a specific trace set (train or eval).
#
# Usage:
#   SUFFIX=train CUDA_VISIBLE_DEVICES=0 bash experiments/0527/run_oracle.sh
#   SUFFIX=eval  CUDA_VISIBLE_DEVICES=1 bash experiments/0527/run_oracle.sh
#
# Output: output/0527_clean/exp4_oracle_dq/{SUFFIX}/{scene}/oracle_dq.npz
#
# AOI disabled (LOO only) — AOI confirmed dead weight in prior analysis.

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
SUFFIX="${SUFFIX:-train}"
OUT_BASE="$UTIL_DIR/output/0527_clean/exp4_oracle_dq/${SUFFIX}"
TILING_BASE="$UTIL_DIR/output/0514/exp1_gs_weight_done"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENES=(hotdog ship chair drums ficus mic materials bicycle)

scene_ply() {
    case "$1" in
        bicycle)   echo "$ROOT/exp-dataset/bicycle/point_cloud.ply" ;;
        *)         echo "$ROOT/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
    esac
}

scene_trace() {
    case "$1" in
        bicycle)   echo "$ROOT/exp-dataset/bicycle/sparse_views_100_${SUFFIX}.json" ;;
        *)         echo "$ROOT/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/sparse_views_100_${SUFFIX}.json" ;;
    esac
}

echo "=========================================="
echo "Oracle LOO  SUFFIX=$SUFFIX  GPU=$CUDA_VISIBLE_DEVICES"
echo "  OUT_BASE : $OUT_BASE"
echo "=========================================="
mkdir -p "$OUT_BASE"

for scene in "${SCENES[@]}"; do
    PLY="$(scene_ply "$scene")"
    TRACE="$(scene_trace "$scene")"
    TILING_CACHE="$TILING_BASE/$scene/.tiling_cache.npz"
    OUT_DIR="$OUT_BASE/$scene"

    if [ ! -f "$PLY" ]; then
        echo "[skip] $scene — PLY not found: $PLY"; continue
    fi
    if [ ! -f "$TRACE" ]; then
        echo "[skip] $scene — trace not found: $TRACE"; continue
    fi
    if [ ! -f "$TILING_CACHE" ]; then
        echo "[skip] $scene — tiling cache not found: $TILING_CACHE"; continue
    fi

    echo ""
    echo "---- [$scene] SUFFIX=$SUFFIX ----"

    conda run -n gaussian_splatting \
        python "$UTIL_DIR/experiments/exp4_oracle_dq.py" \
        --ply "$PLY" \
        --trace "$TRACE" \
        --tiling-cache "$TILING_CACHE" \
        --output-root "$OUT_DIR" \
        --num-cameras 0 \
        --width 1600 \
        --height 1600 \
        --flush-every 10 \
        --skip-existing

    echo "[$scene] done -> $OUT_DIR/oracle_dq.npz"
done

echo ""
echo "All done. SUFFIX=$SUFFIX  OUT: $OUT_BASE"
