#!/usr/bin/env bash
# Generate 100-camera sparse-view traces for all 8 scenes.
#
# Usage:
#   SEED=42  SUFFIX=train bash experiments/0527/gen_traces.sh   # train set
#   SEED=1   SUFFIX=eval  bash experiments/0527/gen_traces.sh   # eval set
#
# Output per scene:
#   exp-dataset/{scene}/.../sparse_views_100_{SUFFIX}.json   (synthetic)
#   exp-dataset/bicycle/sparse_views_100_{SUFFIX}.json       (bicycle)

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SEED="${SEED:-42}"
SUFFIX="${SUFFIX:-train}"
N_VIEWS=100

echo "=========================================="
echo "gen_traces  SEED=$SEED  SUFFIX=$SUFFIX"
echo "=========================================="

scene_ply() {
    case "$1" in
        bicycle)   echo "$ROOT/exp-dataset/bicycle/point_cloud.ply" ;;
        *)         echo "$ROOT/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
    esac
}

scene_out() {
    case "$1" in
        bicycle)   echo "$ROOT/exp-dataset/bicycle/sparse_views_100_${SUFFIX}.json" ;;
        *)         echo "$ROOT/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/sparse_views_100_${SUFFIX}.json" ;;
    esac
}

SCENES=(chair drums ficus hotdog materials mic ship bicycle)

for scene in "${SCENES[@]}"; do
    PLY="$(scene_ply "$scene")"
    OUT="$(scene_out "$scene")"

    if [ ! -f "$PLY" ]; then
        echo "SKIP $scene — PLY not found: $PLY"
        continue
    fi

    if [ "$scene" = "bicycle" ]; then
        SCENE_TYPE="mipnerf360"
    else
        SCENE_TYPE="synthetic"
    fi

    echo "--- [$scene]  scene_type=$SCENE_TYPE  seed=$SEED  -> $OUT"

    conda run -n gsquic python experiments/gen_sparse_views.py \
        --ply "$PLY" \
        --out "$OUT" \
        --n-views "$N_VIEWS" \
        --scene-type "$SCENE_TYPE" \
        --seed "$SEED" \
        2>&1

    echo "Done $scene -> $OUT"
done

echo ""
echo "All done. SUFFIX=$SUFFIX  SEED=$SEED"
echo "Next: verify with output/camera_rendered videos"
echo "  bash experiments/0527/qa_cameras.sh SUFFIX=$SUFFIX"
