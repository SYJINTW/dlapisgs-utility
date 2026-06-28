#!/usr/bin/env bash
# Exp5 selection timing — all 10 scenes × all methods.
# Usage:
#   bash experiments/0628/run_timing.sh                 # all scenes, GPU 2
#   SCENES="chair drums" bash experiments/0628/run_timing.sh
#   METHODS="heuristic ml" bash experiments/0628/run_timing.sh
#   GPU=3 bash experiments/0628/run_timing.sh
#
# Output: output/0628/selection_timing/{scene}/{method}.json
# oracle_online is slow (~8-14 min/scene) — include explicitly via METHODS override.

set -euo pipefail
cd "$(dirname "$0")/../.."   # dlapisgs-utility root

GPU="${GPU:-2}"
export CUDA_VISIBLE_DEVICES="$GPU"

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT_ROOT="output/0628/selection_timing"
ML_ROOT="output/0606/ml_models"

ALL_SCENES="chair drums ficus hotdog materials mic ship bicycle garden stump"
SCENES="${SCENES:-$ALL_SCENES}"

# Default: all fast methods + ml. Exclude oracle_online (slow, run separately).
METHODS="${METHODS:-progressive_screen_area progressive_volume progressive_vol_d2 vd_lod heuristic ml}"

ply_for() {
  local sc="$1"
  case "$sc" in
    bicycle|garden|stump) echo "${ROOT}/exp-dataset/${sc}/point_cloud.ply" ;;
    *)                    echo "${ROOT}/exp-dataset/${sc}/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
  esac
}

echo "GPU=$GPU  SCENES=$SCENES"
echo "METHODS=$METHODS"
echo "OUT=$OUT_ROOT"
echo ""

for SCENE in $SCENES; do
  PLY="$(ply_for "$SCENE")"
  TRACE="${ROOT}/exp-dataset/${SCENE}/sparse_views_eval.json"
  CACHE="output/0606/.tiling_cache/${SCENE}_8x8x8.npz"
  ML_DIR="${ML_ROOT}/${SCENE}/AC"

  if [ ! -f "$PLY" ];   then echo "[SKIP] $SCENE: PLY missing: $PLY"; continue; fi
  if [ ! -f "$TRACE" ]; then echo "[SKIP] $SCENE: trace missing"; continue; fi
  if [ ! -f "$CACHE" ]; then echo "[SKIP] $SCENE: tiling cache missing"; continue; fi

  echo "=== $SCENE ==="
  t0=$SECONDS
  conda run -n gaussian_splatting python experiments/0628/time_selection.py \
    --ply "$PLY" \
    --tiling-cache "$CACHE" \
    --camera-trace "$TRACE" \
    --methods $METHODS \
    --ml-model-dir "$ML_DIR" \
    --ml-model-type rf \
    --output-root "$OUT_ROOT" \
    --scene "$SCENE"
  echo "  done in $(( SECONDS - t0 ))s"
  echo ""
done

echo "All scenes done. Results in $OUT_ROOT/"
