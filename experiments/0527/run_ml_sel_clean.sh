#!/usr/bin/env bash
# Selection sweep (clean): eval cameras, 4 schemes, AC/RF model.
# Schemes: ml  vd_lod  vd_lod_w  oracle_loo
# vd_lod_w uses --weight-mode screen_area (best from exp_wmode_mean: +2.3 dB over vd_lod).
# oracle_loo reads from eval oracle NPZ (no train-camera contamination).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=X bash experiments/0527/run_ml_sel_clean.sh
#
# Prereqs:
#   ml/models_clean/{scene}/AC/rf.pkl  (from run_ml_train_clean.sh)
#   output/0527_clean/exp4_oracle_dq/eval/{scene}/oracle_dq.npz

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
ORACLE_ROOT="output/0527_clean/exp4_oracle_dq/eval"
MODEL_ROOT="ml/models_clean"
OUT_ROOT="output/0527_clean/exp2_2_ml"
BUDGETS="10 25 40 55 70 85 99 100"
MODEL_TYPE="${MODEL_TYPE:-rf}"

# Bicycle excluded from render sweep (~460 min); run separately after analysis.
SCENES=(chair drums ficus hotdog materials mic ship)

get_param() {
    local scene=$1 key=$2
    python3 -c "
import yaml, pathlib
# Use eval oracle params.yaml for PLY/tiling paths
p = pathlib.Path('${ORACLE_ROOT}/${scene}/params.yaml')
if not p.exists():
    exit(0)
with open(p) as f:
    d = yaml.safe_load(f)
print(d.get('args', {}).get('${key}', '') or '')
" 2>/dev/null
}

scene_trace() {
    case "$1" in
        bicycle) echo "${ROOT}/exp-dataset/bicycle/sparse_views_100_eval.json" ;;
        *)       echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/sparse_views_100_eval.json" ;;
    esac
}

scene_ply() {
    case "$1" in
        bicycle) echo "${ROOT}/exp-dataset/bicycle/point_cloud.ply" ;;
        *)       echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
    esac
}

mkdir -p "$OUT_ROOT"

for SCENE in "${SCENES[@]}"; do
    ORACLE_NPZ="${ORACLE_ROOT}/${SCENE}/oracle_dq.npz"
    MODEL_DIR="${MODEL_ROOT}/${SCENE}/AC"
    PLY="$(scene_ply "$SCENE")"
    TRACE="$(scene_trace "$SCENE")"

    if [ ! -f "$ORACLE_NPZ" ]; then
        echo "SKIP $SCENE — eval oracle not found: $ORACLE_NPZ"; continue
    fi
    if [ ! -d "$MODEL_DIR" ]; then
        echo "SKIP $SCENE — model dir not found: $MODEL_DIR (run train first)"; continue
    fi
    if [ ! -f "$PLY" ]; then
        echo "SKIP $SCENE — PLY not found: $PLY"; continue
    fi
    if [ ! -f "$TRACE" ]; then
        echo "SKIP $SCENE — eval trace not found: $TRACE"; continue
    fi

    # Tiling cache: prefer oracle params.yaml; fall back to old exp4 cache
    TILING_CACHE="$(get_param "$SCENE" "tiling_cache")"
    if [ -z "$TILING_CACHE" ] || [ ! -f "$TILING_CACHE" ]; then
        TILING_CACHE="output/0514/exp1_gs_weight_done/${SCENE}/.tiling_cache.npz"
    fi
    TILING_ARGS=""
    [ -f "$TILING_CACHE" ] && TILING_ARGS="--tiling-cache $TILING_CACHE"

    OUT_DIR="${OUT_ROOT}/${SCENE}"
    echo "Running $SCENE -> $OUT_DIR  (model=$MODEL_TYPE) ..."

    conda run -n gsquic python test_utility.py \
        --ply "$PLY" \
        --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" \
        --grid-shape 8 8 8 \
        --budget-pct $BUDGETS \
        --schemes ml vd_lod vd_lod_w oracle_loo \
        --num-lod 1 \
        --camera-index -1 \
        --packing-mode tile_strict \
        --weight-mode screen_area \
        --w-norm sum --c-norm sum \
        --img-w 1600 --img-h 1600 \
        $TILING_ARGS \
        --oracle-npz "$ORACLE_NPZ" \
        --ml-model-dir "$MODEL_DIR" \
        --ml-model-type "$MODEL_TYPE" \
        2>&1 | tee "${OUT_ROOT}/${SCENE}_sel.log"

    echo "Done $SCENE"
done

echo ""
echo "All scenes done. Output: $OUT_ROOT"
echo "Next: bash experiments/0527/run_render_clean.sh"
