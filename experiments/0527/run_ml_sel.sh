#!/usr/bin/env bash
# Run tile selection sweep with the ml scheme.
# Reads PLY/trace/tiling paths from oracle params.yaml (same paths used during oracle computation).
# Runs: ml vs vd_lod vs oracle_loo across all 8 scenes.
# Output: output/0527/exp2_2_ml/{scene}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash experiments/0527/run_ml_sel.sh
#
# Overrides (env vars):
#   ABLATION=ACD    (default: ACD — best on ship pilot)
#   MODEL_TYPE=rf   (default: lgbm)

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ORACLE_ROOT="output/0522/exp4_oracle_dq"
MODEL_ROOT="ml/models"
OUT_ROOT="output/0527/exp2_2_ml"
ABLATION="${ABLATION:-ACD}"
MODEL_TYPE="${MODEL_TYPE:-lgbm}"
BUDGETS="10 25 40 55 70 85 99 100"

# SCENES=(chair drums ficus hotdog materials mic ship bicycle)
SCENES=(chair drums ficus hotdog materials mic ship)


# Extract fields from params.yaml
get_param() {
    local scene=$1
    local key=$2
    python3 -c "
import yaml
with open('${ORACLE_ROOT}/${scene}/params.yaml') as f:
    p = yaml.safe_load(f)
print(p['args'].get('${key}', '') or '')
" 2>/dev/null
}

mkdir -p "$OUT_ROOT"

for SCENE in "${SCENES[@]}"; do
    ORACLE_NPZ="${ORACLE_ROOT}/${SCENE}/oracle_dq.npz"
    MODEL_DIR="${MODEL_ROOT}/${SCENE}/${ABLATION}"

    if [ ! -f "$ORACLE_NPZ" ]; then
        echo "SKIP $SCENE — oracle_dq.npz not found"
        continue
    fi
    if [ ! -d "$MODEL_DIR" ]; then
        echo "SKIP $SCENE — model dir not found: $MODEL_DIR (train first)"
        continue
    fi

    PLY=$(get_param "$SCENE" "ply")
    TRACE=$(get_param "$SCENE" "trace")
    TILING_CACHE=$(get_param "$SCENE" "tiling_cache")

    if [ -z "$PLY" ] || [ ! -f "$PLY" ]; then
        echo "SKIP $SCENE — PLY not found: $PLY"
        continue
    fi
    if [ -z "$TRACE" ] || [ ! -f "$TRACE" ]; then
        echo "SKIP $SCENE — trace not found: $TRACE"
        continue
    fi

    OUT_DIR="${OUT_ROOT}/${SCENE}"
    echo "Running $SCENE -> $OUT_DIR  (ablation=$ABLATION model=$MODEL_TYPE) ..."

    TILING_ARGS=""
    if [ -n "$TILING_CACHE" ] && [ -f "$TILING_CACHE" ]; then
        TILING_ARGS="--tiling-cache $TILING_CACHE"
    fi

    conda run -n gsquic python test_utility.py \
        --ply "$PLY" \
        --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" \
        --grid-shape 8 8 8 \
        --budget-pct $BUDGETS \
        --schemes ml vd_lod oracle_loo \
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
echo "Next:"
echo "  1. conda run -n gaussian_splatting python experiments/render_metrics.py ..."
echo "  2. python experiments/concat_summaries.py ..."
echo "  3. python experiments/plot_metrics.py ..."
