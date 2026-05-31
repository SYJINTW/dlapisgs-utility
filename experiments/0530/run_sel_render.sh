#!/usr/bin/env bash
# 2026-05-30: select + render, clean eval cameras, 4 schemes, AC/RF model.
# Copy of experiments/0527/run_sel_render_clean.sh with OUT_ROOT → output/0530/exp2_2_ml.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=X nohup bash experiments/0530/run_sel_render.sh \
#       > /tmp/sel_render_0530.log 2>&1 &
#
# Prereqs:
#   ml/models_clean/{scene}/AC/rf.pkl
#   output/0527_clean/exp4_oracle_dq/eval/{scene}/oracle_dq.npz

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
ORACLE_ROOT="output/0527_clean/exp4_oracle_dq/eval"
MODEL_ROOT="ml/models_clean"
OUT_ROOT="output/0530/exp2_2_ml"
BUDGETS="10 25 40 55 70 85 99 100"
MODEL_TYPE="${MODEL_TYPE:-rf}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# Bicycle excluded (~460 min render); run separately after analysis.
SCENES=(chair drums ficus hotdog materials mic ship)

get_param() {
    local scene=$1 key=$2
    python3 -c "
import yaml, pathlib
p = pathlib.Path('${ORACLE_ROOT}/${scene}/params.yaml')
if not p.exists():
    exit(0)
with open(p) as f:
    d = yaml.safe_load(f)
print(d.get('args', {}).get('${key}', '') or '')
" 2>/dev/null
}

scene_ply() {
    case "$1" in
        bicycle) echo "${ROOT}/exp-dataset/bicycle/point_cloud.ply" ;;
        *)       echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
    esac
}

scene_trace() {
    case "$1" in
        bicycle) echo "${ROOT}/exp-dataset/bicycle/sparse_views_100_eval.json" ;;
        *)       echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/sparse_views_100_eval.json" ;;
    esac
}

mkdir -p "$OUT_ROOT"

for SCENE in "${SCENES[@]}"; do
    ORACLE_NPZ="${ORACLE_ROOT}/${SCENE}/oracle_dq.npz"
    MODEL_DIR="${MODEL_ROOT}/${SCENE}/AC"
    PLY="$(scene_ply "$SCENE")"
    TRACE="$(scene_trace "$SCENE")"
    OUT_DIR="${OUT_ROOT}/${SCENE}"

    if [ ! -f "$ORACLE_NPZ" ]; then
        echo "SKIP $SCENE — eval oracle not found: $ORACLE_NPZ"; continue
    fi
    if [ ! -d "$MODEL_DIR" ]; then
        echo "SKIP $SCENE — model dir not found: $MODEL_DIR"; continue
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

    # ── Selection ────────────────────────────────────────────────────────────
    echo ""
    echo "=== [$SCENE] selection ==="
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

    echo "[$SCENE] selection done"

    # ── Render + metrics ─────────────────────────────────────────────────────
    echo "=== [$SCENE] render_metrics ==="
    conda run -n gaussian_splatting \
        python experiments/render_metrics.py \
        --output-root "$OUT_DIR" \
        --gt-ply "$PLY" \
        --trace "$TRACE" \
        --scene "$SCENE" \
        --delete-ply \
        2>&1 | tee "${OUT_ROOT}/${SCENE}_render.log"

    echo "[$SCENE] render done"

    # ── Pick representative views ─────────────────────────────────────────
    SUMMARY="${OUT_DIR}/metrics/summary.csv"
    if [ -f "$SUMMARY" ]; then
        echo "  pick_representative_views -> ${OUT_DIR}/representative/"
        conda run -n gsquic \
            python experiments/0514/pick_representative_views.py \
            --summary-csv "$SUMMARY" \
            --output-root "${OUT_DIR}/representative" \
            2>&1 | tail -5 || echo "  WARN: pick_representative_views failed (non-fatal)"
    fi

    echo "[$SCENE] done — results: ${OUT_DIR}/metrics/summary.csv"
done

echo ""
echo "All scenes done. Output: $OUT_ROOT"
echo "Next:"
echo "  conda run -n gsquic python experiments/concat_summaries.py \\"
echo "      --glob 'output/0530/exp2_2_ml/*/metrics/summary.csv' \\"
echo "      --output output/0530/exp2_2_ml/summary_all.csv"
echo ""
echo "  conda run -n gsquic python experiments/plot_metrics.py \\"
echo "      --summary output/0530/exp2_2_ml/summary_all.csv \\"
echo "      --output-dir output/quickplots/0530/exp2_2_ml/ \\"
echo "      --budget-pcts 10 25 40 55 70 85 99 100"
echo ""
echo "  EXIT_CODE=$?"
