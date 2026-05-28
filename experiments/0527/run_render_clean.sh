#!/usr/bin/env bash
# Render metrics for clean selection output (eval cameras, 4 schemes).
# Runs pick_representative_views.py per scene after metrics.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=X bash experiments/0527/run_render_clean.sh
#
# Prereq: output/0527_clean/exp2_2_ml/{scene}/ply/  (from run_ml_sel_clean.sh)

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SEL_ROOT="output/0527_clean/exp2_2_ml"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

SCENES=(chair drums ficus hotdog materials mic ship)

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

for SCENE in "${SCENES[@]}"; do
    SEL_DIR="${SEL_ROOT}/${SCENE}"
    PLY="$(scene_ply "$SCENE")"
    TRACE="$(scene_trace "$SCENE")"

    if [ ! -d "$SEL_DIR" ]; then
        echo "SKIP $SCENE — selection output not found: $SEL_DIR"; continue
    fi
    if [ ! -f "$PLY" ]; then
        echo "SKIP $SCENE — PLY not found: $PLY"; continue
    fi

    echo ""
    echo "=== [$SCENE] render_metrics ==="

    conda run -n gaussian_splatting \
        python experiments/render_metrics.py \
        --output-root "$SEL_DIR" \
        --gt-ply "$PLY" \
        --trace "$TRACE" \
        --scene "$SCENE" \
        --delete-ply \
        2>&1 | tee "${SEL_ROOT}/${SCENE}_render.log"

    echo "[$SCENE] render done"

    # ── Pick representative views ─────────────────────────────────────────
    SUMMARY="${SEL_DIR}/metrics/summary.csv"
    if [ -f "$SUMMARY" ]; then
        echo "  pick_representative_views -> ${SEL_DIR}/representative/"
        conda run -n gsquic \
            python experiments/0514/pick_representative_views.py \
            --summary-csv "$SUMMARY" \
            --output-root "${SEL_DIR}/representative" \
            2>&1 | tail -5 || echo "  WARN: pick_representative_views failed (non-fatal)"
    fi
done

echo ""
echo "All scenes done."
echo "Next:"
echo "  conda run -n gsquic python experiments/concat_summaries.py \\"
echo "      --glob 'output/0527_clean/exp2_2_ml/*/metrics/summary.csv' \\"
echo "      --output output/0527_clean/exp2_2_ml/summary_all.csv"
echo ""
echo "  conda run -n gsquic python experiments/plot_metrics.py \\"
echo "      --summary output/0527_clean/exp2_2_ml/summary_all.csv \\"
echo "      --output-dir output/quickplots/0527_clean/exp2_2_ml/ \\"
echo "      --budget-pcts 10 25 40 55 70 85 99 100"
