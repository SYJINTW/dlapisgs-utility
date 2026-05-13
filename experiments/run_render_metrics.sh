#!/usr/bin/env bash
# Master render + metrics + plot pipeline.
#
# Usage — set env vars, then call this script:
#   OUTPUT_ROOT=/path/to/exp/output \
#   PLOT_GROUP_BY=weight_mode \
#   TITLE_SUFFIX="8x8x8 progressive" \
#   bash experiments/run_render_metrics.sh
#
# Required:
#   OUTPUT_ROOT   — directory produced by test_utility.py (or a run_setup*.sh)
#
# Optional overrides (all have sensible defaults):
#   GT_PLY, TRACE, IMG_WIDTH, IMG_HEIGHT, SH_DEGREE, WHITE_BG
#   SAVE_RENDERS      — 1 = save per-camera PNG renders alongside metrics (default 1)
#   PLOT_GROUP_BY     — CSV column to group plot lines by: scheme | weight_mode |
#                       w_norm | packing_mode  (default: scheme)
#   TITLE_SUFFIX      — appended to plot titles (default: empty)
#   RENDER_CONDA_ENV  — env with gaussian_splatting / LapisGS renderer (default: gaussian_splatting)
#   PLOT_CONDA_ENV    — env with matplotlib / gsquic stack (default: gsquic)
#   CUDA_VISIBLE_DEVICES — GPU to use (default: 2)

set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SCRIPT_DIR="$ROOT/dlapisgs-utility/experiments"

OUTPUT_ROOT="${OUTPUT_ROOT:?ERROR: OUTPUT_ROOT must be set}"
GT_PLY="${GT_PLY:-$ROOT/exp-dataset/bicycle/point_cloud.ply}"
TRACE="${TRACE:-$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json}"
IMG_WIDTH="${IMG_WIDTH:-800}"
IMG_HEIGHT="${IMG_HEIGHT:-800}"
SH_DEGREE="${SH_DEGREE:-3}"
WHITE_BG="${WHITE_BG:-0}"
SAVE_RENDERS="${SAVE_RENDERS:-1}"
PLOT_GROUP_BY="${PLOT_GROUP_BY:-scheme}"
TITLE_SUFFIX="${TITLE_SUFFIX:-}"
RENDER_CONDA_ENV="${RENDER_CONDA_ENV:-gaussian_splatting}"
PLOT_CONDA_ENV="${PLOT_CONDA_ENV:-gsquic}"
DUMMY_IMAGE="${DUMMY_IMAGE:-$ROOT/exp-dataset/chair/predictions/color/test/r_0.png}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

if [ ! -f "$GT_PLY" ];      then echo "ERROR: GT PLY not found: $GT_PLY";   exit 1; fi
if [ ! -f "$TRACE" ];       then echo "ERROR: Trace not found: $TRACE";      exit 1; fi
if [ ! -d "$OUTPUT_ROOT" ]; then echo "ERROR: OUTPUT_ROOT not found: $OUTPUT_ROOT"; exit 1; fi

METRICS_DIR="$OUTPUT_ROOT/metrics"

EXTRA_ARGS=""
[ "$WHITE_BG"     = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --white-bg"
[ "$SAVE_RENDERS" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --render-dir $OUTPUT_ROOT/renders"

echo "=========================================="
echo "Render + Metrics + Plot"
echo "=========================================="
echo "Output root  : $OUTPUT_ROOT"
echo "GT PLY       : $GT_PLY"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Render env   : $RENDER_CONDA_ENV"
echo "Plot env     : $PLOT_CONDA_ENV"
echo "Group by     : $PLOT_GROUP_BY"
echo "Title suffix : ${TITLE_SUFFIX:-(none)}"
echo "Save renders : $SAVE_RENDERS"
echo "=========================================="

echo ""
echo "[1/2] Rendering + computing PSNR/SSIM ..."
LAPISGS_DUMMY_IMAGE="$DUMMY_IMAGE" conda run -n "$RENDER_CONDA_ENV" python3 \
    "$SCRIPT_DIR/render_metrics.py" \
    --output-root "$OUTPUT_ROOT" \
    --gt-ply      "$GT_PLY" \
    --trace       "$TRACE" \
    --width       "$IMG_WIDTH" \
    --height      "$IMG_HEIGHT" \
    --sh-degree   "$SH_DEGREE" \
    $EXTRA_ARGS

echo ""
echo "[2/2] Plotting PSNR / SSIM curves (group-by=$PLOT_GROUP_BY) ..."
if [ ! -f "$METRICS_DIR/summary.csv" ]; then
    echo "ERROR: summary.csv not found — render_metrics.py produced no rows. Did test_utility.py run first?"
    exit 1
fi
conda run -n "$PLOT_CONDA_ENV" python3 \
    "$SCRIPT_DIR/plot_metrics.py" \
    --summary-csv "$METRICS_DIR/summary.csv" \
    --out-dir     "$METRICS_DIR" \
    --group-by    "$PLOT_GROUP_BY" \
    ${TITLE_SUFFIX:+--title-suffix "$TITLE_SUFFIX"}

echo ""
echo "Done. Metrics + plots at: $METRICS_DIR/"
