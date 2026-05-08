#!/usr/bin/env bash
# Run render+metrics pipeline for the 0507 budget sweep.
# Must use the gaussian_splatting conda env (has diff_gaussian_rasterization_lapisgs).
#
# Usage:
#   bash experiments/run_render_metrics_0507.sh
#   OUTPUT_ROOT=/custom/path bash experiments/run_render_metrics_0507.sh
#   SAVE_RENDERS=1 bash experiments/run_render_metrics_0507.sh   # also save PNGs

set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SCRIPT_DIR="$ROOT/dlapisgs-utility/experiments"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0507_budget_sweep}"
GT_PLY="${GT_PLY:-$ROOT/exp-dataset/bicycle/point_cloud.ply}"
TRACE="${TRACE:-$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json}"
IMG_WIDTH="${IMG_WIDTH:-800}"
IMG_HEIGHT="${IMG_HEIGHT:-800}"
SH_DEGREE="${SH_DEGREE:-3}"
WHITE_BG="${WHITE_BG:-0}"
SAVE_RENDERS="${SAVE_RENDERS:-1}"
DUMMY_IMAGE="${DUMMY_IMAGE:-$ROOT/exp-dataset/chair/predictions/color/test/r_0.png}"

if [ ! -f "$GT_PLY" ]; then
    echo "ERROR: GT PLY not found: $GT_PLY"
    exit 1
fi
if [ ! -f "$TRACE" ]; then
    echo "ERROR: Trace not found: $TRACE"
    exit 1
fi
if [ ! -d "$OUTPUT_ROOT" ]; then
    echo "ERROR: Output root not found: $OUTPUT_ROOT"
    echo "Run the budget sweep first: bash experiments/run_budget_sweep_0507.sh"
    exit 1
fi

EXTRA_ARGS=""
if [ "$WHITE_BG" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --white-bg"
fi
if [ "$SAVE_RENDERS" = "1" ]; then
    RENDER_DIR="$OUTPUT_ROOT/renders"
    EXTRA_ARGS="$EXTRA_ARGS --render-dir $RENDER_DIR"
fi

echo "=========================================="
echo "Render + Metrics — 0507 Budget Sweep"
echo "=========================================="
echo "Output root : $OUTPUT_ROOT"
echo "GT PLY      : $GT_PLY"
echo "Trace       : $TRACE"
echo "Conda env   : $CONDA_ENV"
echo "Save renders: $SAVE_RENDERS"
echo "=========================================="

LAPISGS_DUMMY_IMAGE="$DUMMY_IMAGE" conda run -n "$CONDA_ENV" python3 "$SCRIPT_DIR/render_metrics_0507.py" \
    --output-root "$OUTPUT_ROOT" \
    --gt-ply "$GT_PLY" \
    --trace "$TRACE" \
    --width "$IMG_WIDTH" \
    --height "$IMG_HEIGHT" \
    --sh-degree "$SH_DEGREE" \
    $EXTRA_ARGS

echo ""
echo "Done. Metrics at: $OUTPUT_ROOT/metrics/"
