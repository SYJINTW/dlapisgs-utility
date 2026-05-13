#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="0508_vd_weight_sweep"
ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SCRIPT_DIR="$ROOT/dlapisgs-utility/experiments"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/$EXP_NAME}"
GT_PLY="${GT_PLY:-$ROOT/exp-dataset/bicycle/point_cloud.ply}"
TRACE="${TRACE:-$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json}"
IMG_WIDTH="${IMG_WIDTH:-800}"
IMG_HEIGHT="${IMG_HEIGHT:-800}"
SH_DEGREE="${SH_DEGREE:-3}"
WHITE_BG="${WHITE_BG:-0}"
SAVE_RENDERS="${SAVE_RENDERS:-1}"
DUMMY_IMAGE="${DUMMY_IMAGE:-$ROOT/exp-dataset/chair/predictions/color/test/r_0.png}"

if [ ! -f "$GT_PLY" ]; then echo "ERROR: GT PLY not found: $GT_PLY"; exit 1; fi
if [ ! -f "$TRACE" ];   then echo "ERROR: Trace not found: $TRACE";   exit 1; fi
if [ ! -d "$OUTPUT_ROOT" ]; then
    echo "ERROR: Output root not found: $OUTPUT_ROOT"
    exit 1
fi

EXTRA_ARGS=""
[ "$WHITE_BG"    = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --white-bg"
[ "$SAVE_RENDERS" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --render-dir $OUTPUT_ROOT/renders"

echo "=========================================="
echo "Render + Metrics — $EXP_NAME"
echo "=========================================="
echo "Output root : $OUTPUT_ROOT"
echo "GT PLY      : $GT_PLY"
echo "Conda env   : $CONDA_ENV"
echo "Save renders: $SAVE_RENDERS"
echo "=========================================="

LAPISGS_DUMMY_IMAGE="$DUMMY_IMAGE" conda run -n "$CONDA_ENV" python3 "$SCRIPT_DIR/render_metrics.py" \
    --output-root "$OUTPUT_ROOT" \
    --gt-ply "$GT_PLY" \
    --trace "$TRACE" \
    --width "$IMG_WIDTH" \
    --height "$IMG_HEIGHT" \
    --sh-degree "$SH_DEGREE" \
    $EXTRA_ARGS

echo ""
echo "[Post] Plotting PSNR / SSIM curves"
conda run -n gsquic python3 "$SCRIPT_DIR/plot_metrics.py" \
    --summary-csv "$OUTPUT_ROOT/metrics/summary.csv" \
    --out-dir     "$OUTPUT_ROOT/metrics" \
    --title-suffix "8x8x8 grid"

echo ""
echo "Done. Metrics at: $OUTPUT_ROOT/metrics/"
