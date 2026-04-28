#!/usr/bin/env bash
#
# Render evaluation of tiling results from 0428 experiments
# Uses LapisGS renderer to generate images for metrics computation
#

set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT_DIR="$ROOT/dlapisgs-utility/experiments/0428"
TRACE="$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json"
RENDERER_DIR="$ROOT/LapisGS-object-based-renderer"
RENDER_OUT_DIR="$ROOT/dlapisgs-utility/experiments/0428/renders"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
DUMMY_IMAGE="${DUMMY_IMAGE:-$ROOT/exp-dataset/chair/predictions/color/test/r_0.png}"

# Rendering parameters
IMG_WIDTH=${IMG_WIDTH:-800}
IMG_HEIGHT=${IMG_HEIGHT:-800}
SH_DEGREE=${SH_DEGREE:-3}
WHITE_BG=${WHITE_BG:-0}

# Check prerequisites
if [ ! -d "$OUT_DIR" ]; then
    echo "ERROR: Output directory $OUT_DIR does not exist"
    echo "Run: bash experiments/run_0428.sh first"
    exit 1
fi

if [ ! -f "$RENDERER_DIR/render-lapisgs_streaming_trace.py" ]; then
    echo "ERROR: Renderer script not found at $RENDERER_DIR/render-lapisgs_streaming_trace.py"
    exit 1
fi

# Check if PLY files exist
for ply in bicycle_vd.ply bicycle_vd_lod.ply bicycle_vd_lod_w_c.ply; do
    if [ ! -f "$OUT_DIR/$ply" ]; then
        echo "ERROR: PLY file not found: $OUT_DIR/$ply"
        exit 1
    fi
done

mkdir -p "$RENDER_OUT_DIR"

echo "=========================================="
echo "Rendering Evaluation (0428 Experiments)"
echo "=========================================="
echo "Trace: $TRACE"
echo "Width x Height: $IMG_WIDTH x $IMG_HEIGHT"
echo "Output: $RENDER_OUT_DIR"
echo "Conda env: $CONDA_ENV"
echo "Dummy image: $DUMMY_IMAGE"
echo "=========================================="
echo ""

# Scheme 1: vd (visibility + distance)
echo "[1/3] Rendering: vd (visibility + distance)"
LAPISGS_DUMMY_IMAGE="$DUMMY_IMAGE" conda run -n "$CONDA_ENV" python3 "$RENDERER_DIR/render-lapisgs_streaming_trace.py" \
    --name "bicycle_vd" \
    --gs_path_list "$OUT_DIR/bicycle_vd.ply" \
    --gs_res_list 1 \
    --trace_path "$TRACE" \
    --sh_deg $SH_DEGREE \
    --output_dir "$RENDER_OUT_DIR" \
    --width $IMG_WIDTH \
    --height $IMG_HEIGHT \
    $([ "$WHITE_BG" == "1" ] && echo "--white_bg" || true)

echo ""

# Scheme 2: vd_lod (with LOD levels)
echo "[2/3] Rendering: vd_lod (with LOD levels)"
LAPISGS_DUMMY_IMAGE="$DUMMY_IMAGE" conda run -n "$CONDA_ENV" python3 "$RENDERER_DIR/render-lapisgs_streaming_trace.py" \
    --name "bicycle_vd_lod" \
    --gs_path_list "$OUT_DIR/bicycle_vd_lod.ply" \
    --gs_res_list 1 \
    --trace_path "$TRACE" \
    --sh_deg $SH_DEGREE \
    --output_dir "$RENDER_OUT_DIR" \
    --width $IMG_WIDTH \
    --height $IMG_HEIGHT \
    $([ "$WHITE_BG" == "1" ] && echo "--white_bg" || true)

echo ""

# Scheme 3: vd_lod_w_c (full model)
echo "[3/3] Rendering: vd_lod_w_c (full model with weights + complexity)"
LAPISGS_DUMMY_IMAGE="$DUMMY_IMAGE" conda run -n "$CONDA_ENV" python3 "$RENDERER_DIR/render-lapisgs_streaming_trace.py" \
    --name "bicycle_vd_lod_w_c" \
    --gs_path_list "$OUT_DIR/bicycle_vd_lod_w_c.ply" \
    --gs_res_list 1 \
    --trace_path "$TRACE" \
    --sh_deg $SH_DEGREE \
    --output_dir "$RENDER_OUT_DIR" \
    --width $IMG_WIDTH \
    --height $IMG_HEIGHT \
    $([ "$WHITE_BG" == "1" ] && echo "--white_bg" || true)

echo ""
echo "=========================================="
echo "Rendering Complete"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  $RENDER_OUT_DIR/bicycle_vd/renders/"
echo "  $RENDER_OUT_DIR/bicycle_vd_lod/renders/"
echo "  $RENDER_OUT_DIR/bicycle_vd_lod_w_c/renders/"
echo ""
echo "Next: Compute metrics using:"
echo "  python LapisGS-object-based-renderer/metrics.py \\"
echo "    --model-paths <render_dir> \\"
echo "    --gt-dir <ground_truth> \\"
echo "    --renders-dir <renders>"
