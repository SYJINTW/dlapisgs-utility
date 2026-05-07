#!/usr/bin/env bash
#
# Render evaluation of tiling results from 0428 experiments
# Uses LapisGS renderer to generate images for metrics computation
#

set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT_DIR="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0428}"
TRACE="$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json"
RENDERER_DIR="$ROOT/LapisGS-object-based-renderer"
RENDER_OUT_DIR="$OUT_DIR/renders"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
DUMMY_IMAGE="${DUMMY_IMAGE:-$ROOT/exp-dataset/chair/predictions/color/test/r_0.png}"
PLY="$ROOT/exp-dataset/bicycle/point_cloud.ply"

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

# Check if selected PLY files exist
if ! find "$OUT_DIR" -path '*/selected.ply' | grep -q .; then
    echo "ERROR: No selected PLY files found under $OUT_DIR"
    echo "Run: bash experiments/run_0428.sh first"
    exit 1
fi

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

echo "[1/2] Rendering ground-truth reference"
LAPISGS_DUMMY_IMAGE="$DUMMY_IMAGE" conda run -n "$CONDA_ENV" python3 "$RENDERER_DIR/render-lapisgs_streaming_trace.py" \
    --name "ground_truth" \
    --gs_path_list "$PLY" \
    --gs_res_list 1 \
    --trace_path "$TRACE" \
    --sh_deg $SH_DEGREE \
    --output_dir "$RENDER_OUT_DIR" \
    --width $IMG_WIDTH \
    --height $IMG_HEIGHT \
    $([ "$WHITE_BG" == "1" ] && echo "--white_bg" || true)

echo ""

echo "[2/2] Rendering selected outputs"
while IFS= read -r PLY_FILE; do
    REL_PATH="${PLY_FILE#"$OUT_DIR"/}"
    SCHEME_CAMERA="${REL_PATH%/selected.ply}"
    echo "Rendering: $SCHEME_CAMERA"
    LAPISGS_DUMMY_IMAGE="$DUMMY_IMAGE" conda run -n "$CONDA_ENV" python3 "$RENDERER_DIR/render-lapisgs_streaming_trace.py" \
        --name "$SCHEME_CAMERA" \
        --gs_path_list "$PLY_FILE" \
        --gs_res_list 1 \
        --trace_path "$TRACE" \
        --sh_deg $SH_DEGREE \
        --output_dir "$RENDER_OUT_DIR" \
        --width $IMG_WIDTH \
        --height $IMG_HEIGHT \
        $([ "$WHITE_BG" == "1" ] && echo "--white_bg" || true)
    echo ""
done < <(find "$OUT_DIR" -path '*/selected.ply' | sort)

echo ""
echo "=========================================="
echo "Rendering Complete"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  $RENDER_OUT_DIR/ground_truth/renders/"
echo "  $RENDER_OUT_DIR/<scheme>/camera_XXX/renders/"
echo ""
echo "Next: Compute metrics using:"
echo "  python dlapisgs-utility/experiments/aggregate_metrics_0428.py --output-root \"$OUT_DIR\""
