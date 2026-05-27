#!/usr/bin/env bash
# Camera QA: render GT frames for a trace set and concat to MP4.
# Also generates 4-panel trace visualization PNGs.
#
# Usage:
#   SUFFIX=train  bash experiments/0527/qa_cameras.sh
#   SUFFIX=eval   bash experiments/0527/qa_cameras.sh
#
# Outputs:
#   output/camera_rendered/{scene}_{SUFFIX}_views.mp4   — GT render video
#   plotting/traces/{scene}_{SUFFIX}_views.png          — camera position plot
#
# Envs used:
#   gaussian_splatting  (GT rendering)
#   gsquic              (trace visualization)

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SUFFIX="${SUFFIX:-train}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"

GT_OUT_BASE="output/camera_rendered/qa"
MP4_OUT_DIR="output/camera_rendered"
PLOT_OUT_DIR="plotting/traces"

mkdir -p "$MP4_OUT_DIR" "$PLOT_OUT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "=========================================="
echo "qa_cameras  SUFFIX=$SUFFIX  GPU=$GPU"
echo "=========================================="

scene_ply() {
    case "$1" in
        bicycle)   echo "$ROOT/exp-dataset/bicycle/point_cloud.ply" ;;
        *)         echo "$ROOT/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
    esac
}

scene_trace() {
    case "$1" in
        bicycle)   echo "$ROOT/exp-dataset/bicycle/sparse_views_100_${SUFFIX}.json" ;;
        *)         echo "$ROOT/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/sparse_views_100_${SUFFIX}.json" ;;
    esac
}

SCENES=(chair drums ficus hotdog materials mic ship bicycle)

for scene in "${SCENES[@]}"; do
    PLY="$(scene_ply "$scene")"
    TRACE="$(scene_trace "$scene")"
    GT_DIR="${GT_OUT_BASE}/${scene}_${SUFFIX}"
    MP4="${MP4_OUT_DIR}/${scene}_${SUFFIX}_views.mp4"
    PLOT="${PLOT_OUT_DIR}/${scene}_${SUFFIX}_views.png"

    if [ ! -f "$PLY" ]; then
        echo "SKIP $scene — PLY not found: $PLY"; continue
    fi
    if [ ! -f "$TRACE" ]; then
        echo "SKIP $scene — trace not found: $TRACE"; continue
    fi

    echo ""
    echo "=== [$scene] ==="

    # ── 1. GT render ─────────────────────────────────────────────────────────
    echo "  Rendering GT frames -> ${GT_DIR}/gt_renders/ ..."
    conda run -n gaussian_splatting \
        python experiments/render_metrics.py \
        --output-root "$GT_DIR" \
        --gt-ply "$PLY" \
        --trace "$TRACE" \
        --scene "$scene" \
        2>&1 | grep -E "camera_[0-9]+\.png|GT rendered|ERROR|SUCCESS|SKIP" || true

    # ── 2. ffmpeg → MP4 ──────────────────────────────────────────────────────
    PNG_DIR="${GT_DIR}/gt_renders"
    N_PNGS=$(ls "${PNG_DIR}"/camera_*.png 2>/dev/null | wc -l)
    if [ "$N_PNGS" -eq 0 ]; then
        echo "  WARN: no PNGs in ${PNG_DIR} — skipping MP4"
    else
        echo "  ffmpeg: $N_PNGS frames -> $MP4"
        ffmpeg -y -framerate 10 \
            -pattern_type glob -i "${PNG_DIR}/camera_*.png" \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
            -c:v libx264 -pix_fmt yuv420p \
            "$MP4" 2>&1 | tail -3
        echo "  Wrote $MP4"
    fi

    # ── 3. Trace position plot ────────────────────────────────────────────────
    echo "  Generating trace plot -> $PLOT ..."
    conda run -n gsquic \
        python experiments/visualize_trace_views.py \
        --trace "$TRACE" \
        --ply "$PLY" \
        --out "$PLOT" \
        2>&1 | tail -3 || echo "  WARN: trace plot failed (non-fatal)"

    echo "  Done $scene"
done

echo ""
echo "=========================================="
echo "QA done.  SUFFIX=$SUFFIX"
echo "Videos : $MP4_OUT_DIR/{scene}_${SUFFIX}_views.mp4"
echo "Plots  : $PLOT_OUT_DIR/{scene}_${SUFFIX}_views.png"
echo "Review before proceeding to oracle compute."
echo "=========================================="
