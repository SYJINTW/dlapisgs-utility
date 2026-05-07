#!/usr/bin/env bash
# Produce headless PNG visualizations for 0428 experiments
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT_DIR="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0428}"
TILING_DIR="$ROOT/dlapisgs-tiling"
VISIBLE_COLOR="${VISIBLE_COLOR:-green}"
CULLED_COLOR="${CULLED_COLOR:-red}"
SHOW_FRUSTUM="${SHOW_FRUSTUM:-1}"
VIS_OUT_DIR="${VIS_OUT_DIR:-$OUT_DIR/visualizations}"

mkdir -p "$VIS_OUT_DIR"

while IFS= read -r META; do
    REL_PATH="${META#"$OUT_DIR"/}"
    STEM="${REL_PATH%.npz}"
    SCHEME="$(printf '%s' "$STEM" | cut -d/ -f1)"
    CAMERA="$(printf '%s' "$STEM" | cut -d/ -f2)"
    VIS="${META%.npz}.vis.npz"
    OUTPNG="$VIS_OUT_DIR/${SCHEME}_${CAMERA}.png"
    if [ ! -f "$VIS" ]; then
        echo "Skipping $STEM: visibility metadata not found: $VIS"
        continue
    fi
    echo "Rendering headless visualization for $STEM -> $OUTPNG"
    python3 "$TILING_DIR/headless_visualize_tiles.py" \
        --meta "$META" \
        --vis "$VIS" \
        --out "$OUTPNG" \
        --title "$STEM" \
        --visible-color "$VISIBLE_COLOR" \
        --culled-color "$CULLED_COLOR" \
        --show-frustum "$SHOW_FRUSTUM"
done < <(find "$OUT_DIR" -path '*/selected.npz' | sort)

    echo "Saved visualizations to $VIS_OUT_DIR/"
