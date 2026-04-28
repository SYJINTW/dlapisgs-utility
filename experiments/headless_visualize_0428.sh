#!/usr/bin/env bash
# Produce headless PNG visualizations for 0428 experiments
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT_DIR="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0428}"
TILING_DIR="$ROOT/dlapisgs-tiling"

mkdir -p "$OUT_DIR/visualizations"

while IFS= read -r META; do
    REL_PATH="${META#"$OUT_DIR"/}"
    STEM="${REL_PATH%.npz}"
    VIS="${META%.npz}.vis.npz"
    OUTPNG="$OUT_DIR/visualizations/${STEM}.png"
    mkdir -p "$(dirname "$OUTPNG")"
    if [ ! -f "$VIS" ]; then
        echo "Skipping $STEM: visibility metadata not found: $VIS"
        continue
    fi
    echo "Rendering headless visualization for $STEM -> $OUTPNG"
    python3 "$TILING_DIR/headless_visualize_tiles.py" --meta "$META" --vis "$VIS" --out "$OUTPNG" --title "$STEM"
done < <(find "$OUT_DIR" -path '*/selected.npz' | sort)

echo "Saved visualizations to $OUT_DIR/visualizations/"
