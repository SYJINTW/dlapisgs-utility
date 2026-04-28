#!/usr/bin/env bash
# Produce headless PNG visualizations for 0428 experiments
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT_DIR="$ROOT/dlapisgs-utility/experiments/0428"
TILING_DIR="$ROOT/dlapisgs-tiling"

mkdir -p "$OUT_DIR/visualizations"

for base in bicycle_vd bicycle_vd_lod bicycle_vd_lod_w_c; do
    META="$OUT_DIR/${base}.npz"
    VIS="$OUT_DIR/${base}.vis.npz"
    OUTPNG="$OUT_DIR/visualizations/${base}.png"
    if [ ! -f "$META" ]; then
        echo "Skipping $base: metadata not found: $META"
        continue
    fi
    echo "Rendering headless visualization for $base -> $OUTPNG"
    python3 "$TILING_DIR/headless_visualize_tiles.py" --meta "$META" --vis "$VIS" --out "$OUTPNG" --title "$base"
done

echo "Saved visualizations to $OUT_DIR/visualizations/"
