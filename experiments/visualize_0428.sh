#!/usr/bin/env bash
#
# Visualize tiling results from 0428 experiments
# Displays 3D tile AABBs for each scheme
#

set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT_DIR="$ROOT/dlapisgs-utility/experiments/0428"
TILING_DIR="$ROOT/dlapisgs-tiling"

# Check if output directory exists
if [ ! -d "$OUT_DIR" ]; then
    echo "ERROR: Output directory $OUT_DIR does not exist"
    echo "Run: bash experiments/run_0428.sh first"
    exit 1
fi

# Check if NPZ files were generated
if [ ! -f "$OUT_DIR/bicycle_vd.npz" ]; then
    echo "ERROR: NPZ files not found in $OUT_DIR"
    echo "Make sure test_utility.py was run with the latest version"
    exit 1
fi

echo "=========================================="
echo "Visualizing Tiling Results (0428)"
echo "=========================================="
echo ""
echo "Scheme: vd (visibility + distance)"
echo "  Tiles: $OUT_DIR/bicycle_vd.npz"
python3 << EOF
import sys
sys.path.insert(0, '$TILING_DIR')
from tiles_visualization import visualize_two_tile_metadata

print("Loading: $OUT_DIR/bicycle_vd.npz")
print("Loading: $OUT_DIR/bicycle_vd_lod.npz")
visualize_two_tile_metadata(
    "$OUT_DIR/bicycle_vd.npz",
    "$OUT_DIR/bicycle_vd_lod.npz",
    color_a="red",
    color_b="blue"
)
EOF

echo ""
echo "=========================================="
echo "Scheme: vd_lod (with LOD levels)"
echo "=========================================="
python3 << EOF
import sys
sys.path.insert(0, '$TILING_DIR')
from tiles_visualization import visualize_two_tile_metadata

print("Loading: $OUT_DIR/bicycle_vd_lod.npz")
print("Loading: $OUT_DIR/bicycle_vd_lod_w_c.npz")
visualize_two_tile_metadata(
    "$OUT_DIR/bicycle_vd_lod.npz",
    "$OUT_DIR/bicycle_vd_lod_w_c.npz",
    color_a="green",
    color_b="orange"
)
EOF

echo ""
echo "=========================================="
echo "Scheme: vd_lod_w_c (full model)"
echo "=========================================="
echo "NPZ file: $OUT_DIR/bicycle_vd_lod_w_c.npz"
echo ""
echo "To programmatically inspect NPZ contents:"
echo "  import numpy as np"
echo "  npz = np.load('$OUT_DIR/bicycle_vd_lod_w_c.npz')"
echo "  print('Tiles:', npz['tile_keys'].shape[0])"
echo "  print('Gaussians:', npz['flat_indices'].shape[0])"
