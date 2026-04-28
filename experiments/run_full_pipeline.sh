#!/usr/bin/env bash
#
# Master script: Run complete 0428 pipeline
# 1. Tiling + Utility Selection
# 2. Tile Visualization
# 3. Rendering Evaluation
#

set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SCRIPT_DIR="$ROOT/dlapisgs-utility/experiments"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  3DGS Tiling + Utility + Rendering Pipeline (0428)             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Run utility selection and tiling
echo "[STEP 1/3] Tiling + Utility Selection"
echo "────────────────────────────────────────────────────────────────"
bash "$SCRIPT_DIR/run_0428.sh"
echo ""

# Step 2: Visualize tiling results
echo "[STEP 2/3] Visualize Tiling Results"
echo "────────────────────────────────────────────────────────────────"
bash "$SCRIPT_DIR/visualize_0428.sh"
echo ""

# Step 3: Render for evaluation
echo "[STEP 3/3] Rendering for Evaluation"
echo "────────────────────────────────────────────────────────────────"
bash "$SCRIPT_DIR/render_0428.sh"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Pipeline Complete!                                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. View tile layouts: bash experiments/visualize_0428.sh"
echo "2. View rendered images: $ROOT/dlapisgs-utility/experiments/0428/renders/"
echo "3. Compute metrics: cd LapisGS-object-based-renderer && python metrics.py ..."
echo ""
