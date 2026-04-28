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
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0428}"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  3DGS Tiling + Utility + Rendering Pipeline (0428)             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Run utility selection and tiling
echo "[STEP 1/3] Tiling + Utility Selection"
echo "────────────────────────────────────────────────────────────────"
OUTPUT_ROOT="$OUTPUT_ROOT" bash "$SCRIPT_DIR/run_0428.sh"
echo ""

# Step 2: Visualize tiling results
echo "[STEP 2/3] Visualize Tiling Results"
echo "────────────────────────────────────────────────────────────────"
OUTPUT_ROOT="$OUTPUT_ROOT" bash "$SCRIPT_DIR/headless_visualize_0428.sh"
echo ""

# Step 3: Render for evaluation
echo "[STEP 3/4] Rendering for Evaluation"
echo "────────────────────────────────────────────────────────────────"
OUTPUT_ROOT="$OUTPUT_ROOT" conda run -n "$CONDA_ENV" bash "$SCRIPT_DIR/render_0428.sh"
echo ""

echo "[STEP 4/4] Aggregate Metrics"
echo "────────────────────────────────────────────────────────────────"
OUTPUT_ROOT="$OUTPUT_ROOT" conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/aggregate_metrics_0428.py"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Pipeline Complete!                                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. View tile layouts: OUTPUT_ROOT=\"$OUTPUT_ROOT\" bash experiments/headless_visualize_0428.sh"
echo "2. View rendered images: $OUTPUT_ROOT/renders/"
echo "3. Inspect aggregated metrics: $OUTPUT_ROOT/metrics/summary.csv"
echo ""
