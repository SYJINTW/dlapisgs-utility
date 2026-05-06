#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
SCRIPT_DIR="$ROOT/dlapisgs-utility/experiments"
PLY="$ROOT/exp-dataset/bicycle/point_cloud.ply"
TRACE="$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json"
OUT_DIR="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0507_budget_sweep}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"

# Exactly 6 budget points by default. Override via BUDGET_LIST="...".
BUDGET_LIST="${BUDGET_LIST:-10 20 40 60 80 100}"
SCHEMES=("vd_lod" "vd_lod_w" "vd_lod_c" "vd_lod_w_c")
NUM_LOD="${NUM_LOD:-1}"
GRID_SHAPE="${GRID_SHAPE:-4 4 4}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"  # -1 means all views in trace (trace1 has 50)

echo ""
echo "=========================================="
echo "Budget Sweep 0507 (Selection-only)"
echo "=========================================="
echo "Output root: $OUT_DIR"
echo "Budgets MB: $BUDGET_LIST"
echo "Schemes: ${SCHEMES[*]}"
echo "Camera index mode: $CAMERA_INDEX"
echo "=========================================="
echo ""

mkdir -p "$OUT_DIR"

for budget in $BUDGET_LIST; do
    budget_tag="${budget//./p}"
    budget_out="$OUT_DIR/budget_${budget_tag}mb"
    mkdir -p "$budget_out"

    echo "[Budget ${budget} MB]"
    for scheme in "${SCHEMES[@]}"; do
        echo "  Running scheme: $scheme"
        conda run -n "$CONDA_ENV" python "$ROOT/dlapisgs-utility/test_utility.py" \
            --ply "$PLY" \
            --output-root "$budget_out" \
            --camera-trace "$TRACE" \
            --grid-shape $GRID_SHAPE \
            --budget-mb "$budget" \
            --scheme "$scheme" \
            --num-lod "$NUM_LOD" \
            --camera-index "$CAMERA_INDEX"
    done
    echo ""
done

echo "[Post] Aggregating selection metrics"
conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/aggregate_selection_sweep_0507.py" \
    --output-root "$OUT_DIR" \
    --metrics-dir "$OUT_DIR/metrics"

echo "[Post] Plotting line charts"
conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/plot_selection_sweep_0507.py" \
    --summary-csv "$OUT_DIR/metrics/summary_by_budget_scheme.csv" \
    --out-dir "$OUT_DIR/metrics"

echo ""
echo "Done."
echo "Metrics CSV: $OUT_DIR/metrics/summary_by_budget_scheme.csv"
echo "Plot (gaussians): $OUT_DIR/metrics/selected_gaussians_vs_budget.png"
echo "Plot (utilization): $OUT_DIR/metrics/budget_utilization_vs_budget.png"
