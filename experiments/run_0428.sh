#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
PLY="$ROOT/exp-dataset/bicycle/point_cloud.ply"
TRACE="$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json"
OUT_DIR="${OUTPUT_ROOT:-$ROOT/dlapisgs-utility/output/0428}"

mkdir -p "$OUT_DIR"

SCHEMES=("vd_lod" "vd_lod_w" "vd_lod_c" "vd_lod_w_c")
NUM_LOD=1
BUDGET_MB=100
GRID_SHAPE="4 4 4"

for scheme in "${SCHEMES[@]}"; do
	echo "Running utility selection for scheme: $scheme"
	python "$ROOT/dlapisgs-utility/test_utility.py" \
		--ply "$PLY" \
		--output-root "$OUT_DIR" \
		--camera-trace "$TRACE" \
		--grid-shape $GRID_SHAPE \
		--budget-mb "$BUDGET_MB" \
		--scheme "$scheme" \
		--num-lod "$NUM_LOD" \
		--camera-index -1
done