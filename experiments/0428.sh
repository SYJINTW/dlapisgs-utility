#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
PLY="$ROOT/exp-dataset/bicycle/point_cloud.ply"
TRACE="$ROOT/Frustum-for-3DGS/sample_data/camera_trace/trace1.json"
OUT_DIR="$ROOT/experiments/0428"

mkdir -p "$OUT_DIR"

python "$ROOT/dlapisgs-utility/test_utility.py" \
	--ply "$PLY" \
	--output "$OUT_DIR/bicycle_vd.ply" \
	--camera-trace "$TRACE" \
	--grid-shape 4 4 4 \
	--budget-mb 100 \
	--scheme vd \
	--num-lod 1 \
	--camera-index 0

python "$ROOT/dlapisgs-utility/test_utility.py" \
	--ply "$PLY" \
	--output "$OUT_DIR/bicycle_vd_lod.ply" \
	--camera-trace "$TRACE" \
	--grid-shape 4 4 4 \
	--budget-mb 100 \
	--scheme vd_lod \
	--num-lod 1 \
	--camera-index 0

python "$ROOT/dlapisgs-utility/test_utility.py" \
	--ply "$PLY" \
	--output "$OUT_DIR/bicycle_vd_lod_w_c.ply" \
	--camera-trace "$TRACE" \
	--grid-shape 4 4 4 \
	--budget-mb 100 \
	--scheme vd_lod_w_c \
	--num-lod 1 \
	--camera-index 0