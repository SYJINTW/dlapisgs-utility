#!/bin/bash
# Workstream A: pure prune-then-render degradation sweep (experiments/measure_downsample_degradation.py).
# GPU1 batch: bicycle x {per_tile,per_scene}, chair x per_tile.
set -e
export CUDA_VISIBLE_DEVICES=1
ROOT=/mnt/data1/samk/gs-quic/cs5262_tile_quic
OUT_BASE=output/0716/downsample_degradation

run() {
  local scene=$1 mode=$2 ply=$3 trace=$4 tiling=$5
  local out=${OUT_BASE}/${scene}/${mode}
  echo "=== ${scene} ${mode} -> ${out} ==="
  conda run -n gaussian_splatting python experiments/measure_downsample_degradation.py \
    --ply "${ply}" \
    --tiling-cache "${tiling}" \
    --camera-trace "${trace}" \
    --scene "${scene}" \
    --mode "${mode}" \
    --keep-fracs 0.5 0.33 0.1 \
    --output-root "${out}"
}

run bicycle per_tile  "${ROOT}/exp-dataset/bicycle/point_cloud.ply" "${ROOT}/exp-dataset/bicycle/sparse_views_eval.json" output/oracle_tiling_cache/bicycle_8x8x8.npz
run bicycle per_scene "${ROOT}/exp-dataset/bicycle/point_cloud.ply" "${ROOT}/exp-dataset/bicycle/sparse_views_eval.json" output/oracle_tiling_cache/bicycle_8x8x8.npz
run chair   per_tile  "${ROOT}/exp-dataset/chair/checkpoint/point_cloud/iteration_30000/point_cloud.ply" "${ROOT}/exp-dataset/chair/sparse_views_eval.json" output/oracle_tiling_cache/chair_8x8x8.npz
echo "GPU1 batch done"
