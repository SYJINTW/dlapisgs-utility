#!/bin/bash
# Workstream A: pure prune-then-render degradation sweep (experiments/measure_downsample_degradation.py).
# GPU2 batch: chair x per_scene, garden x {per_tile,per_scene}.
set -e
export CUDA_VISIBLE_DEVICES=2
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

run chair   per_scene "${ROOT}/exp-dataset/chair/checkpoint/point_cloud/iteration_30000/point_cloud.ply" "${ROOT}/exp-dataset/chair/sparse_views_eval.json" output/oracle_tiling_cache/chair_8x8x8.npz
run garden  per_tile  "${ROOT}/exp-dataset/garden/point_cloud.ply" "${ROOT}/exp-dataset/garden/sparse_views_eval.json" output/oracle_tiling_cache/garden_8x8x8.npz
run garden  per_scene "${ROOT}/exp-dataset/garden/point_cloud.ply" "${ROOT}/exp-dataset/garden/sparse_views_eval.json" output/oracle_tiling_cache/garden_8x8x8.npz
echo "GPU2 batch done"
