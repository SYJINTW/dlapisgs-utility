#!/bin/bash
# 300/600Mbps canon validation, greedy_n4 canon schedule, 24-trace full bicycle set.
# 5-arm method comparison via 2 invocations per trace:
#   noprune: vd_lod, v_lod_w, ml (--online-prune-keep-frac 1.0, default)
#   prune50: v_lod_w, ml (--online-prune-keep-frac 0.5 -- vd_lod auto-exempt anyway, skip it)
# GPU1: traces 1-12.
set -e
export CUDA_VISIBLE_DEVICES=1
TRACES="1 2 3 4 5 6 7 8 9 10 11 12"
PLY=/mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply
TILING_CACHE=output/oracle_tiling_cache/bicycle_8x8x8.npz
ML_MODEL_DIR=output/ml_models_experimental/per_scene/bicycle/AC
OUT_BASE=output/0716/streaming_sim_bw300_600

for N in $TRACES; do
  TRACE=dataset/EyeNavGS_NTHU_Dataset/bicycle/user${N}_bicycle.csv

  echo "=== noprune user${N} ==="
  conda run -n gaussian_splatting python streaming_sim.py \
    --output-root "${OUT_BASE}/noprune/user${N}" \
    --trace-file "${TRACE}" --ply "${PLY}" --tiling-cache "${TILING_CACHE}" \
    --bandwidths-mbps 300 600 \
    --interval-sec 3 --render-interval-sec 0.3 --duration-sec 30 \
    --methods vd_lod v_lod_w ml \
    --ml-model-dir "${ML_MODEL_DIR}" --ml-model-type lgbm \
    --weight-mode screen_area --w-norm sum --c-norm sum \
    --n-tracks 4 --track-schedule greedy

  echo "=== prune50 user${N} ==="
  conda run -n gaussian_splatting python streaming_sim.py \
    --output-root "${OUT_BASE}/prune50/user${N}" \
    --trace-file "${TRACE}" --ply "${PLY}" --tiling-cache "${TILING_CACHE}" \
    --bandwidths-mbps 300 600 \
    --interval-sec 3 --render-interval-sec 0.3 --duration-sec 30 \
    --methods v_lod_w ml \
    --ml-model-dir "${ML_MODEL_DIR}" --ml-model-type lgbm \
    --weight-mode screen_area --w-norm sum --c-norm sum \
    --n-tracks 4 --track-schedule greedy \
    --online-prune-keep-frac 0.5
done
echo "GPU1 batch done"
