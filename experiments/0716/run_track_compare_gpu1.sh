#!/bin/bash
# GPU1: greedy n_tracks=4 (new) -- round_robin n_tracks=4 already exists in
# output/0715/streaming_sim_multitrack/n_tracks4/, reused not rerun.
set -e
export CUDA_VISIBLE_DEVICES=1
TRACES="1 4 7 10 13 16 19 22"
PLY=/mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply
TILING_CACHE=output/oracle_tiling_cache/bicycle_8x8x8.npz
ML_MODEL_DIR=output/ml_models_experimental/per_scene/bicycle/AC
OUT_BASE=output/0716/streaming_sim_track_compare

for N in $TRACES; do
  TRACE=dataset/EyeNavGS_NTHU_Dataset/bicycle/user${N}_bicycle.csv
  OUT=${OUT_BASE}/greedy_n4/user${N}
  echo "=== greedy n_tracks=4 user${N} -> ${OUT} ==="
  conda run -n gaussian_splatting python streaming_sim.py \
    --output-root "${OUT}" \
    --trace-file "${TRACE}" \
    --ply "${PLY}" \
    --tiling-cache "${TILING_CACHE}" \
    --bandwidths-mbps 600 900 1200 \
    --interval-sec 3 \
    --render-interval-sec 0.3 \
    --duration-sec 30 \
    --methods vd_lod v_lod_w ml \
    --ml-model-dir "${ML_MODEL_DIR}" \
    --ml-model-type lgbm \
    --weight-mode screen_area --w-norm sum --c-norm sum \
    --n-tracks 4 \
    --track-schedule greedy
done
echo "GPU1 batch done"
