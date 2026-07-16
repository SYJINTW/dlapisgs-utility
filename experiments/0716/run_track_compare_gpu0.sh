#!/bin/bash
# n_tracks={1,2,4} round_robin-vs-greedy comparison, 8-trace subsample (bicycle).
# GPU0: round_robin n_tracks=2 (new) + greedy n_tracks=2 (new).
# n_tracks=1 and round_robin n_tracks=4 are NOT rerun -- both already exist in
# output/0715/streaming_sim_multitrack/{n_tracks1,n_tracks4}/ and (for n_tracks=1
# specifically) round_robin/greedy are already GPU-regression-verified byte-identical there.
set -e
export CUDA_VISIBLE_DEVICES=0
TRACES="1 4 7 10 13 16 19 22"
PLY=/mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply
TILING_CACHE=output/oracle_tiling_cache/bicycle_8x8x8.npz
ML_MODEL_DIR=output/ml_models_experimental/per_scene/bicycle/AC
OUT_BASE=output/0716/streaming_sim_track_compare

for SCHED in round_robin greedy; do
  for N in $TRACES; do
    TRACE=dataset/EyeNavGS_NTHU_Dataset/bicycle/user${N}_bicycle.csv
    OUT=${OUT_BASE}/${SCHED}_n2/user${N}
    echo "=== ${SCHED} n_tracks=2 user${N} -> ${OUT} ==="
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
      --n-tracks 2 \
      --track-schedule ${SCHED}
  done
done
echo "GPU0 batch done"
