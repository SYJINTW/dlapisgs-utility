#!/bin/bash
# Single missing point on the 60/120/300 Mbps canon curve (output/0717/): n_tracks=16 was
# never run at that tier (only 1-8 swept). User's hypothesis from the 120/180/240 by_ntracks
# plot: higher bandwidth favors more tracks before the makespan/straggler penalty kicks in
# (see PLAN.md 3d closed-form makespan note) -- 300Mbps is the highest tier we have, so this
# is the sharpest test of whether the T=8 peak keeps climbing at T=16 there too. Scoped to
# ONLY bandwidth=300 (not 60/120) per direct instruction -- those n_tracks16 points weren't
# asked for.
#
# output/0717/streaming_sim_stateful/n_tracks16/{user1,user13} already exist but are stale
# partial canary dirs (sanity_check + gt_renders only, no metrics/summary.csv from an old
# aborted run) -- this overwrites them with a real full run, same as every other user.
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash run_bw300_ntracks16.sh <trace...>
set -e
TRACES="$*"
PLY=/mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply
TILING_CACHE=output/oracle_tiling_cache/bicycle_8x8x8.npz
ML_MODEL_DIR=output/ml_models_experimental/per_scene/bicycle/AC
OUT_BASE=output/0717/streaming_sim_stateful

for N in $TRACES; do
  TRACE=dataset/EyeNavGS_NTHU_Dataset/bicycle/user${N}_bicycle.csv
  echo "=== user${N} n_tracks=16 bw=300 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}) ==="
  conda run -n gaussian_splatting python streaming_sim.py \
    --output-root "${OUT_BASE}/n_tracks16/user${N}" \
    --trace-file "${TRACE}" --ply "${PLY}" --tiling-cache "${TILING_CACHE}" \
    --bandwidths-mbps 300 \
    --interval-sec 0.3 --render-interval-sec 0.5 \
    --methods vd_lod v_lod_w ml \
    --ml-model-dir "${ML_MODEL_DIR}" --ml-model-type lgbm \
    --weight-mode screen_area --w-norm sum --c-norm sum \
    --n-tracks 16 --png-workers 4
done
echo "GPU${CUDA_VISIBLE_DEVICES} bw300_ntracks16 batch done"
