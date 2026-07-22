#!/bin/bash
# Exact per-cadence-tick LOO oracle (2026-07-22) -- no nearest-pose snap, no NPZ. See
# selection_core.compute_exact_loo_scores() / streaming_sim.py's "oracle_loo_exact"
# dispatch. Kept as a SEPARATE method from the existing (approximate, pose-snapped)
# "oracle_loo" -- canary on user1 confirmed the exact version actually behaves like an
# upper bound (ties/leads ml), unlike the pose-snapped one which lagged the real
# methods. Real per-trace cost measured from the user1 canary: ~1907s (31.8min) for
# Pass 1 alone (200 ticks x 147 tiles), not a guess.
#
# n_tracks=4 only (paper default), 120/180/240 tier, separate output tree
# (n_tracks4_oracle_exact/) -- do NOT reuse n_tracks4/ or n_tracks4_oracle/, same
# overwrite-not-append reasoning as the original oracle_loo launch.
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash run_ntracks4_oracle_exact.sh <trace...>
set -e
TRACES="$*"
PLY=/mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply
TILING_CACHE=output/oracle_tiling_cache/bicycle_8x8x8.npz
OUT_BASE=output/0720/streaming_sim_stateful
RAW=${OUT_BASE}/_raw

for N in $TRACES; do
  TRACE=dataset/EyeNavGS_NTHU_Dataset/bicycle/user${N}_bicycle.csv
  echo "=== user${N} n_tracks=4 bw=120,180,240 oracle_loo_exact (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}) ==="
  conda run -n gaussian_splatting python -u streaming_sim.py \
    --output-root "${RAW}/user${N}/n_tracks4_oracle_exact" \
    --trace-file "${TRACE}" --ply "${PLY}" --tiling-cache "${TILING_CACHE}" \
    --bandwidths-mbps 120 180 240 \
    --interval-sec 0.3 --render-interval-sec 0.5 \
    --methods oracle_loo_exact \
    --weight-mode screen_area --w-norm sum --c-norm sum \
    --n-tracks 4 --png-workers 4
  mkdir -p "${OUT_BASE}/n_tracks4_oracle_exact"
  ln -sfn "../_raw/user${N}/n_tracks4_oracle_exact" "${OUT_BASE}/n_tracks4_oracle_exact/user${N}"
done
echo "GPU${CUDA_VISIBLE_DEVICES} ntracks4_oracle_exact batch done"
