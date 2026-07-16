#!/bin/bash
# Stateful streaming_sim.py sweep (2026-07-16 rewrite): persistent per-client tile buffer,
# 300ms cadence, full ~60s session (no --duration-sec cap), 60/120/300/600Mbps, n_tracks
# swept 1/2/3/4 (n=1 isolates the pure statefulness effect vs the old stateless baseline;
# n=4 is existing multi-track canon). Each invocation already covers all 3 methods x 4
# bandwidths internally (both are vectorized CLI args) -- only n_tracks needs an outer loop,
# since it's a single-int CLI arg by design (see .claude/PLAN.md stateful-rewrite design
# note on why n_tracks stays scalar-per-run).
# GPU1: traces 1-12 (48 invocations: 12 traces x 4 n_tracks values, each covering all 3
# methods x 3 bandwidths internally). Real measured cost at 4 bandwidths was ~192s/
# invocation (user1, n_tracks=4) -> at 3 bandwidths (600Mbps dropped, 2026-07-16: too
# saturated to discriminate, see .claude/PLAN.md) expect somewhat under that, not
# re-measured -> conservatively budget ~2.5hr for this GPU's batch.
set -e
export CUDA_VISIBLE_DEVICES=1
TRACES="1 2 3 4 5 6 7 8 9 10 11 12"
PLY=/mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply
TILING_CACHE=output/oracle_tiling_cache/bicycle_8x8x8.npz
ML_MODEL_DIR=output/ml_models_experimental/per_scene/bicycle/AC
OUT_BASE=output/0717/streaming_sim_stateful

for N in $TRACES; do
  TRACE=dataset/EyeNavGS_NTHU_Dataset/bicycle/user${N}_bicycle.csv
  for NT in 1 2 3 4; do
    echo "=== user${N} n_tracks=${NT} ==="
    conda run -n gaussian_splatting python streaming_sim.py \
      --output-root "${OUT_BASE}/n_tracks${NT}/user${N}" \
      --trace-file "${TRACE}" --ply "${PLY}" --tiling-cache "${TILING_CACHE}" \
      --bandwidths-mbps 60 120 300 \
      --interval-sec 0.3 --render-interval-sec 0.5 \
      --methods vd_lod v_lod_w ml \
      --ml-model-dir "${ML_MODEL_DIR}" --ml-model-type lgbm \
      --weight-mode screen_area --w-norm sum --c-norm sum \
      --n-tracks "${NT}" --png-workers 4
  done
done
echo "GPU1 batch done"
