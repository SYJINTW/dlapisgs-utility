#!/bin/bash
# New bandwidth tier (2026-07-20): 120/180/240 Mbps replaces 60/120/300 canon -- 300 Mbps
# saturated too fast and regressed with more tracks, 60 Mbps was bandwidth-starved at every
# n_tracks (see .claude/PLAN.md n_tracks1-8 reading). n_tracks sparsified to 1,2,4,8,16
# (3/5/6/7 added little over 4/8 in the prior 1-8 sweep). Same trace/method canon as
# experiments/0717/run_stateful_sweep_gpu2.sh, output goes to a fresh output/0720/ dir
# (not output/0717/ -- different bandwidth tier, not directly comparable/mergeable).
# GPU2: traces 13-24. Raw nesting is user{N}/n_tracks{NT}/ (fold's native order), symlinked
# into canonical n_tracks{NT}/user{N}/ for aggregate_streaming_sim.py's glob convention --
# same pattern as experiments/0720/run_stateful_sweep_ntracks5678_gpu2.sh.
set -e
export CUDA_VISIBLE_DEVICES=2
TRACES="13 14 15 16 17 18 19 20 21 22 23 24"
PLY=/mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply
TILING_CACHE=output/oracle_tiling_cache/bicycle_8x8x8.npz
ML_MODEL_DIR=output/ml_models_experimental/per_scene/bicycle/AC
OUT_BASE=output/0720/streaming_sim_stateful
RAW=${OUT_BASE}/_raw

for N in $TRACES; do
  TRACE=dataset/EyeNavGS_NTHU_Dataset/bicycle/user${N}_bicycle.csv
  echo "=== user${N} n_tracks=1,2,4,8,16 ==="
  conda run -n gaussian_splatting python streaming_sim.py \
    --output-root "${RAW}/user${N}" \
    --trace-file "${TRACE}" --ply "${PLY}" --tiling-cache "${TILING_CACHE}" \
    --bandwidths-mbps 120 180 240 \
    --interval-sec 0.3 --render-interval-sec 0.5 \
    --methods vd_lod v_lod_w ml \
    --ml-model-dir "${ML_MODEL_DIR}" --ml-model-type lgbm \
    --weight-mode screen_area --w-norm sum --c-norm sum \
    --n-tracks 1 2 4 8 16 --png-workers 4
  for NT in 1 2 4 8 16; do
    mkdir -p "${OUT_BASE}/n_tracks${NT}"
    ln -sfn "../_raw/user${N}/n_tracks${NT}" "${OUT_BASE}/n_tracks${NT}/user${N}"
  done
done
echo "GPU2 bw120180240_ntracks124816 batch done"
