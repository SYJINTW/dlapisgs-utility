#!/bin/bash
# Workstream B (oracle_loo in streaming) -- fully wired, never actually run before this.
# streaming_sim.py's summary.csv write is a full overwrite per invocation (open(..., "w"),
# metric_rows built fresh from THIS run's --methods only, confirmed by reading the code) --
# NOT an append. Running --methods oracle_loo alone, to a SEPARATE output tree, avoids
# wastefully re-rendering the already-collected vd_lod/v_lod_w/ml rows in n_tracks4/ and
# avoids any risk of clobbering them. Merge oracle_loo's summary.csv with the existing one
# at analysis time (aggregate_streaming_sim.py --extra-glob, same trick used for the 300Mbps
# by_ntracks merge) rather than trying to get streaming_sim.py to append in-place.
#
# n_tracks=4 only (the just-decided paper default), 120/180/240 tier, same 24 bicycle traces.
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash run_bw120180240_ntracks4_oracle.sh <trace...>
set -e
TRACES="$*"
PLY=/mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply
TILING_CACHE=output/oracle_tiling_cache/bicycle_8x8x8.npz
ORACLE_NPZ=output/oracle/8/eval/bicycle/oracle_dq.npz
OUT_BASE=output/0720/streaming_sim_stateful
RAW=${OUT_BASE}/_raw

for N in $TRACES; do
  TRACE=dataset/EyeNavGS_NTHU_Dataset/bicycle/user${N}_bicycle.csv
  echo "=== user${N} n_tracks=4 bw=120,180,240 oracle_loo (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}) ==="
  conda run -n gaussian_splatting python streaming_sim.py \
    --output-root "${RAW}/user${N}/n_tracks4_oracle" \
    --trace-file "${TRACE}" --ply "${PLY}" --tiling-cache "${TILING_CACHE}" \
    --bandwidths-mbps 120 180 240 \
    --interval-sec 0.3 --render-interval-sec 0.5 \
    --methods oracle_loo \
    --oracle-npz "${ORACLE_NPZ}" \
    --weight-mode screen_area --w-norm sum --c-norm sum \
    --n-tracks 4 --png-workers 4
  mkdir -p "${OUT_BASE}/n_tracks4_oracle"
  ln -sfn "../_raw/user${N}/n_tracks4_oracle" "${OUT_BASE}/n_tracks4_oracle/user${N}"
done
echo "GPU${CUDA_VISIBLE_DEVICES} bw120180240_ntracks4_oracle batch done"
