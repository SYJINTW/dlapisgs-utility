#!/bin/bash
# Waits for both raw-sweep GPU wrappers (bw120180240_ntracks124816) to exit, then runs
# finish_streaming_sim.sh (VMAF backfill + per-trace plots + utilization diagnostic) over
# all 120 new dirs (24 users x n_tracks{1,2,4,8,16}), 4-way parallel sharded like the
# n_tracks5-8 backfill earlier this session -- see .claude/PLAN.md "streaming_sim needs
# finish_streaming_sim.sh" rule: VMAF is never inline, always chain this automatically,
# don't wait to be asked.
set -e
cd "$(dirname "$0")/../.."
GPU1_PID=$1
GPU2_PID=$2

while kill -0 "$GPU1_PID" 2>/dev/null || kill -0 "$GPU2_PID" 2>/dev/null; do
  sleep 30
done
echo "=== both raw-sweep GPU jobs exited, starting VMAF/plots backfill ==="

OUT_BASE=output/0720/streaming_sim_stateful
RAW=${OUT_BASE}/_raw

# 4-way shard over users 1-24 x n_tracks{1,2,4,8,16}
declare -a SHARD1 SHARD2 SHARD3 SHARD4
i=0
for N in $(seq 1 24); do
  for NT in 1 2 4 8 16; do
    d="${RAW}/user${N}/n_tracks${NT}"
    case $((i % 4)) in
      0) SHARD1+=("$d") ;;
      1) SHARD2+=("$d") ;;
      2) SHARD3+=("$d") ;;
      3) SHARD4+=("$d") ;;
    esac
    i=$((i+1))
  done
done

nohup bash experiments/finish_streaming_sim.sh "${SHARD1[@]}" > logs/0720_finish_bw120180240_shard1.log 2>&1 &
nohup bash experiments/finish_streaming_sim.sh "${SHARD2[@]}" > logs/0720_finish_bw120180240_shard2.log 2>&1 &
nohup bash experiments/finish_streaming_sim.sh "${SHARD3[@]}" > logs/0720_finish_bw120180240_shard3.log 2>&1 &
nohup bash experiments/finish_streaming_sim.sh "${SHARD4[@]}" > logs/0720_finish_bw120180240_shard4.log 2>&1 &
wait
echo "FINISH_AFTER_SWEEP_ALL_SHARDS_DONE"
