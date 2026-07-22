#!/bin/bash
# Waits for both bw120180240_ntracks4_oracle GPU wrappers to exit, then runs
# finish_streaming_sim.sh over the 24 n_tracks4_oracle dirs. Only 24 dirs, no sharding needed.
#
# Usage: finish_after_ntracks4_oracle.sh <gpu_wrapper_pid...>
set -e
cd "$(dirname "$0")/../.."
GPU_PIDS=("$@")

any_alive() {
  for p in "${GPU_PIDS[@]}"; do
    kill -0 "$p" 2>/dev/null && return 0
  done
  return 1
}
while any_alive; do
  sleep 20
done
echo "=== both bw120180240_ntracks4_oracle GPU wrappers exited, starting finish_streaming_sim.sh ==="

DIRS=()
for N in $(seq 1 24); do
  DIRS+=("output/0720/streaming_sim_stateful/_raw/user${N}/n_tracks4_oracle")
done
bash experiments/finish_streaming_sim.sh "${DIRS[@]}"
echo "FINISH_AFTER_NTRACKS4_ORACLE_DONE"
