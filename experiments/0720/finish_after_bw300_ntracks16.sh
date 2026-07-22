#!/bin/bash
# Waits for both bw300_ntracks16 GPU wrappers to exit, then runs finish_streaming_sim.sh
# (PSNR/SSIM plots + utilization + VMAF backfill) over the 24 n_tracks16 dirs. Only 24 dirs
# (not 120 like the earlier bw120180240 sweep) so no parallel sharding needed.
#
# Usage: finish_after_bw300_ntracks16.sh <gpu_wrapper_pid...>
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
echo "=== both bw300_ntracks16 GPU wrappers exited, starting finish_streaming_sim.sh ==="

DIRS=()
for N in $(seq 1 24); do
  DIRS+=("output/0717/streaming_sim_stateful/n_tracks16/user${N}")
done
bash experiments/finish_streaming_sim.sh "${DIRS[@]}"
echo "FINISH_AFTER_BW300_NTRACKS16_DONE"
