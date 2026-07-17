#!/bin/bash
# Canonical streaming_sim post-processing: VMAF backfill + per-trace plots (raw+smoothed) +
# bandwidth-utilization diagnostic, always chained in one script so none of these are ever
# a separate, forgettable manual step.
#
# Usage: bash experiments/finish_streaming_sim.sh <dir1> <dir2> ...
set -e
cd "$(dirname "$0")/.."
bash experiments/run_vmaf_only.sh "$@"
bash experiments/plot_streaming_sim_batch.sh "$@"
bash experiments/plot_track_utilization_batch.sh "$@"
echo "FINISH_STREAMING_SIM_DONE"
