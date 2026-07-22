#!/bin/bash
# Canonical streaming_sim post-processing: per-trace plots (raw+smoothed) + bandwidth-
# utilization diagnostic + VMAF backfill, always chained in one script so none of these are
# ever a separate, forgettable manual step.
#
# PSNR/SSIM plots and the utilization diagnostic only need summary.csv/schedule JSON, both
# already written by run_sweep -- they don't need VMAF. Run them FIRST so partial results
# (PSNR/SSIM) are available immediately instead of waiting on the slow VMAF stage, then run
# VMAF backfill, then re-plot (adds the vmaf_vs_time panel; psnr/ssim panels are cheap to
# regenerate, unchanged).
#
# Usage: bash experiments/finish_streaming_sim.sh <dir1> <dir2> ...
set -e
cd "$(dirname "$0")/.."
bash experiments/plot_streaming_sim_batch.sh "$@"
bash experiments/plot_track_utilization_batch.sh "$@"
bash experiments/run_vmaf_only.sh "$@"
bash experiments/plot_streaming_sim_batch.sh "$@"
echo "FINISH_STREAMING_SIM_DONE"
