#!/bin/bash
# Reusable VMAF backfill: loops streaming_sim.py --vmaf-only (threaded CPU, vmaf_4k_v0.6.1,
# the only profile this project uses) over a list of output-root dirs that already have
# vmaf/gt.yuv + vmaf/bw_*mbps/*/distorted.yuv from a prior run_sweep. gaussian_splatting env
# only -- run_vmaf() lives in streaming_sim.py, not portable to gsquic.
#
# Usage: bash experiments/run_vmaf_only.sh <dir1> <dir2> ...
#        bash experiments/run_vmaf_only.sh $(cat dirs.txt)
set -e
cd "$(dirname "$0")/.."
for d in "$@"; do
  echo "=== $d ==="
  conda run -n gaussian_splatting python streaming_sim.py --output-root "$d" --vmaf-only
done
echo "VMAF_ONLY_BATCH_DONE"
