#!/bin/bash
# Reusable per-trace plot generation: loops plot_streaming_sim.py (raw + smoothed twin,
# automatic per-invocation) over a list of output-root dirs. gaussian_splatting env.
#
# Usage: bash experiments/plot_streaming_sim_batch.sh <dir1> <dir2> ...
set -e
cd "$(dirname "$0")/.."
for d in "$@"; do
  echo "=== $d ==="
  conda run -n gaussian_splatting python experiments/plot_streaming_sim.py \
    --summary-csv "$d/metrics/summary.csv" --out-dir "$d/plots"
done
echo "PLOT_BATCH_DONE"
