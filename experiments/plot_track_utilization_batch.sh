#!/bin/bash
# Reusable bandwidth-utilization diagnostic: loops plot_track_utilization.py (Gantt +
# achieved-vs-nominal throughput, from the exact schedule/*.json dumps) over a list of
# output-root dirs. Sanity-checks the stateful scheduler isn't pathologically stalling a
# track while unclaimed tiles remain -- chained into finish_streaming_sim.sh (2026-07-16)
# so this check is automatic for every sweep, not a manual afterthought.
#
# Usage: bash experiments/plot_track_utilization_batch.sh <dir1> <dir2> ...
set -e
cd "$(dirname "$0")/.."
for d in "$@"; do
  if [ -d "$d/schedule" ]; then
    echo "=== $d ==="
    conda run -n gaussian_splatting python experiments/plot_track_utilization.py \
      --schedule-json "$d"/schedule/*.json --out-dir "$d/plots/utilization"
  fi
done
echo "UTILIZATION_BATCH_DONE"
