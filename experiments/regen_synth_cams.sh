#!/usr/bin/env bash
# Regen 300-view full-sphere traces for all 7 synth scenes with min_edge_var=0.
# Overwrites existing sparse_views_300_*_sphere.json + gt_renders_300_sphere/.
# Rebuilds eval JSON + eval symlinks via --eval-start 150.
#
# Run detached:
#   screen -dmS regen_cams bash experiments/regen_synth_cams.sh
#   screen -r regen_cams
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL="$ROOT/dlapisgs-utility"
EXP="$ROOT/exp-dataset"
SCRIPT="$UTIL/experiments/gen_sparse_views.py"
LOG_DIR="$UTIL/logs/regen_synth_cams"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
CONDA_ENV="gaussian_splatting"

# Per-scene params matching original generation (only min_edge_var changed to 0)
declare -A MAX_BF=( [chair]=0.50 [drums]=0.50 [ficus]=0.95 [hotdog]=0.95 [materials]=0.95 [mic]=0.95 [ship]=0.50 )

SCENES="chair drums ficus hotdog materials mic ship"

for scene in $SCENES; do
    ckpt="$EXP/$scene/checkpoint/point_cloud/iteration_30000"
    ply="$ckpt/point_cloud.ply"
    out="$ckpt/sparse_views_300_${scene}_sphere.json"
    gt_dir="$ckpt/gt_renders_300_sphere"
    log="$LOG_DIR/${scene}.log"
    mbf="${MAX_BF[$scene]}"

    echo "=============================="
    echo "[$scene]  ($(date +%H:%M:%S))"
    echo "  out: $out"
    echo "  log: $log"
    echo "=============================="

    conda run -n "$CONDA_ENV" python "$SCRIPT" \
        --ply "$ply" \
        --out "$out" \
        --gt-renders-dir "$gt_dir" \
        --n-views 300 \
        --width 1600 --height 1600 \
        --scene-type synthetic \
        --full-sphere \
        --render-check \
        --min-edge-var 0 \
        --max-white-frac 0.35 \
        --max-black-frac "$mbf" \
        --seed 3 \
        --max-proposals 50000 \
        --eval-start 150 \
        > "$log" 2>&1

    echo "[$scene] done  ($(date +%H:%M:%S))  log=$log"
    # Print final lines from log
    tail -6 "$log"
    echo ""
done

echo "All 7 scenes done. ($(date +%H:%M:%S))"
echo "Logs: $LOG_DIR/"
