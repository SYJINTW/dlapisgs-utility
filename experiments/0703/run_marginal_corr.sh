#!/usr/bin/env bash
# Per-camera marginal-key + raw-W_k Spearman correlation vs oracle ΔMSE, all 10 scenes,
# all 150 eval cams, 8x8x8 tiling. Rerun of experiments/0618/run_marginal_corr.sh (stale
# paths: output/0606/.tiling_cache, output/0615/exp5_oracle/eval) against current canon
# paths (output/0605/exp1_gs_weights/<scene>/.tiling_cache.npz,
# output/oracle/8/eval/<scene>/oracle_dq.npz). Uses experiments/0521/diag_ck_spearman.py,
# fixed 2026-07-03 to drop the removed w_mode="sum" kwarg (w_mode option deleted from
# compute_tile_weights_and_counts 2026-07-02).
#
# Purpose: isolate ρ(W_k_raw, ΔMSE) directly -- does raw per-tile screen_area weight (opacity
# x projected area) correlate with actual oracle LOO-MSE importance, real vs synthetic scenes?
set -euo pipefail
cd "$(dirname "$0")/../.."
WORKSPACE="$PWD"
DATASET="$WORKSPACE/../exp-dataset"
TILING_BASE="$WORKSPACE/output/0605/exp1_gs_weights"
ORACLE_BASE="$WORKSPACE/output/oracle/8/eval"
OUT_ROOT="${OUTPUT_ROOT:-$WORKSPACE/output/0703/diag_marginal_corr}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENES="${SCENES:-chair drums ficus hotdog materials mic ship bicycle garden stump}"
mkdir -p "$WORKSPACE/logs" "$OUT_ROOT"

for scene in $SCENES; do
    echo "=== $scene ==="
    conda run -n gaussian_splatting python \
        "$WORKSPACE/experiments/0521/diag_ck_spearman.py" \
        --ply          "$DATASET/$scene/point_cloud.ply" \
        --tiling-cache "$TILING_BASE/$scene/.tiling_cache.npz" \
        --camera-trace "$DATASET/$scene/sparse_views_eval.json" \
        --oracle       "$ORACLE_BASE/$scene/oracle_dq.npz" \
        --img-w 1600 --img-h 1600 \
        --out-dir      "$OUT_ROOT/$scene"
done
echo "=== ALL DONE: $OUT_ROOT ==="
