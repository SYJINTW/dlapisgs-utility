#!/usr/bin/env bash
# Per-camera marginal-key Spearman correlation diagnostic (all 10 scenes, 8x8x8 tiling).
# Rewrites diag_ck_spearman.py; see exp_marginal_corr.md for spec.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="$WORKSPACE/../exp-dataset"
TILING_CACHE="$WORKSPACE/output/0606/.tiling_cache"
ORACLE_ROOT="$WORKSPACE/output/0615/exp5_oracle/eval"
OUT_ROOT="$WORKSPACE/output/diag_marginal_corr"
LOG="$WORKSPACE/logs/diag_marginal_corr.log"

mkdir -p "$WORKSPACE/logs"

SCENES=(chair drums ficus hotdog materials mic ship bicycle garden stump)

run_scene() {
    local scene="$1"
    echo "=== $scene ==="
    conda run -n gaussian_splatting python \
        "$WORKSPACE/experiments/0521/diag_ck_spearman.py" \
        --ply          "$DATASET/$scene/point_cloud.ply" \
        --tiling-cache "$TILING_CACHE/${scene}_8x8x8.npz" \
        --camera-trace "$DATASET/$scene/sparse_views_eval.json" \
        --oracle       "$ORACLE_ROOT/$scene/oracle_dq.npz" \
        --img-w 1600 --img-h 1600 \
        --out-dir      "$OUT_ROOT/$scene"
}

for scene in "${SCENES[@]}"; do
    run_scene "$scene"
done

echo "=== ALL DONE ==="
echo "Results in $OUT_ROOT"
