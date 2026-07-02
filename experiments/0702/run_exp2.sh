#!/usr/bin/env bash
# exp2 rerun (8³, tile_partial, 1600²), superseding 0630/run_exp2.sh.
#
# Why rerun: 0630/run_exp2.sh hardcoded --w-mode mean, silently overriding the sum
# canon fixed 2026-06-28. w_mode=mean was removed from the codebase entirely
# 2026-07-02 (utility_calculation.py W_MODES now ("sum",) only, --w-mode CLI flag
# deleted) -- this script relies on the tool's own current defaults instead of
# restating them, so a future default change can't silently diverge from what
# actually ran the way the mean/sum bug did.
#
# Every flag below that is omitted matches test_utility_inmem.py's own current
# default -- verified against its argparse (grid-shape=[8,8,8], num-lod=1,
# img-w/h=1600, w-norm=c-norm=sum, packing-mode=tile_partial, gs-order=weight,
# weight-mode=screen_area, ml-model-type=lgbm, group-by=scheme). Only
# --camera-index -1 (default is single-camera 0) and --budget-pct (no sane
# default) are ever passed; --gs-order ply is the one deliberate override, for
# vd_lod only (baseline has no per-GS weight signal to order by).
#
# Schemes: vd_lod (baseline, gs-order ply) | vd_lod_w (heuristic) |
# v_lod_w (heuristic, no distance) | ml (LGBM, canon default) |
# oracle_loo (offline LOO-MSE labels, output/oracle/8/eval/<scene>/oracle_dq.npz --
# already computed, never recomputed here). vd_lod_w/v_lod_w/ml/oracle_loo share
# gs-order=weight and no scheme-name collisions (single ml model type) so they run
# in one invocation per scene; vd_lod is a separate invocation for gs-order=ply.
#
# Env: SCENES, CUDA_VISIBLE_DEVICES, OUTPUT_ROOT
set -euo pipefail
cd "$(dirname "$0")/../.."
UTIL="$PWD"; ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"; DSET="$ROOT/exp-dataset"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL/output/0702/exp2}"
TILING_BASE="$UTIL/output/0605/exp1_gs_weights"
ORACLE_BASE="$UTIL/output/oracle/8/eval"
ML_BASE="$UTIL/output/ml_models/8"
mkdir -p "$OUT_BASE"
echo "OUT=$OUT_BASE SCENES=$SCENES CUDA=$CUDA_VISIBLE_DEVICES"

run_one() {  # $1=out_subdir $2=gs_order $3..=extra args (schemes etc.)
    local sub="$1"; shift; local gso="$1"; shift
    local od="$OUT_DIR/$sub"
    [ -f "$od/metrics/summary.csv" ] && { echo "  [skip] $sub done"; return; }
    mkdir -p "$od"
    conda run -n gaussian_splatting python "$UTIL/test_utility_inmem.py" \
        --ply "$PLY" --gt-ply "$PLY" --output-root "$od" \
        --camera-trace "$TRACE" --budget-pct $BUDGET_PCTS \
        --camera-index -1 --gs-order "$gso" \
        --scene "$scene" --tiling-cache "$TILING_CACHE" "$@"
}

for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    TRACE="$DSET/$scene/sparse_views_eval.json"
    ORACLE_NPZ="$ORACLE_BASE/$scene/oracle_dq.npz"
    TILING_CACHE="$TILING_BASE/$scene/.tiling_cache.npz"
    ML_DIR="$ML_BASE/$scene/AC"
    OUT_DIR="$OUT_BASE/$scene"
    [ -f "$PLY" ] && [ -f "$ORACLE_NPZ" ] && [ -f "$TILING_CACHE" ] && [ -d "$ML_DIR" ] || { echo "[skip] $scene: missing input"; continue; }
    echo "---- [$scene] ----"; mkdir -p "$OUT_DIR"
    run_one main   weight --schemes vd_lod_w v_lod_w ml oracle_loo \
                          --oracle-npz "$ORACLE_NPZ" --ml-model-dir "$ML_DIR" --save-rep-only
    run_one vd_lod ply    --schemes vd_lod --save-rep-only
done
echo "run_exp2.sh lane done: $CUDA_VISIBLE_DEVICES"
