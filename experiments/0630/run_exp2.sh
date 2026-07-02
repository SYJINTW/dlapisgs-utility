#!/usr/bin/env bash
# Step v — exp2 redo (8³, tile_partial, 1600²). Methods:
#   vd_lod (--gs-order ply, baseline)  vd_lod_w (ours)  v_lod_w (distance-free)
#   ml × {rf,lgbm,xgb}                 oracle_loo (MSE ceiling)
# ml variants collide under scheme "ml" -> run in own subdirs, relabel at concat.
# GT rendered once per invocation (schemes sharing an invocation share GT).
#
# Env: SCENES, CUDA_VISIBLE_DEVICES (≤2 on this shared box), OUTPUT_ROOT
set -euo pipefail
cd "$(dirname "$0")/../.."
UTIL="$PWD"; ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"; DSET="$ROOT/exp-dataset"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL/output/0630/exp2}"
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
        --camera-trace "$TRACE" --grid-shape 8 8 8 --budget-pct $BUDGET_PCTS \
        --num-lod 1 --camera-index -1 --packing-mode tile_partial \
        --weight-mode screen_area --w-mode mean --img-w 1600 --img-h 1600 \
        --scene "$scene" --group-by scheme --gs-order "$gso" \
        --tiling-cache "$TILING_CACHE" "$@"
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
    # main: 3 heuristics/oracle + ml(rf), weight order, one GT render
    run_one main   weight --schemes vd_lod_w v_lod_w oracle_loo ml \
                          --oracle-npz "$ORACLE_NPZ" --ml-model-dir "$ML_DIR" --ml-model-type rf --save-rep-only
    run_one ml_lgbm weight --schemes ml --ml-model-dir "$ML_DIR" --ml-model-type lgbm --save-rep-only
    run_one ml_xgb  weight --schemes ml --ml-model-dir "$ML_DIR" --ml-model-type xgb --save-rep-only
    run_one vd_lod  ply    --schemes vd_lod --save-rep-only
done
echo "run_exp2.sh lane done: $CUDA_VISIBLE_DEVICES"
