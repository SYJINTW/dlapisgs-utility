#!/usr/bin/env bash
# Step iii — MSE-oracle vs SSIM-oracle selection compare (8³, tile_partial).
# Same eval oracle NPZ feeds both schemes (oracle_loo uses mse, oracle_loo_ssim uses ssim).
# Eval PSNR/SSIM vs budget. Saves representative renders for qualitative grounding.
#
# Env overrides:
#   SCENES, CUDA_VISIBLE_DEVICES (≤2 GPUs on this shared box), OUTPUT_ROOT
set -euo pipefail
cd "$(dirname "$0")/../.."

UTIL="$PWD"
ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
DSET="$ROOT/exp-dataset"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL/output/0630/oracle_compare}"
TILING_BASE="$UTIL/output/0605/exp1_gs_weights"
ORACLE_BASE="$UTIL/output/oracle/8/eval"

echo "OUT=$OUT_BASE  SCENES=$SCENES  CUDA=$CUDA_VISIBLE_DEVICES"
mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    TRACE="$DSET/$scene/sparse_views_eval.json"
    ORACLE_NPZ="$ORACLE_BASE/$scene/oracle_dq.npz"
    TILING_CACHE="$TILING_BASE/$scene/.tiling_cache.npz"
    OUT_DIR="$OUT_BASE/$scene"
    [ -f "$PLY" ]          || { echo "[skip] $scene: no PLY"; continue; }
    [ -f "$ORACLE_NPZ" ]   || { echo "[skip] $scene: no oracle"; continue; }
    [ -f "$TILING_CACHE" ] || { echo "[skip] $scene: no tiling"; continue; }
    [ -f "$OUT_DIR/metrics/summary.csv" ] && { echo "[skip] $scene: done"; continue; }

    echo "---- [$scene] ----"
    mkdir -p "$OUT_DIR"
    conda run -n gaussian_splatting python "$UTIL/test_utility_inmem.py" \
        --ply "$PLY" --gt-ply "$PLY" --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" --grid-shape 8 8 8 \
        --budget-pct $BUDGET_PCTS \
        --schemes oracle_loo oracle_loo_ssim \
        --num-lod 1 --camera-index -1 \
        --packing-mode tile_partial \
        --weight-mode screen_area --w-mode mean \
        --img-w 1600 --img-h 1600 --scene "$scene" \
        --group-by scheme --tiling-cache "$TILING_CACHE" \
        --oracle-npz "$ORACLE_NPZ" --save-rep-only
done

echo "Concatenating + plotting per scene..."
for scene in $SCENES; do
    SCENE_CSV="$OUT_BASE/$scene/summary_scene.csv"
    conda run -n gsquic python "$UTIL/experiments/concat_summaries.py" \
        --glob "$OUT_BASE/$scene/metrics/summary.csv" --out "$SCENE_CSV" || true
    [ -f "$SCENE_CSV" ] && conda run -n gsquic python "$UTIL/experiments/plot_metrics.py" \
        --summary-csv "$SCENE_CSV" --out-dir "$OUT_BASE/$scene/plots" \
        --group-by scheme --title-suffix "$scene (oracle MSE vs SSIM, tile_partial)" || true
done
conda run -n gsquic python "$UTIL/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/metrics/summary.csv" --out "$OUT_BASE/summary_all.csv" || true
echo "Done. Root: $OUT_BASE"
