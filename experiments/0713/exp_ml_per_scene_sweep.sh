#!/usr/bin/env bash
# Part C sweep: individual per-scene LGBM/AC models, grid8/tile_partial/screen_area only
# (matches the two reused pooled arms' slice). Structural template: experiments/0709/
# rerun_ml_only.sh -- but that script shares ONE --ml-model-dir per grid across all scenes;
# per-scene models need the model dir INSIDE the scene loop instead, so this is a new
# script, not a re-invocation.
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
DSET="$ROOT/exp-dataset"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MIN_FREE_MIB="${MIN_FREE_MIB:-8192}"
check_gpu_free() {
    local gpu_idx="${CUDA_VISIBLE_DEVICES%%,*}"
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null)
    if [[ -z "$free_mib" || "$free_mib" -lt "$MIN_FREE_MIB" ]]; then
        echo "[GPU GUARD] GPU $gpu_idx: ${free_mib:-?} MiB free (need >=${MIN_FREE_MIB}). Aborting."
        return 1
    fi
    echo "[GPU GUARD] GPU $gpu_idx: ${free_mib} MiB free — OK"
}

SCENES="bicycle garden stump chair drums ficus hotdog materials mic ship"
BUDGET_PCTS="10 25 40 55 70 85 99 100"
OUT_BASE="$UTIL_DIR/output/0713/exp_ml_per_scene"
ORACLE_ROOT="$UTIL_DIR/output/oracle/8/eval"
TILING_CACHE_BASE="$UTIL_DIR/output/oracle_tiling_cache"
MODEL_BASE="$UTIL_DIR/output/ml_models_experimental/per_scene"

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    TRACE="$DSET/$scene/sparse_views_eval.json"
    TILING_CACHE="$TILING_CACHE_BASE/${scene}_8x8x8.npz"
    ORACLE_NPZ="$ORACLE_ROOT/$scene/oracle_dq.npz"
    ML_DIR="$MODEL_BASE/$scene/AC"

    if [[ ! -f "$ML_DIR/lgbm.pkl" ]]; then
        echo "[skip] $scene: no trained model at $ML_DIR (run train_per_scene_all.sh first)"
        continue
    fi

    OUT_DIR="$OUT_BASE/$scene/grid8_tile_partial"
    if [[ -f "$OUT_DIR/params.yaml" ]]; then
        echo "[skip] $scene: already done"
        continue
    fi
    echo "---- [$scene] grid8 tile_partial (per-scene ml) ----"
    mkdir -p "$OUT_DIR"
    check_gpu_free
    conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
        --ply "$PLY" --gt-ply "$PLY" \
        --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" \
        --grid-shape 8 8 8 \
        --budget-pct $BUDGET_PCTS \
        --schemes ml \
        --num-lod 1 --camera-index -1 \
        --packing-mode tile_partial \
        --weight-mode screen_area \
        --w-norm sum --c-norm sum \
        --greedy-key marginal \
        --img-w 1600 --img-h 1600 \
        --scene "$scene" --group-by scheme \
        --tiling-cache "$TILING_CACHE" \
        --oracle-npz "$ORACLE_NPZ" \
        --ml-model-dir "$ML_DIR" --ml-model-type lgbm \
        --lpips --save-rep-only
done

SWEEP_CSV="$OUT_BASE/summary_all.csv"
echo ""
echo "Concatenating -> $SWEEP_CSV"
conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

echo ""
echo "Done. Sweep root: $OUT_BASE"
