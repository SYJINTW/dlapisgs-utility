#!/usr/bin/env bash
# Three-model ML selection eval on the exp3 setting.
# Matrix: model_type {rf,lgbm,xgb} × packing {tile_strict,tile_partial} × 10 scenes.
# scheme=ml only (baselines vd_lod/oracle are model-independent — not rerun here).
# Fixed: grid 8x8x8, screen_area, 1600x1600, --save-rep-only, --ml-feature-cache auto.
# Models: output/0626/ml_models/{scene}/AC  (render-env sklearn 1.0.2 — loadable at selection).
# Separate out dir per model_type (scheme name is always "ml" → would collide).
#
# Env overrides: SCENES, OUTPUT_ROOT, BUDGET_PCTS, CUDA_VISIBLE_DEVICES
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
DSET="$ROOT/exp-dataset"
MODEL_ROOT="$UTIL_DIR/output/0626/ml_models_train"   # trained on TRAIN oracle (sparse_views_full); eval scores on disjoint eval cams
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
PACKINGS="${PACKINGS:-tile_strict tile_partial}"
MODELS="${MODELS:-rf lgbm xgb}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0626/exp3_3model_clean}"

echo "=========================================="
echo "Exp3 3-model @1600² — {rf,lgbm,xgb} × {tile_strict,tile_partial}"
echo "OUTPUT_ROOT : $OUT_BASE"
echo "SCENES      : $SCENES"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="
mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    TRACE="$DSET/$scene/sparse_views_eval.json"
    ML_DIR="$MODEL_ROOT/$scene/AC"
    TILING_CACHE="$OUT_BASE/$scene/.tiling_cache.npz"

    if [[ ! -f "$PLY" ]];    then echo "[skip] $scene: no point_cloud.ply";       continue; fi
    if [[ ! -f "$TRACE" ]];  then echo "[skip] $scene: no sparse_views_eval.json"; continue; fi
    if [[ ! -d "$ML_DIR" ]]; then echo "[skip] $scene: no ML model dir $ML_DIR";   continue; fi
    mkdir -p "$(dirname "$TILING_CACHE")"

    for packing in $PACKINGS; do
        # ── oracle_loo ceiling (model-independent; eval-cam LOO ground truth) ──
        ORACLE_NPZ="$UTIL_DIR/output/0606/exp4_oracle_dq/eval/$scene/oracle_dq.npz"
        if [[ -f "$ORACLE_NPZ" ]]; then
            OUT_DIR="$OUT_BASE/$scene/${packing}_oracle"
            echo "---- [$scene] $packing oracle_loo ----"
            mkdir -p "$OUT_DIR"; check_gpu_free
            conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
                --ply "$PLY" --gt-ply "$PLY" --output-root "$OUT_DIR" \
                --camera-trace "$TRACE" --grid-shape 8 8 8 --budget-pct $BUDGET_PCTS \
                --schemes oracle_loo --num-lod 1 --camera-index -1 \
                --packing-mode "$packing" --greedy-key marginal \
                --weight-mode screen_area --w-mode mean --img-w 1600 --img-h 1600 \
                --scene "$scene" --group-by scheme --tiling-cache "$TILING_CACHE" \
                --oracle-npz "$ORACLE_NPZ" --save-rep-only
            conda run -n gsquic python "$UTIL_DIR/experiments/aggregate_timings.py" \
                --output-root "$OUT_DIR" || true
        else
            echo "[skip] $scene: no oracle_dq.npz at $ORACLE_NPZ"
        fi

        # ── ml: 3 models ──────────────────────────────────────────────────────
        for mt in $MODELS; do
            OUT_DIR="$OUT_BASE/$scene/${packing}_${mt}"
            echo "---- [$scene] $packing $mt ----"
            mkdir -p "$OUT_DIR"
            check_gpu_free

            conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
                --ply "$PLY" \
                --gt-ply "$PLY" \
                --output-root "$OUT_DIR" \
                --camera-trace "$TRACE" \
                --grid-shape 8 8 8 \
                --budget-pct $BUDGET_PCTS \
                --schemes ml \
                --num-lod 1 \
                --camera-index -1 \
                --packing-mode "$packing" \
                --greedy-key marginal \
                --weight-mode screen_area \
                --w-mode mean \
                --img-w 1600 --img-h 1600 \
                --scene "$scene" \
                --group-by scheme \
                --tiling-cache "$TILING_CACHE" \
                --ml-model-dir "$ML_DIR" \
                --ml-model-type "$mt" \
                --ml-feature-cache auto \
                --save-rep-only

            conda run -n gsquic python "$UTIL_DIR/experiments/aggregate_timings.py" \
                --output-root "$OUT_DIR" || true
        done
    done
done

# ── Concat sweep-level quality CSV ────────────────────────────────────────────
SWEEP_CSV="$OUT_BASE/summary_all.csv"
echo "Concatenating -> $SWEEP_CSV"
conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

echo "ALL_EVAL_DONE. Sweep root: $OUT_BASE"
