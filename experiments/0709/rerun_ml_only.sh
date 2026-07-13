#!/usr/bin/env bash
# Rerun only the `ml` scheme (tile_partial + tile_strict, every grid/scene) after the
# ml/features.py label-censoring fix + retrained models (2026-07-09/10). vd_lod/v_lod_w/
# oracle_loo don't depend on the ml model -- their rows in the existing
# output/0704/quality_sweep/summary_all.csv are still correct and are not rerun here.
# Progressive doesn't use `ml` at all, also skipped. Fresh output root (old dirs' PLY
# skip-check would otherwise treat everything as already done).
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

REAL_SCENES="bicycle garden stump"
SYNTH_SCENES="chair drums ficus hotdog materials mic ship"
SCENES="${SCENES:-$REAL_SCENES $SYNTH_SCENES}"
GRIDS="${GRIDS:-8 1 2 4 16}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
NUM_LOD="${NUM_LOD:-1}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0709/quality_sweep_ml_retrain}"
ORACLE_ROOT="${ORACLE_ROOT:-$UTIL_DIR/output/oracle}"
ML_TYPE="${ML_TYPE:-lgbm}"
TILING_CACHE_BASE="${TILING_CACHE_BASE:-$UTIL_DIR/output/oracle_tiling_cache}"

echo "=========================================="
echo "ML-only rerun (post label-fix retrain), tile_partial+tile_strict"
echo "OUTPUT_ROOT : $OUT_BASE"
echo "SCENES      : $SCENES"
echo "GRIDS       : $GRIDS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for g in $GRIDS; do
    echo ""
    echo "====== grid=${g}x${g}x${g} ======"
    ML_DIR_G="${ML_DIR:-$UTIL_DIR/output/ml_models_experimental/pooled_all10_ACG_grid${g}}"
    echo "ML_DIR (grid${g}): $ML_DIR_G"

    for scene in $SCENES; do
        PLY="$DSET/$scene/point_cloud.ply"
        TRACE="$DSET/$scene/sparse_views_eval.json"
        TILING_CACHE="$TILING_CACHE_BASE/${scene}_${g}x${g}x${g}.npz"
        ORACLE_NPZ="$ORACLE_ROOT/$g/eval/$scene/oracle_dq.npz"

        if [[ ! -f "$PLY" ]];   then echo "[skip] $scene: no point_cloud.ply";       continue; fi
        if [[ ! -f "$TRACE" ]]; then echo "[skip] $scene: no sparse_views_eval.json"; continue; fi
        if [[ ! -f "$ORACLE_NPZ" ]]; then echo "[skip] $scene grid${g}: no oracle_dq.npz"; continue; fi

        for packing in tile_partial tile_strict; do
            OUT_DIR="$OUT_BASE/$scene/grid${g}_${packing}"
            if [[ -f "$OUT_DIR/params.yaml" ]]; then
                echo "[skip] $scene grid${g} ${packing}: already done"
                continue
            fi
            echo "---- [$scene] grid=${g}x${g}x${g} ${packing} (ml only) ----"
            mkdir -p "$OUT_DIR"
            check_gpu_free
            conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
                --ply "$PLY" --gt-ply "$PLY" \
                --output-root "$OUT_DIR" \
                --camera-trace "$TRACE" \
                --grid-shape "$g" "$g" "$g" \
                --budget-pct $BUDGET_PCTS \
                --schemes ml \
                --num-lod "$NUM_LOD" --camera-index -1 \
                --packing-mode "$packing" \
                --weight-mode screen_area \
                --w-norm sum --c-norm sum \
                --greedy-key marginal \
                --img-w 1600 --img-h 1600 \
                --scene "$scene" --group-by scheme \
                --tiling-cache "$TILING_CACHE" \
                --oracle-npz "$ORACLE_NPZ" \
                --ml-model-dir "$ML_DIR_G" --ml-model-type "$ML_TYPE" \
                --lpips --save-rep-only
        done
    done
done

SWEEP_CSV="$OUT_BASE/summary_all.csv"
echo ""
echo "Concatenating -> $SWEEP_CSV"
conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

echo ""
echo "Done. Sweep root: $OUT_BASE"
