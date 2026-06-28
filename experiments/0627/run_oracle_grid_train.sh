#!/usr/bin/env bash
# Exp 4 — TRAIN oracles (for ML training; fixes eval-cam leak, 2026-06-27).
# Grid-parametrized generalization of 0626/run_oracle_grid16_train.sh.
# Mirrors 8³ run_oracle.sh TRAIN branch: --trace full --num-cameras 150 (cams 0-149),
# NO --compute-aoi (ml label = log(mse_loo) only, does not use aoi → ~half render).
# Output: output/0618/oracle_grid_train/{g}/{scene}/oracle_dq.npz  (disjoint from eval cams 150-299).
# Eval-grid tiling caches already exist at output/0606/.tiling_cache.
#
# Env: GRIDS="1 2 4 16", SCENES=..., CUDA_VISIBLE_DEVICES=...
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

GRIDS="${GRIDS:-1 2 4 16}"
SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"
TILING_CACHE_BASE="$UTIL_DIR/output/0606/.tiling_cache"
ORACLE_ROOT="$UTIL_DIR/output/0618/oracle_grid_train"

echo "=========================================="
echo "Exp 4 — TRAIN oracles (full trace, cams 0-149, NO aoi)"
echo "GRIDS       : $GRIDS"
echo "ORACLE_ROOT : $ORACLE_ROOT/{g}/{scene}/oracle_dq.npz"
echo "SCENES      : $SCENES"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

for GRID in $GRIDS; do
    ORACLE_BASE="$ORACLE_ROOT/$GRID"
    mkdir -p "$ORACLE_BASE"
    echo ""
    echo "====== grid=${GRID}x${GRID}x${GRID} ======"

    for scene in $SCENES; do
        PLY="$DSET/$scene/point_cloud.ply"
        FULL_TRACE="$DSET/$scene/sparse_views_full.json"
        TILING_CACHE="$TILING_CACHE_BASE/${scene}_${GRID}x${GRID}x${GRID}.npz"

        if [[ ! -f "$PLY" ]];          then echo "[skip] $scene: no point_cloud.ply"; continue; fi
        if [[ ! -f "$FULL_TRACE" ]];   then echo "[skip] $scene: no sparse_views_full.json"; continue; fi
        if [[ ! -f "$TILING_CACHE" ]]; then echo "[skip] $scene: no ${GRID}³ tiling cache at $TILING_CACHE"; continue; fi

        OUT_DIR="$ORACLE_BASE/$scene"
        echo "---- [$scene] TRAIN oracle ${GRID}³ (cams 0-149) ----"
        mkdir -p "$OUT_DIR"
        check_gpu_free
        conda run -n gaussian_splatting python "$UTIL_DIR/experiments/exp4_oracle_dq.py" \
            --ply "$PLY" \
            --trace "$FULL_TRACE" \
            --tiling-cache "$TILING_CACHE" \
            --output-root "$OUT_DIR" \
            --num-cameras 150 \
            --width 1600 --height 1600 \
            --flush-every 10 \
            --skip-existing
        echo "[$scene] train oracle ${GRID}³ done -> $OUT_DIR/oracle_dq.npz"
    done
done

echo ""
echo "ALL TRAIN ORACLE GRID DONE $(date)"
echo "Next: GRIDS=\"$GRIDS\" ORACLE_ROOT=output/0618/oracle_grid_train ML_OUT=output/0618/ml_models_train train_ml_per_grid.sh"
