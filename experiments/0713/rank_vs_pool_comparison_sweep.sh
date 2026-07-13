#!/usr/bin/env bash
# 5-arm scoped selection-PSNR sweep comparing domain-split pooled regressors and the
# revisited LGBMRanker candidate against the deployed pooled_all10_ACG regressor (that arm's
# rows come from the existing output/0709/quality_sweep_ml_retrain/ sweep, not rerun here --
# no code changed in the regressor path since 07-09). Grid8/tile_partial only, per the
# approved comparison scope -- see .claude/plans equivalent doc "declarative-beaming-cascade".
#
# Rank arms use --ml-model-type lgbm_rank; --greedy-key marginal is passed uniformly below,
# the auto-override to "utility" (pure rank-order sort, no byte division) for lgbm_rank
# happens transparently inside test_utility_inmem.py -- no branching needed here.
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
DSET="$ROOT/exp-dataset"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # GPU0 banned

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
GRID=8
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
NUM_LOD="${NUM_LOD:-1}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0713/rank_vs_pool_comparison}"
ORACLE_ROOT="${ORACLE_ROOT:-$UTIL_DIR/output/oracle}"
TILING_CACHE_BASE="${TILING_CACHE_BASE:-$UTIL_DIR/output/oracle_tiling_cache}"
MODELS_BASE="$UTIL_DIR/output/ml_models_experimental"

# arm_name|scenes|ml_model_dir|ml_model_type
ARMS=(
    "real3_reg|$REAL_SCENES|$MODELS_BASE/pooled_real3_ACG|lgbm"
    "synth7_reg|$SYNTH_SCENES|$MODELS_BASE/pooled_synth7_ACG|lgbm"
    "all10_rank|$REAL_SCENES $SYNTH_SCENES|$MODELS_BASE/pooled_rank_all10|lgbm_rank"
    "real3_rank|$REAL_SCENES|$MODELS_BASE/pooled_rank_real3|lgbm_rank"
    "synth7_rank|$SYNTH_SCENES|$MODELS_BASE/pooled_rank_synth7|lgbm_rank"
)

echo "=========================================="
echo "Rank-vs-pool comparison sweep, grid8/tile_partial, 5 arms"
echo "OUTPUT_ROOT : $OUT_BASE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for arm_spec in "${ARMS[@]}"; do
    IFS='|' read -r arm_name arm_scenes arm_model_dir arm_model_type <<< "$arm_spec"
    echo ""
    echo "====== arm=${arm_name} model_dir=${arm_model_dir} model_type=${arm_model_type} ======"

    if [[ ! -d "$arm_model_dir" ]]; then
        echo "[FATAL] $arm_name: model dir missing: $arm_model_dir"
        exit 1
    fi

    ARM_OUT="$OUT_BASE/$arm_name"
    for scene in $arm_scenes; do
        PLY="$DSET/$scene/point_cloud.ply"
        TRACE="$DSET/$scene/sparse_views_eval.json"
        TILING_CACHE="$TILING_CACHE_BASE/${scene}_${GRID}x${GRID}x${GRID}.npz"
        ORACLE_NPZ="$ORACLE_ROOT/$GRID/eval/$scene/oracle_dq.npz"

        if [[ ! -f "$PLY" ]];   then echo "[skip] $scene: no point_cloud.ply";       continue; fi
        if [[ ! -f "$TRACE" ]]; then echo "[skip] $scene: no sparse_views_eval.json"; continue; fi
        if [[ ! -f "$ORACLE_NPZ" ]]; then echo "[skip] $scene grid${GRID}: no oracle_dq.npz"; continue; fi

        OUT_DIR="$ARM_OUT/$scene/tile_partial"
        if [[ -f "$OUT_DIR/params.yaml" ]]; then
            echo "[skip] $arm_name/$scene: already done"
            continue
        fi
        echo "---- [$arm_name] $scene tile_partial ----"
        mkdir -p "$OUT_DIR"
        check_gpu_free
        conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
            --ply "$PLY" --gt-ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape "$GRID" "$GRID" "$GRID" \
            --budget-pct $BUDGET_PCTS \
            --schemes ml \
            --num-lod "$NUM_LOD" --camera-index -1 \
            --packing-mode tile_partial \
            --weight-mode screen_area \
            --w-norm sum --c-norm sum \
            --greedy-key marginal \
            --img-w 1600 --img-h 1600 \
            --scene "$scene" --group-by scheme \
            --tiling-cache "$TILING_CACHE" \
            --oracle-npz "$ORACLE_NPZ" \
            --ml-model-dir "$arm_model_dir" --ml-model-type "$arm_model_type" \
            --lpips --save-rep-only
    done

    SWEEP_CSV="$ARM_OUT/summary_all.csv"
    echo "Concatenating -> $SWEEP_CSV"
    conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
        --glob "$ARM_OUT/*/*/metrics/summary.csv" \
        --out "$SWEEP_CSV"
done

echo ""
echo "Done. Comparison root: $OUT_BASE"
