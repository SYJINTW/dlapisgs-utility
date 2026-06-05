#!/usr/bin/env bash
# Experiment 2 — three-way comparison: formula baseline / ML / oracle ceiling.
# Fixed:  packing=tile_strict, weight_mode=screen_area, grid=8x8x8, img=1600x1600.
# Schemes: vd_lod  vd_lod_w  ml  oracle_loo
# Scenes:  all 10. Skip guard skips any scene missing oracle_dq.npz or ML model.
#
# Env overrides:
#   SCENES="hotdog bicycle"
#   CUDA_VISIBLE_DEVICES=0
#   OUTPUT_ROOT=.../output/MMDD/exp2_schemes
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
DSET="$ROOT/exp-dataset"
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
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0605/exp2_schemes}"

echo "=========================================="
echo "Exp 2 @1600² — three-way scheme comparison (tile_strict, screen_area)"
echo "OUTPUT_ROOT  : $OUT_BASE"
echo "SCENES       : $SCENES"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    TRACE="$DSET/$scene/sparse_views_eval.json"
    GT_DIR="$DSET/$scene/gt_renders_eval"
    ORACLE_NPZ="$UTIL_DIR/output/0527_clean/exp4_oracle_dq/eval/$scene/oracle_dq.npz"
    ML_DIR="$UTIL_DIR/ml/models_clean/$scene/AC"
    TILING_CACHE="$OUT_BASE/$scene/.tiling_cache.npz"
    OUT_DIR="$OUT_BASE/$scene"

    if [[ ! -f "$PLY" ]];       then echo "[skip] $scene: no point_cloud.ply";        continue; fi
    if [[ ! -f "$TRACE" ]];     then echo "[skip] $scene: no sparse_views_eval.json";  continue; fi
    if [[ ! -d "$GT_DIR" ]];    then echo "[skip] $scene: no gt_renders_eval/";        continue; fi
    if [[ ! -f "$ORACLE_NPZ" ]]; then echo "[skip] $scene: no oracle_dq.npz";          continue; fi
    if [[ ! -d "$ML_DIR" ]];    then echo "[skip] $scene: no ML model dir";            continue; fi

    echo ""
    echo "---- [$scene] ----"
    mkdir -p "$OUT_DIR"
    check_gpu_free

    conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
        --ply "$PLY" \
        --gt-ply "$PLY" \
        --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" \
        --grid-shape $GRID_SHAPE \
        --budget-pct $BUDGET_PCTS \
        --schemes vd_lod vd_lod_w ml oracle_loo \
        --num-lod "$NUM_LOD" \
        --camera-index -1 \
        --packing-mode tile_strict \
        --weight-mode screen_area \
        --img-w 1600 --img-h 1600 \
        --scene "$scene" \
        --group-by scheme \
        --tiling-cache "$TILING_CACHE" \
        --gt-renders-cache "$GT_DIR" \
        --oracle-npz "$ORACLE_NPZ" \
        --ml-model-dir "$ML_DIR" \
        --save-rep-only

    conda run -n gsquic python "$UTIL_DIR/experiments/aggregate_timings.py" \
        --output-root "$OUT_DIR" || true
done

SWEEP_CSV="$OUT_BASE/summary_all.csv"
echo ""
echo "Concatenating -> $SWEEP_CSV"
conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

for scene in $SCENES; do
    SCENE_CSV="$OUT_BASE/$scene/summary_scene.csv"
    conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
        --glob "$OUT_BASE/$scene/metrics/summary.csv" \
        --out "$SCENE_CSV" || true
    [[ -f "$SCENE_CSV" ]] && conda run -n gsquic python "$UTIL_DIR/experiments/plot_metrics.py" \
        --summary-csv "$SCENE_CSV" \
        --out-dir "$OUT_BASE/$scene/plots" \
        --group-by scheme \
        --title-suffix "$scene (Exp2 @1600² tile_strict screen_area)" || true
done

echo ""
echo "Done. Sweep root: $OUT_BASE"
