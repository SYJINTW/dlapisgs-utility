#!/usr/bin/env bash
# Experiment 4 — grid-size ablation.
# Fixed:  weight_mode=screen_area, img=1600x1600.
# Swept:  grid ∈ {1x1x1, 2x2x2, 4x4x4, 8x8x8} × packing ∈ {tile_strict, progressive}.
# Scenes: all 10.
#   tile_strict: schemes = vd_lod oracle_loo ml.
#     Requires oracle NPZ per grid size at: $ORACLE_ROOT/{g}/{scene}/oracle_dq.npz
#     and ML model at:                      $ML_ROOT/{scene}/AC/
#     Skip scene if either missing.
#   progressive: scheme = vd_lod (scheme-invariant; shows pure culling granularity effect).
#
# Prereq: run exp4_oracle_dq.py at each grid size to produce oracle NPZs.
#
# Env overrides:
#   SCENES="chair bicycle"
#   GRIDS="2 4 8"
#   CUDA_VISIBLE_DEVICES=0
#   OUTPUT_ROOT=.../output/MMDD/exp4_grid
#   ORACLE_ROOT=.../output/MMDD/oracle_grid   (expects {ORACLE_ROOT}/{g}/{scene}/oracle_dq.npz)
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
GRIDS="${GRIDS:-1 2 4 8}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
NUM_LOD="${NUM_LOD:-1}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0605/exp4_grid}"
ORACLE_ROOT="${ORACLE_ROOT:-$UTIL_DIR/output/0605/oracle_grid}"
ML_ROOT="${ML_ROOT:-$UTIL_DIR/ml/models_clean}"

echo "=========================================="
echo "Exp 4 @1600² — grid-size ablation (tile_strict + progressive)"
echo "OUTPUT_ROOT  : $OUT_BASE"
echo "ORACLE_ROOT  : $ORACLE_ROOT/{g}/{scene}/oracle_dq.npz"
echo "SCENES       : $SCENES"
echo "GRIDS        : $GRIDS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for g in $GRIDS; do
    echo ""
    echo "====== grid=${g}x${g}x${g} ======"

    for scene in $SCENES; do
        PLY="$DSET/$scene/point_cloud.ply"
        TRACE="$DSET/$scene/sparse_views_eval.json"
        GT_DIR="$DSET/$scene/gt_renders_eval"
        TILING_CACHE="$OUT_BASE/$scene/.tiling_cache_${g}.npz"

        if [[ ! -f "$PLY" ]];    then echo "[skip] $scene: no point_cloud.ply";       continue; fi
        if [[ ! -f "$TRACE" ]];  then echo "[skip] $scene: no sparse_views_eval.json"; continue; fi
        if [[ ! -d "$GT_DIR" ]]; then echo "[skip] $scene: no gt_renders_eval/";       continue; fi

        mkdir -p "$(dirname "$TILING_CACHE")"

        # ── progressive ──────────────────────────────────────────────────────
        OUT_DIR="$OUT_BASE/$scene/grid${g}_progressive"
        echo "---- [$scene] grid=${g}x${g}x${g} progressive ----"
        mkdir -p "$OUT_DIR"
        check_gpu_free

        conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
            --ply "$PLY" \
            --gt-ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape "$g" "$g" "$g" \
            --budget-pct $BUDGET_PCTS \
            --schemes vd_lod \
            --num-lod "$NUM_LOD" \
            --camera-index -1 \
            --packing-mode progressive \
            --weight-mode screen_area \
            --img-w 1600 --img-h 1600 \
            --scene "$scene" \
            --group-by grid_shape \
            --tiling-cache "$TILING_CACHE" \
            --gt-renders-cache "$GT_DIR" \
            --save-rep-only

        # ── tile_strict ───────────────────────────────────────────────────────
        ORACLE_NPZ="$ORACLE_ROOT/$g/$scene/oracle_dq.npz"
        ML_DIR="$ML_ROOT/$scene/AC"

        if [[ ! -f "$ORACLE_NPZ" ]]; then echo "[skip tile_strict] $scene grid${g}: no oracle_dq.npz at $ORACLE_NPZ"; continue; fi
        if [[ ! -d "$ML_DIR" ]];     then echo "[skip tile_strict] $scene: no ML model dir"; continue; fi

        OUT_DIR="$OUT_BASE/$scene/grid${g}_tile_strict"
        echo "---- [$scene] grid=${g}x${g}x${g} tile_strict ----"
        mkdir -p "$OUT_DIR"
        check_gpu_free

        conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
            --ply "$PLY" \
            --gt-ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape "$g" "$g" "$g" \
            --budget-pct $BUDGET_PCTS \
            --schemes vd_lod oracle_loo ml \
            --num-lod "$NUM_LOD" \
            --camera-index -1 \
            --packing-mode tile_strict \
            --weight-mode screen_area \
            --img-w 1600 --img-h 1600 \
            --scene "$scene" \
            --group-by grid_shape \
            --tiling-cache "$TILING_CACHE" \
            --gt-renders-cache "$GT_DIR" \
            --oracle-npz "$ORACLE_NPZ" \
            --ml-model-dir "$ML_DIR" \
            --save-rep-only
    done
done

SWEEP_CSV="$OUT_BASE/summary_all.csv"
echo ""
echo "Concatenating -> $SWEEP_CSV"
conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

for scene in $SCENES; do
    SCENE_CSV="$OUT_BASE/$scene/summary_all.csv"
    conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
        --glob "$OUT_BASE/$scene/*/metrics/summary.csv" \
        --out "$SCENE_CSV" || true
    [[ -f "$SCENE_CSV" ]] && conda run -n gsquic python "$UTIL_DIR/experiments/plot_metrics.py" \
        --summary-csv "$SCENE_CSV" \
        --out-dir "$OUT_BASE/$scene/plots" \
        --group-by grid_shape \
        --title-suffix "$scene (Exp4 @1600² grid ablation)" || true
done

echo ""
echo "Done. Sweep root: $OUT_BASE"
