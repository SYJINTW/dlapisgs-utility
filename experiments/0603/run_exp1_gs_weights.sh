#!/usr/bin/env bash
# Experiment 1 @1600² — GS weight-mode sweep, all 10 scenes.
#
# Fixed:  packing=progressive, scheme=vd_lod, grid=8x8x8, img=1600x1600.
# Swept:  weight-mode ∈ {screen_area, volume_over_d2, volume, random}.
# Scenes: all 10. Uses test_utility_inmem.py.
# Random control: weight_mode=random (U[0,1] seed=42 in compute_gaussian_weights_v2).
#
# Env overrides:
#   SCENES="chair drums"
#   WEIGHT_MODES="volume volume_over_d2"
#   CUDA_VISIBLE_DEVICES=0
#   OUTPUT_ROOT=.../output/MMDD/exp1_gs_weights
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Abort if target GPU has < MIN_FREE_MIB free (default 8 GiB).
MIN_FREE_MIB="${MIN_FREE_MIB:-8192}"
check_gpu_free() {
    local gpu_idx="${CUDA_VISIBLE_DEVICES%%,*}"
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null)
    if [[ -z "$free_mib" || "$free_mib" -lt "$MIN_FREE_MIB" ]]; then
        echo "[GPU GUARD] GPU $gpu_idx: ${free_mib:-?} MiB free (need >=${MIN_FREE_MIB}). Aborting."
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$gpu_idx" 2>/dev/null
        return 1
    fi
    echo "[GPU GUARD] GPU $gpu_idx: ${free_mib} MiB free — OK"
}

SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"
WEIGHT_MODES="${WEIGHT_MODES:-screen_area volume_over_d2 random volume}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
GRID_SHAPE="${GRID_SHAPE:-8 8 8}"
NUM_LOD="${NUM_LOD:-1}"
SCHEME="${SCHEME:-vd_lod}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0605/exp1_1600_vol_modes}"

# All scenes expose canonical symlinks at exp-dataset/{scene}/:
#   point_cloud.ply, sparse_views_eval.json, gt_renders_eval/
DSET="$ROOT/exp-dataset"

echo "=========================================="
echo "Exp 1 @1600² — weight-mode sweep, 10 scenes"
echo "OUTPUT_ROOT  : $OUT_BASE"
echo "SCENES       : $SCENES"
echo "WEIGHT_MODES : $WEIGHT_MODES"
echo "BUDGET_PCTS  : $BUDGET_PCTS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    TRACE="$DSET/$scene/sparse_views_eval.json"
    GT_DIR="$DSET/$scene/gt_renders_eval"

    if [[ ! -f "$PLY" ]]; then echo "[skip] $scene: no point_cloud.ply"; continue; fi
    if [[ ! -f "$TRACE" ]]; then echo "[skip] $scene: no sparse_views_eval.json"; continue; fi
    if [[ ! -d "$GT_DIR" ]]; then echo "[skip] $scene: no gt_renders_eval/"; continue; fi

    TILING_CACHE="$OUT_BASE/$scene/.tiling_cache.npz"
    mkdir -p "$(dirname "$TILING_CACHE")"

    for wm in $WEIGHT_MODES; do
        OUT_DIR="$OUT_BASE/$scene/$wm"
        echo ""
        echo "---- [$scene] weight_mode=$wm ----"
        mkdir -p "$OUT_DIR"
        check_gpu_free

        conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
            --ply "$PLY" \
            --gt-ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape $GRID_SHAPE \
            --budget-pct $BUDGET_PCTS \
            --schemes "$SCHEME" \
            --num-lod "$NUM_LOD" \
            --camera-index -1 \
            --packing-mode progressive \
            --weight-mode "$wm" \
            --img-w 1600 --img-h 1600 \
            --scene "$scene" \
            --group-by weight_mode \
            --tiling-cache "$TILING_CACHE" \
            --gt-renders-cache "$GT_DIR" \
            --save-rep-only

        conda run -n gsquic python "$UTIL_DIR/experiments/aggregate_timings.py" \
            --output-root "$OUT_DIR" || true
    done
done

# Concat per-cell CSVs + plot
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
    if [[ -f "$SCENE_CSV" ]]; then
        conda run -n gsquic python "$UTIL_DIR/experiments/plot_metrics.py" \
            --summary-csv "$SCENE_CSV" \
            --out-dir "$OUT_BASE/$scene/plots" \
            --group-by weight_mode \
            --title-suffix "$scene (Exp1 @1600² progressive)" || true
    fi
done

echo ""
echo "Done. Sweep root: $OUT_BASE"
