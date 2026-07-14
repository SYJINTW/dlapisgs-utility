#!/usr/bin/env bash
# Exp1 (2026-07-13): does culling GS-weight compute to the visible-tile subset (--gs-weight-scope
# visible, selection_core.py::compute_camera_weights_culled) preserve progressive-packing
# selection quality vs the current full-scene compute? Grid8 only (the canonical grid this
# whole plan pins to -- see .claude/PLAN.md), all 3 weight modes (screen_area, volume,
# volume_over_d2). Only the `visible` variant is new here -- `full`-scope grid8 progressive
# rows for all 3 modes already exist in output/0704/quality_sweep/summary_all_ml_fixed.csv,
# reused for comparison, not rerun.
#
# Structural template: the Exp1 slice in experiments/0704/run_quality_sweep.sh:153-188
# (copied structure, not invoked directly, per this project's dated-script convention).
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
OUT_BASE="$UTIL_DIR/output/0713/exp1_culled_weight"
TILING_CACHE_BASE="$UTIL_DIR/output/oracle_tiling_cache"

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    TRACE="$DSET/$scene/sparse_views_eval.json"
    TILING_CACHE="$TILING_CACHE_BASE/${scene}_8x8x8.npz"
    [[ -f "$PLY" && -f "$TRACE" ]] || { echo "[skip] $scene: missing ply/trace"; continue; }

    for wm in screen_area volume volume_over_d2; do
        OUT_DIR="$OUT_BASE/$scene/grid8_progressive_${wm}_visible"
        if [[ -f "$OUT_DIR/params.yaml" ]]; then
            echo "[skip] $scene weight_mode=$wm: already done"
            continue
        fi
        echo "---- [$scene] weight_mode=$wm (gs-weight-scope=visible) ----"
        mkdir -p "$OUT_DIR"
        check_gpu_free
        conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
            --ply "$PLY" --gt-ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape 8 8 8 \
            --budget-pct $BUDGET_PCTS \
            --schemes vd_lod \
            --num-lod 1 --camera-index -1 \
            --packing-mode progressive \
            --weight-mode "$wm" --gs-weight-scope visible \
            --w-norm sum --c-norm sum \
            --img-w 1600 --img-h 1600 \
            --scene "$scene" --group-by weight_mode \
            --tiling-cache "$TILING_CACHE" \
            --lpips --save-rep-only
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
