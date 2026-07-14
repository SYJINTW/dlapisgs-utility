#!/usr/bin/env bash
# Exp1 extension (Open item 4, 2026-07-14): new weight_mode `screen_area_over_d2` --
# opacity*screen_area, stacked with an EXPLICIT extra /d^2 distance-decay, testing whether
# it's redundant with screen_area's own implicit ~1/d^2 perspective-projection shrinkage
# (project_covariance_2d). Same grid8/progressive/vd_lod/full-scope setup as the original
# Exp1 weight-mode sweep (screen_area/volume/volume_over_d2, already in
# output/0704/quality_sweep/<scene>/grid8_progressive/metrics/summary.csv, scheme=vd_lod,
# weight_mode=screen_area -- reused as baseline, not rerun here). --gs-weight-scope is
# full-only now (2026-07-14 decision) -- no scope axis in this sweep, unlike 0713's exp1.
#
# Structural template: experiments/0713/exp1_culled_weight_sweep.sh (copied structure, not
# invoked directly, per dated-script convention).
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
OUT_BASE="$UTIL_DIR/output/0714/exp1_weight_screen_area_over_d2"
TILING_CACHE_BASE="$UTIL_DIR/output/oracle_tiling_cache"

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    TRACE="$DSET/$scene/sparse_views_eval.json"
    TILING_CACHE="$TILING_CACHE_BASE/${scene}_8x8x8.npz"
    [[ -f "$PLY" && -f "$TRACE" ]] || { echo "[skip] $scene: missing ply/trace"; continue; }

    OUT_DIR="$OUT_BASE/$scene/grid8_progressive_screen_area_over_d2"
    if [[ -f "$OUT_DIR/params.yaml" ]]; then
        echo "[skip] $scene: already done"
        continue
    fi
    echo "---- [$scene] weight_mode=screen_area_over_d2 ----"
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
        --weight-mode screen_area_over_d2 --gs-weight-scope full \
        --w-norm sum --c-norm sum \
        --img-w 1600 --img-h 1600 \
        --scene "$scene" --group-by weight_mode \
        --tiling-cache "$TILING_CACHE" \
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
