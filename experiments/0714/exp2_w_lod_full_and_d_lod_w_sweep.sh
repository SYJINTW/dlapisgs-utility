#!/usr/bin/env bash
# Exp2 (2026-07-14): completing the 4-variant W_k-inclusive formula comparison.
#
# Context: w_lod (2026-07-13) was originally run with --gs-weight-scope visible, since its
# premise (W_k alone, no explicit v) is only sound when W_k is visibility-aware. Exp1's
# timing sweep (2026-07-13) then showed 'visible' uniformly SLOWER than 'full' (1.1-6x) with
# no quality upside once PSNR was read correctly (up to 8dB real-scene gap, not "neutral" --
# see PLAN.md 2026-07-14 correction). DECISION: --gs-weight-scope is always 'full', no
# exception -- 'visible' choice removed from both scripts' argparse (kept as unused code).
#
# This means w_lod must be re-tested under 'full' scope before it can be considered for
# canonical status -- expected to degrade (invisible-tile W_k won't be suppressed without
# culling) but not yet measured. Not assumed.
#
# Also new: d_lod_w (W_k + explicit /d, no v) -- the 4th of 4 W_k-inclusive formula variants
# (vd_lod_w, v_lod_w, w_lod, d_lod_w = all 4 {v,d}-presence subsets paired with w). Same
# soundness dependency on culled W_k as w_lod, same "full scope only" constraint now.
# selection_core.py::compute_raw_scores dispatcher updated (include_v/include_d/include_w
# flags), added to VALID_SCHEMES -- verified via debug.sh chair/0/40/8/tile_partial/d_lod_w
# before this sweep.
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
DSET="$ROOT/exp-dataset"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

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
OUT_BASE="$UTIL_DIR/output/0714/exp2_w_lod_full_and_d_lod_w"
ORACLE_ROOT="$UTIL_DIR/output/oracle/8/eval"
TILING_CACHE_BASE="$UTIL_DIR/output/oracle_tiling_cache"

mkdir -p "$OUT_BASE"

run_arm() {
    local arm="$1" scheme="$2"
    for scene in $SCENES; do
        PLY="$DSET/$scene/point_cloud.ply"
        TRACE="$DSET/$scene/sparse_views_eval.json"
        TILING_CACHE="$TILING_CACHE_BASE/${scene}_8x8x8.npz"
        ORACLE_NPZ="$ORACLE_ROOT/$scene/oracle_dq.npz"
        [[ -f "$PLY" && -f "$TRACE" ]] || { echo "[skip] $scene: missing ply/trace"; continue; }

        OUT_DIR="$OUT_BASE/$arm/$scene/grid8_tile_partial"
        if [[ -f "$OUT_DIR/params.yaml" ]]; then
            echo "[skip] $arm/$scene: already done"
            continue
        fi
        echo "---- [$arm] $scene ----"
        mkdir -p "$OUT_DIR"
        check_gpu_free
        conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
            --ply "$PLY" --gt-ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape 8 8 8 \
            --budget-pct $BUDGET_PCTS \
            --schemes "$scheme" \
            --num-lod 1 --camera-index -1 \
            --packing-mode tile_partial \
            --weight-mode screen_area --gs-weight-scope full \
            --w-norm sum --c-norm sum \
            --greedy-key marginal \
            --img-w 1600 --img-h 1600 \
            --scene "$scene" --group-by scheme \
            --tiling-cache "$TILING_CACHE" \
            --oracle-npz "$ORACLE_NPZ" \
            --lpips --save-rep-only
    done
}

run_arm w_lod_full w_lod
run_arm d_lod_w    d_lod_w

echo ""
echo "Concatenating per-arm summaries..."
for arm in w_lod_full d_lod_w; do
    conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
        --glob "$OUT_BASE/$arm/*/*/metrics/summary.csv" \
        --out "$OUT_BASE/$arm/summary_all.csv"
done

echo ""
echo "Done. Sweep root: $OUT_BASE"
