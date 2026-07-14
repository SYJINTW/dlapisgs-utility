#!/usr/bin/env bash
# Exp2 (2026-07-13): 3-way heuristic formula comparison, grid8/tile_partial/screen_area,
# all 10 scenes. vd_lod_w (v.W_k/d, old rejected -- zero rows in the current canonical
# sweep, fresh rerun under the current pipeline) and w_lod (W_k alone, needs the culled
# --gs-weight-scope visible weights from Exp1) run as SEPARATE test_utility_inmem.py
# invocations -- w_gi is computed once per camera, shared across every scheme requested in
# one invocation (test_utility_inmem.py:892-898-ish), so combining w_lod (needs visible
# scope) with vd_lod_w (needs full scope) in one --schemes list would corrupt one or the
# other. v_lod_w (current canonical) is reused from the existing canonical sweep, not rerun.
#
# Exp1 verdict (2026-07-13): --gs-weight-scope visible is quality-neutral (confirmed, all 10
# scenes) but uniformly SLOWER than full (1.1-6x across scenes/methods, confirmed) -- the
# "adopt visible as default" decision rule failed on speed. Launched anyway on user's call:
# w_lod's premise is whether W_k alone (encoding visibility) suffices for tile ranking
# quality, not selection speed -- the overhead is a separate question being investigated
# independently, not a gate on this quality experiment.
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
OUT_BASE="$UTIL_DIR/output/0713/exp2_formula_comparison"
ORACLE_ROOT="$UTIL_DIR/output/oracle/8/eval"
TILING_CACHE_BASE="$UTIL_DIR/output/oracle_tiling_cache"

mkdir -p "$OUT_BASE"

run_arm() {
    local arm="$1" scheme="$2" gs_weight_scope="$3"
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
            --weight-mode screen_area --gs-weight-scope "$gs_weight_scope" \
            --w-norm sum --c-norm sum \
            --greedy-key marginal \
            --img-w 1600 --img-h 1600 \
            --scene "$scene" --group-by scheme \
            --tiling-cache "$TILING_CACHE" \
            --oracle-npz "$ORACLE_NPZ" \
            --lpips --save-rep-only
    done
}

run_arm vd_lod_w vd_lod_w full
run_arm w_lod    w_lod    visible

echo ""
echo "Concatenating per-arm summaries..."
for arm in vd_lod_w w_lod; do
    conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
        --glob "$OUT_BASE/$arm/*/*/metrics/summary.csv" \
        --out "$OUT_BASE/$arm/summary_all.csv"
done

echo ""
echo "Done. Sweep root: $OUT_BASE"
