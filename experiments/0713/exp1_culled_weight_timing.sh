#!/usr/bin/env bash
# Exp1 (2026-07-13) wall-time half: --gs-weight-scope full vs visible, all 3 progressive
# methods, all 10 scenes. progressive_volume has no existing timing data at all (new for
# both scopes); progressive_screen_area/progressive_vol_d2 `full`-scope already exist in
# output/0713/selection_timing_gsorder_weight/ (reused, not rerun here) -- only `visible`
# is new for those two. Template: experiments/0713/rerun_timing_gsorder_weight*.sh.
set -euo pipefail
cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility

ROOT=/mnt/data1/samk/gs-quic/cs5262_tile_quic
OUT=output/0713/exp1_culled_weight_timing
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # GPU0 banned unless explicitly overridden

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

mkdir -p "$OUT"

for SCENE in bicycle garden stump chair drums ficus hotdog materials mic ship; do
    PLY="$ROOT/exp-dataset/$SCENE/point_cloud.ply"
    CACHE="output/oracle_tiling_cache/${SCENE}_8x8x8.npz"
    TRACE="$ROOT/exp-dataset/$SCENE/sparse_views_eval.json"

    for scope in full visible; do
        echo "########## $SCENE: 3 progressive methods, gs-weight-scope=$scope (150 cams) ##########"
        check_gpu_free
        time conda run -n gaussian_splatting python time_selection.py \
            --ply "$PLY" --tiling-cache "$CACHE" --camera-trace "$TRACE" \
            --methods progressive_screen_area progressive_volume progressive_vol_d2 \
            --gs-weight-scope "$scope" \
            --output-root "$OUT" --scene "$SCENE"
    done
done

echo "ALL DONE"
