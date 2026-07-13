#!/usr/bin/env bash
# Consolidated quality sweep — supersedes exp1_gs_weights.sh / exp3_packing.sh /
# exp4_grid.sh (all 3 deleted 2026-07-04: all three referenced dead flags,
# --w-mode and --gt-renders-cache, removed from test_utility_inmem.py since; running
# any of them as-is would error).
#
# Covers the table entries for Exp1 (weight-mode), Exp2 (scheme), Exp3 (packing:
# subsumed by the grid=8 slice's packing-mode axis), and Exp4 (grid-size) in one
# design, with LPIPS enabled throughout (--lpips; lpips pkg already installed).
#
# 2026-07-05: ML-improvement track concluded. Single deployed model everywhere:
# pooled_all10_ACG (LGBM regressor pooled across all 10 scenes, Group A now
# includes cos_cam_tile directly -- see ml/features.py). Domain-split pooling
# (pooled_real3_ACG + pooled_synth7_ACG) scored slightly higher per-domain, but
# the user chose one model for simplicity over that marginal gain. Rank models
# (candidate 1, pooled_rank_*) are DEFERRED -- a bug in their exp()/marginal-key
# handling was found and partially fixed, but the fix only stops a crash, it
# does not restore competitive selection quality; not usable this round.
#
# oracle_dq.npz now regenerated at grid 1/2/4/16 (previously only existed at
# grid=8) so oracle_loo and the ml/vd_lod/v_lod_w schemes all run at every grid.
#
# Env overrides:
#   SCENES="chair bicycle"
#   GRIDS="8 1 2 4 16"
#   CUDA_VISIBLE_DEVICES=2
#   OUTPUT_ROOT=.../output/MMDD/quality_sweep
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
SCENES="${SCENES:-$REAL_SCENES $SYNTH_SCENES}"
GRIDS="${GRIDS:-8 1 2 4 16}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
NUM_LOD="${NUM_LOD:-1}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0704/quality_sweep}"
ORACLE_ROOT="${ORACLE_ROOT:-$UTIL_DIR/output/oracle}"     # {ORACLE_ROOT}/{grid}/eval/{scene}/oracle_dq.npz

# ML_DIR: leave unset to get the per-grid model picked inside the grid loop
# (pooled_all10_ACG_grid{1,2,4,8,16}); set explicitly to pin one dir for every grid.
ML_TYPE="${ML_TYPE:-lgbm}"

TILING_CACHE_BASE="${TILING_CACHE_BASE:-$UTIL_DIR/output/oracle_tiling_cache}"

echo "=========================================="
echo "Quality sweep @1600² (exp1+2+3+4 tables, LPIPS enabled)"
echo "OUTPUT_ROOT : $OUT_BASE"
echo "SCENES      : $SCENES"
echo "GRIDS       : $GRIDS"
echo "ML_DIR      : ${ML_DIR:-<per-grid: pooled_all10_ACG_grid<N>>} ($ML_TYPE)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for g in $GRIDS; do
    echo ""
    echo "====== grid=${g}x${g}x${g} ======"
    # ml scheme needs a model trained on THIS grid's tile geometry -- a grid8-trained
    # model scores nonsense on grid1/2/4/16 tile features. ML_DIR env override (if set)
    # still wins for a single fixed dir; otherwise pick the per-grid trained model.
    ML_DIR_G="${ML_DIR:-$UTIL_DIR/output/ml_models_experimental/pooled_all10_ACG_grid${g}}"
    echo "ML_DIR (grid${g}): $ML_DIR_G"

    for scene in $SCENES; do
        PLY="$DSET/$scene/point_cloud.ply"
        TRACE="$DSET/$scene/sparse_views_eval.json"
        TILING_CACHE="$TILING_CACHE_BASE/${scene}_${g}x${g}x${g}.npz"
        ORACLE_NPZ="$ORACLE_ROOT/$g/eval/$scene/oracle_dq.npz"

        if [[ ! -f "$PLY" ]];   then echo "[skip] $scene: no point_cloud.ply";       continue; fi
        if [[ ! -f "$TRACE" ]]; then echo "[skip] $scene: no sparse_views_eval.json"; continue; fi

        # ── progressive (vd_lod, geometric only — runs at every grid) ────────────
        OUT_DIR="$OUT_BASE/$scene/grid${g}_progressive"
        if [[ -f "$OUT_DIR/params.yaml" ]]; then
            echo "[skip] $scene grid${g} progressive: already done"
        else
            echo "---- [$scene] grid=${g}x${g}x${g} progressive ----"
            mkdir -p "$OUT_DIR"
            check_gpu_free
            conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
                --ply "$PLY" --gt-ply "$PLY" \
                --output-root "$OUT_DIR" \
                --camera-trace "$TRACE" \
                --grid-shape "$g" "$g" "$g" \
                --budget-pct $BUDGET_PCTS \
                --schemes vd_lod \
                --num-lod "$NUM_LOD" --camera-index -1 \
                --packing-mode progressive \
                --weight-mode screen_area \
                --w-norm sum --c-norm sum \
                --img-w 1600 --img-h 1600 \
                --scene "$scene" --group-by scheme \
                --tiling-cache "$TILING_CACHE" \
                --lpips --save-rep-only
        fi

        # ── tile_partial / tile_strict (vd_lod+v_lod_w+oracle_loo+ml) — every grid,
        #    gated on oracle_dq.npz existing for that (scene, grid) ────────────────
        if [[ ! -f "$ORACLE_NPZ" ]]; then
            echo "[skip tiled] $scene grid${g}: no oracle_dq.npz at $ORACLE_NPZ"; continue
        fi

        for packing in tile_partial tile_strict; do
            OUT_DIR="$OUT_BASE/$scene/grid${g}_${packing}"
            if [[ -f "$OUT_DIR/params.yaml" ]]; then
                echo "[skip] $scene grid${g} ${packing}: already done"
                continue
            fi
            echo "---- [$scene] grid=${g}x${g}x${g} ${packing} ----"
            mkdir -p "$OUT_DIR"
            check_gpu_free
            conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
                --ply "$PLY" --gt-ply "$PLY" \
                --output-root "$OUT_DIR" \
                --camera-trace "$TRACE" \
                --grid-shape "$g" "$g" "$g" \
                --budget-pct $BUDGET_PCTS \
                --schemes vd_lod v_lod_w oracle_loo ml \
                --num-lod "$NUM_LOD" --camera-index -1 \
                --packing-mode "$packing" \
                --weight-mode screen_area \
                --w-norm sum --c-norm sum \
                --greedy-key marginal \
                --img-w 1600 --img-h 1600 \
                --scene "$scene" --group-by scheme \
                --tiling-cache "$TILING_CACHE" \
                --oracle-npz "$ORACLE_NPZ" \
                --ml-model-dir "$ML_DIR_G" --ml-model-type "$ML_TYPE" \
                --lpips --save-rep-only
        done
    done
done

# ── Exp1 slice: weight-mode comparison at grid=8/progressive/vd_lod (screen_area
#    already covered by the main sweep above; only the 3 non-canonical modes here) ──
echo ""
echo "====== Exp1 slice: weight-mode @ grid8/progressive/vd_lod ======"
for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    TRACE="$DSET/$scene/sparse_views_eval.json"
    TILING_CACHE="$TILING_CACHE_BASE/${scene}_8x8x8.npz"
    [[ -f "$PLY" && -f "$TRACE" ]] || { echo "[skip] $scene: missing ply/trace"; continue; }

    for wm in volume_over_d2 volume random; do
        OUT_DIR="$OUT_BASE/$scene/grid8_progressive_${wm}"
        if [[ -f "$OUT_DIR/params.yaml" ]]; then
            echo "[skip] $scene weight_mode=$wm: already done"
            continue
        fi
        echo "---- [$scene] weight_mode=$wm ----"
        mkdir -p "$OUT_DIR"
        check_gpu_free
        conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
            --ply "$PLY" --gt-ply "$PLY" \
            --output-root "$OUT_DIR" \
            --camera-trace "$TRACE" \
            --grid-shape 8 8 8 \
            --budget-pct $BUDGET_PCTS \
            --schemes vd_lod \
            --num-lod "$NUM_LOD" --camera-index -1 \
            --packing-mode progressive \
            --weight-mode "$wm" \
            --w-norm sum --c-norm sum \
            --img-w 1600 --img-h 1600 \
            --scene "$scene" --group-by weight_mode \
            --tiling-cache "$TILING_CACHE" \
            --lpips --save-rep-only
    done
done

# ── Concat ────────────────────────────────────────────────────────────────────
SWEEP_CSV="$OUT_BASE/summary_all.csv"
echo ""
echo "Concatenating -> $SWEEP_CSV"
conda run -n gsquic python "$UTIL_DIR/experiments/concat_summaries.py" \
    --glob "$OUT_BASE/*/*/metrics/summary.csv" \
    --out "$SWEEP_CSV"

echo ""
echo "Done. Sweep root: $OUT_BASE"
