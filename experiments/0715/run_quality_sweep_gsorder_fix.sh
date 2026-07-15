#!/usr/bin/env bash
# Exp4 grid-sweep rerun after the gs_order bug fix (PLAN.md "Bug pattern found
# (2026-07-15)"): selection_core.py's greedy_order() now force-overrides gs_order="ply" for
# vd_lod (NO_WEIGHT_SCHEMES) regardless of what a caller passes -- test_utility_inmem.py
# itself needed NO code change, it already forwarded `scheme` to build_greedy_order(). This
# script reruns all 4 schemes together (vd_lod v_lod_w oracle_loo ml) per (scene, grid,
# packing) -- matches original experiments/0704/run_quality_sweep.sh scope for tile_partial +
# the progressive-only (vd_lod) section, MINUS the Exp1 weight-mode slice (progressive/vd_lod/
# {volume_over_d2,volume,random}) -- untouched by this bug (progressive packing always uses
# real weight order, unaffected by NO_WEIGHT_SCHEMES; see selection_core.py NO_WEIGHT_SCHEMES
# docstring), not rerun here. tile_strict dropped (2026-07-15, user decision) -- also not
# rerun; note tile_strict's vd_lod numbers were never actually corrupted by this bug in the
# first place (whole-tile selection is invariant to intra-tile GS order for the final
# selected SET, only progressive/tile_partial select partial tiles), so this is a scope cut,
# not a correctness fix.
#
# Writes to a FRESH output root (0715, not overwriting 0704 in place) -- avoids any stale
# params.yaml skip-guard trap and avoids mutating existing data. plotting/paper_plot_metrics.ipynb
# needs repointing at this once done (separate step).
#
# ml scheme needs per-scene-per-grid models -- grid8 reuses the existing canonical
# per_scene/<scene>/AC (2026-07-13, unaffected by either issue); grid{1,2,4,16} need the new
# per_scene_grid{g}/<scene>/AC models (experiments/0715/train_per_scene_grids.sh must be run
# first for those grids).
#
# Env overrides:
#   SCENES="chair bicycle"
#   GRIDS="8 1 2 4 16"
#   CUDA_VISIBLE_DEVICES=1
#   OUTPUT_ROOT=.../output/MMDD/quality_sweep
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
DSET="$ROOT/exp-dataset"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # GPU0 banned

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
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0715/quality_sweep}"
ORACLE_ROOT="${ORACLE_ROOT:-$UTIL_DIR/output/oracle}"     # {ORACLE_ROOT}/{grid}/eval/{scene}/oracle_dq.npz
ML_TYPE="${ML_TYPE:-lgbm}"

TILING_CACHE_BASE="${TILING_CACHE_BASE:-$UTIL_DIR/output/oracle_tiling_cache}"

echo "=========================================="
echo "Exp4 grid sweep, gs_order-bug rerun (all 4 schemes)"
echo "OUTPUT_ROOT : $OUT_BASE"
echo "SCENES      : $SCENES"
echo "GRIDS       : $GRIDS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for g in $GRIDS; do
    echo ""
    echo "====== grid=${g}x${g}x${g} ======"
    if [[ "$g" == "8" ]]; then
        ML_DIR_BASE="$UTIL_DIR/output/ml_models_experimental/per_scene"
    else
        ML_DIR_BASE="$UTIL_DIR/output/ml_models_experimental/per_scene_grid${g}"
    fi

    for scene in $SCENES; do
        PLY="$DSET/$scene/point_cloud.ply"
        TRACE="$DSET/$scene/sparse_views_eval.json"
        TILING_CACHE="$TILING_CACHE_BASE/${scene}_${g}x${g}x${g}.npz"
        ORACLE_NPZ="$ORACLE_ROOT/$g/eval/$scene/oracle_dq.npz"
        ML_DIR_G="$ML_DIR_BASE/$scene/AC"

        if [[ ! -f "$PLY" ]];   then echo "[skip] $scene: no point_cloud.ply";       continue; fi
        if [[ ! -f "$TRACE" ]]; then echo "[skip] $scene: no sparse_views_eval.json"; continue; fi
        if [[ ! -f "$ML_DIR_G/lgbm.pkl" ]]; then
            echo "[abort] $scene grid${g}: no ml model at $ML_DIR_G -- run train_per_scene_grids.sh first"
            exit 1
        fi

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

        # ── tile_partial (vd_lod+v_lod_w+oracle_loo+ml) — every grid, gated on
        #    oracle_dq.npz existing for that (scene, grid). tile_strict dropped. ──
        if [[ ! -f "$ORACLE_NPZ" ]]; then
            echo "[skip tiled] $scene grid${g}: no oracle_dq.npz at $ORACLE_NPZ"; continue
        fi

        for packing in tile_partial; do
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

echo ""
echo "Done (this shard). Sweep root: $OUT_BASE"
