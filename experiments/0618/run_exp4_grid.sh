#!/usr/bin/env bash
# Experiment 4 — grid-size ablation (redesigned 2026-06-18).
# Fixed:  weight_mode=screen_area, img=1600x1600, greedy-key=marginal.
# Swept:  grid ∈ {1x1x1, 2x2x2, 4x4x4, 8x8x8}
#         × packing ∈ {progressive(vd_lod), tile_partial(vd_lod+oracle_loo+ml), tile_strict(vd_lod+oracle_loo+ml)}.
# Scenes: all 10.
#
# Oracle NPZs:  $ORACLE_ROOT/{g}/{scene}/oracle_dq.npz  (1/2/4 from output/0618/oracle_grid; 8 from exp5_oracle)
# ML models:   $ML_ROOT/{g}/{scene}/AC/  (1/2/4 from output/0618/ml_models; 8 from output/0606/ml_models)
# Tiling cache: output/0606/.tiling_cache/{scene}_{g}x{g}x{g}.npz (all 40 exist)
#
# Env overrides:
#   SCENES="chair bicycle"
#   GRIDS="2 4 8"
#   CUDA_VISIBLE_DEVICES=0
#   OUTPUT_ROOT=.../output/MMDD/exp4_grid
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
GRIDS="${GRIDS:-8 1 2 4 16}"
BUDGET_PCTS="${BUDGET_PCTS:-10 25 40 55 70 85 99 100}"
NUM_LOD="${NUM_LOD:-1}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0618/exp4_grid}"

# Oracle/ML roots — 8^3 uses existing exp5 outputs; 1/2/4 use fresh 0618 regen.
ORACLE_ROOT_124="${ORACLE_ROOT_124:-$UTIL_DIR/output/0618/oracle_grid}"
ORACLE_ROOT_8="${ORACLE_ROOT_8:-$UTIL_DIR/output/0615/exp5_oracle/eval}"
ML_ROOT_124="${ML_ROOT_124:-$UTIL_DIR/output/0618/ml_models}"
ML_ROOT_8="${ML_ROOT_8:-$UTIL_DIR/output/0606/ml_models}"

TILING_CACHE_BASE="$UTIL_DIR/output/0606/.tiling_cache"

echo "=========================================="
echo "Exp 4 @1600² — grid-size ablation (prog + tile_partial + tile_strict)"
echo "OUTPUT_ROOT : $OUT_BASE"
echo "SCENES      : $SCENES"
echo "GRIDS       : $GRIDS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for g in $GRIDS; do
    echo ""
    echo "====== grid=${g}x${g}x${g} ======"

    # Resolve oracle/ML root per grid
    if [[ "$g" == "8" ]]; then
        ORACLE_ROOT="$ORACLE_ROOT_8"
        ML_ROOT="$ML_ROOT_8"
    else
        ORACLE_ROOT="$ORACLE_ROOT_124/$g"
        ML_ROOT="$ML_ROOT_124/$g"
    fi

    for scene in $SCENES; do
        PLY="$DSET/$scene/point_cloud.ply"
        TRACE="$DSET/$scene/sparse_views_eval.json"
        TILING_CACHE="$TILING_CACHE_BASE/${scene}_${g}x${g}x${g}.npz"

        if [[ ! -f "$PLY" ]];   then echo "[skip] $scene: no point_cloud.ply";       continue; fi
        if [[ ! -f "$TRACE" ]]; then echo "[skip] $scene: no sparse_views_eval.json"; continue; fi

        # ── progressive ──────────────────────────────────────────────────────
        OUT_DIR="$OUT_BASE/$scene/grid${g}_progressive"
        if [[ -f "$OUT_DIR/params.yaml" ]]; then
            echo "[skip] $scene grid${g} progressive: already done"
        else
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
                --w-mode sum \
                --img-w 1600 --img-h 1600 \
                --scene "$scene" \
                --group-by grid_shape \
                --tiling-cache "$TILING_CACHE" \
                --save-rep-only
        fi

        # Oracle/ML check (shared by tile_partial and tile_strict)
        if [[ "$g" == "8" ]]; then
            ORACLE_NPZ="$ORACLE_ROOT/$scene/oracle_dq.npz"
        else
            ORACLE_NPZ="$ORACLE_ROOT/$scene/oracle_dq.npz"
        fi
        ML_DIR="$ML_ROOT/$scene/AC"

        if [[ ! -f "$ORACLE_NPZ" ]]; then echo "[skip tiled] $scene grid${g}: no oracle_dq.npz at $ORACLE_NPZ"; continue; fi
        if [[ ! -d "$ML_DIR" ]];     then echo "[skip tiled] $scene grid${g}: no ML model at $ML_DIR";           continue; fi

        # ── tile_partial ──────────────────────────────────────────────────────
        OUT_DIR="$OUT_BASE/$scene/grid${g}_tile_partial"
        if [[ -f "$OUT_DIR/params.yaml" ]]; then
            echo "[skip] $scene grid${g} tile_partial: already done"
        else
            echo "---- [$scene] grid=${g}x${g}x${g} tile_partial ----"
            mkdir -p "$OUT_DIR"
            check_gpu_free
            conda run -n gaussian_splatting python "$UTIL_DIR/test_utility_inmem.py" \
                --ply "$PLY" \
                --gt-ply "$PLY" \
                --output-root "$OUT_DIR" \
                --camera-trace "$TRACE" \
                --grid-shape "$g" "$g" "$g" \
                --budget-pct $BUDGET_PCTS \
                --schemes vd_lod vd_lod_w oracle_loo ml \
                --num-lod "$NUM_LOD" \
                --camera-index -1 \
                --packing-mode tile_partial \
                --weight-mode screen_area \
                --w-mode sum \
                --greedy-key marginal \
                --img-w 1600 --img-h 1600 \
                --scene "$scene" \
                --group-by grid_shape \
                --tiling-cache "$TILING_CACHE" \
                --oracle-npz "$ORACLE_NPZ" \
                --ml-model-dir "$ML_DIR" \
                --ml-model-type rf \
                --save-rep-only
        fi

        # ── tile_strict ───────────────────────────────────────────────────────
        OUT_DIR="$OUT_BASE/$scene/grid${g}_tile_strict"
        if [[ -f "$OUT_DIR/params.yaml" ]]; then
            echo "[skip] $scene grid${g} tile_strict: already done"
        else
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
                --schemes vd_lod vd_lod_w oracle_loo ml \
                --num-lod "$NUM_LOD" \
                --camera-index -1 \
                --packing-mode tile_strict \
                --weight-mode screen_area \
                --w-mode sum \
                --greedy-key marginal \
                --img-w 1600 --img-h 1600 \
                --scene "$scene" \
                --group-by grid_shape \
                --tiling-cache "$TILING_CACHE" \
                --oracle-npz "$ORACLE_NPZ" \
                --ml-model-dir "$ML_DIR" \
                --ml-model-type rf \
                --save-rep-only
        fi

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
