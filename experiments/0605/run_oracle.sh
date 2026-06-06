#!/usr/bin/env bash
# Exp 0 — LOO oracle ΔQ for all 10 scenes.
# Runs SUFFIX=train (cameras 0-149 from full trace) and SUFFIX=eval (cameras 0-149
# from eval trace = original 150-299). Both needed: eval feeds experiments, train feeds ML.
#
# Canonical inputs at exp-dataset/{scene}/:
#   sparse_views_full.json  — full 300-view trace (train oracle: --num-cameras 150)
#   sparse_views_eval.json  — 150-camera eval trace, reindexed 0-149 (eval oracle)
#   point_cloud.ply
#
# TILING_BASE: directory containing {scene}/.tiling_cache.npz at the target grid size.
#   Default: output from exp1_gs_weights (8x8x8). Must exist before running oracle.
#   For exp4 grid ablation: set TILING_BASE to the exp4 tiling output and GRID accordingly.
#
# Output: $OUT_BASE/{train|eval}/{scene}/oracle_dq.npz
#
# Env overrides:
#   SCENES="chair bicycle"
#   CUDA_VISIBLE_DEVICES=0
#   TILING_BASE=output/MMDD/exp1_gs_weights
#   OUTPUT_ROOT=output/MMDD/oracle
#   GRID=8   (informational; actual grid comes from TILING_BASE)
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"
DSET="$ROOT/exp-dataset"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"
OUT_BASE="${OUTPUT_ROOT:-$UTIL_DIR/output/0605/oracle}"
TILING_BASE="${TILING_BASE:-$UTIL_DIR/output/0605/exp1_gs_weights}"
GRID="${GRID:-8}"

echo "=========================================="
echo "Exp 0 — LOO oracle ΔQ, all 10 scenes"
echo "OUT_BASE     : $OUT_BASE"
echo "TILING_BASE  : $TILING_BASE/{scene}/.tiling_cache.npz"
echo "GRID         : ${GRID}x${GRID}x${GRID}  (from TILING_BASE)"
echo "SCENES       : $SCENES"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

mkdir -p "$OUT_BASE"

for scene in $SCENES; do
    PLY="$DSET/$scene/point_cloud.ply"
    FULL_TRACE="$DSET/$scene/sparse_views_full.json"
    EVAL_TRACE="$DSET/$scene/sparse_views_eval.json"
    TILING_CACHE="$TILING_BASE/$scene/.tiling_cache.npz"

    if [[ ! -f "$PLY" ]];          then echo "[skip] $scene: no point_cloud.ply";       continue; fi
    if [[ ! -f "$TILING_CACHE" ]]; then echo "[skip] $scene: no tiling cache at $TILING_CACHE"; continue; fi

    # ── Train oracle (cameras 0-149 from full trace) ──────────────────────────
    if [[ ! -f "$FULL_TRACE" ]]; then
        echo "[skip train] $scene: no sparse_views_full.json"
    else
        echo ""
        echo "---- [$scene] oracle SUFFIX=train ----"
        mkdir -p "$OUT_BASE/train/$scene"

        conda run -n gaussian_splatting \
            python "$UTIL_DIR/experiments/exp4_oracle_dq.py" \
            --ply "$PLY" \
            --trace "$FULL_TRACE" \
            --tiling-cache "$TILING_CACHE" \
            --output-root "$OUT_BASE/train/$scene" \
            --num-cameras 150 \
            --width 1600 --height 1600 \
            --flush-every 10 \
            --skip-existing

        echo "[$scene] train done -> $OUT_BASE/train/$scene/oracle_dq.npz"
    fi

    # ── Eval oracle (cameras 0-149 from eval trace = orig 150-299) ────────────
    if [[ ! -f "$EVAL_TRACE" ]]; then
        echo "[skip eval] $scene: no sparse_views_eval.json"
    else
        echo ""
        echo "---- [$scene] oracle SUFFIX=eval ----"
        mkdir -p "$OUT_BASE/eval/$scene"

        conda run -n gaussian_splatting \
            python "$UTIL_DIR/experiments/exp4_oracle_dq.py" \
            --ply "$PLY" \
            --trace "$EVAL_TRACE" \
            --tiling-cache "$TILING_CACHE" \
            --output-root "$OUT_BASE/eval/$scene" \
            --num-cameras 0 \
            --width 1600 --height 1600 \
            --flush-every 10 \
            --skip-existing

        echo "[$scene] eval done -> $OUT_BASE/eval/$scene/oracle_dq.npz"
    fi
done

echo ""
echo "Done. Oracle root: $OUT_BASE"
echo "Next: retrain ML models using train oracle, then run exp1-4."
