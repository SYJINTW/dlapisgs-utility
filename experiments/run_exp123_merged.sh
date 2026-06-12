#!/usr/bin/env bash
# Merged exp1-3 rerun driver (2026-06-06).
# 6 test_utility_inmem.py invocations per scene; GT rendered fresh (no cache).
#
# Usage:
#   bash experiments/run_exp123_merged.sh [scene ...]
#   If no scenes given, runs all 10 (7 synth + 3 real).
#
# Override env vars:
#   OUT_BASE   — output root          (default: <repo>/output/0606/exp123_merged)
#   CACHE_DIR  — shared tiling cache  (default: <repo>/output/0606/.tiling_cache)
#   ORACLE_DIR — dir holding $scene/oracle_dq.npz  (default: <repo>/output/0606/exp4_oracle_dq)
#   ML_DIR     — dir holding $scene/lgbm.pkl        (default: <repo>/output/0606/ml_models)
#   GPU        — CUDA_VISIBLE_DEVICES               (default: 0)
#   LOG_DIR    — log dir                            (default: <repo>/logs/exp123_merged)
#
# Inv-5b (oracle_loo + ml) is skipped with a warning if prereqs are missing.
# Run in a detached screen/tmux session for long experiments.

set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL="$ROOT/dlapisgs-utility"
EXP="$ROOT/exp-dataset"
SCRIPT="$UTIL/test_utility_inmem.py"

OUT_BASE="${OUT_BASE:-$UTIL/output/0606/exp123_merged}"
CACHE_DIR="${CACHE_DIR:-$UTIL/output/0606/.tiling_cache}"
ORACLE_DIR="${ORACLE_DIR:-$UTIL/output/0606/exp4_oracle_dq}"
ML_DIR="${ML_DIR:-$UTIL/output/0606/ml_models}"
LOG_DIR="${LOG_DIR:-$UTIL/logs/exp123_merged}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

CONDA_ENV="gaussian_splatting"
DUMMY_IMAGE="$EXP/chair/predictions/color/test/r_0.png"

ALL_SCENES="chair drums ficus hotdog materials mic ship bicycle garden stump"
SCENES="${*:-$ALL_SCENES}"

BUDGETS="10 25 40 55 70 85 99 100"
COMMON_ARGS="--num-lod 1 --camera-index -1 --img-w 1600 --img-h 1600
             --w-mode mean --w-norm sum --c-norm sum
             --save-rep-only --png-workers 8"

mkdir -p "$CACHE_DIR" "$LOG_DIR"

# run_inv SCENE INV_TAG PACKING GRID_3 WEIGHT SCHEMES [EXTRA_ARGS...]
# GRID_3: three numbers, e.g. "8 8 8"
run_inv() {
    local scene="$1" inv_tag="$2" packing="$3" weight="$5" schemes="$6"
    local grid="$4"
    local grid_str="${grid// /x}"
    local extra="${7:-}"

    local ply="$EXP/$scene/point_cloud.ply"
    local trace="$EXP/$scene/sparse_views_eval.json"
    local cache="$CACHE_DIR/${scene}_${grid_str}.npz"
    local out="$OUT_BASE/$scene/$inv_tag"
    local log="$LOG_DIR/${scene}_${inv_tag}.log"

    if [ ! -f "$ply" ];   then echo "[SKIP] $scene/$inv_tag: PLY not found: $ply";   return; fi
    if [ ! -f "$trace" ]; then echo "[SKIP] $scene/$inv_tag: trace not found: $trace"; return; fi
    # Idempotent rerun: with SKIP_EXISTING=1, skip invocations already producing metrics.
    if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$out/metrics/summary.csv" ]; then
        echo "[SKIP-EXIST] $scene/$inv_tag: summary.csv present"; return
    fi

    echo "[RUN ] $scene / $inv_tag  packing=$packing grid=$grid_str weight=$weight schemes=[$schemes]"
    # shellcheck disable=SC2086
    LAPISGS_DUMMY_IMAGE="$DUMMY_IMAGE" \
    conda run -n "$CONDA_ENV" python "$SCRIPT" \
        --ply "$ply" --gt-ply "$ply" \
        --camera-trace "$trace" \
        --output-root "$out" \
        --grid-shape $grid \
        --budget-pct $BUDGETS \
        --schemes $schemes \
        --packing-mode "$packing" \
        --weight-mode "$weight" \
        --scene "$scene" \
        --tiling-cache "$cache" \
        $COMMON_ARGS \
        $extra \
        > "$log" 2>&1
    echo "[DONE] $scene / $inv_tag  -> $log"
}

for scene in $SCENES; do
    echo "=============================="
    echo "Scene: $scene   ($(date +%H:%M:%S))"
    echo "=============================="

    # Inv 1: progressive 8³ screen_area → vd_lod  [exp1(screen_area) + exp3(progressive)]
    run_inv "$scene" prog_8_sa     progressive "8 8 8" screen_area    "vd_lod"

    # Inv 2: progressive 8³ volume_over_d2 → vd_lod  [exp1]
    run_inv "$scene" prog_8_vod2   progressive "8 8 8" volume_over_d2 "vd_lod"

    # Inv 3: progressive 8³ volume → vd_lod  [exp1]
    run_inv "$scene" prog_8_vol    progressive "8 8 8" volume         "vd_lod"

    # Inv 4: progressive 8³ random → vd_lod  [exp1 random baseline]
    run_inv "$scene" prog_8_rand   progressive "8 8 8" random         "vd_lod"

    # Inv 5a: tile_strict 8³ screen_area → vd_lod vd_lod_w  [exp2 + exp3(tile_strict)]
    run_inv "$scene" strict_8_sa_a tile_strict "8 8 8" screen_area    "vd_lod vd_lod_w"

    # Inv 5b: tile_strict 8³ screen_area → oracle_loo ml  [prereq-gated]
    _oracle="$ORACLE_DIR/$scene/oracle_dq.npz"
    _mlpkl="$ML_DIR/$scene/rf.pkl"
    if [ -f "$_oracle" ] && [ -f "$_mlpkl" ]; then
        run_inv "$scene" strict_8_sa_b tile_strict "8 8 8" screen_area "oracle_loo ml" \
            "--oracle-npz $_oracle --ml-model-dir $ML_DIR/$scene --ml-model-type rf"
    else
        echo "[SKIP] $scene/strict_8_sa_b: prereqs missing"
        [ -f "$_oracle" ] || echo "       missing: $_oracle"
        [ -f "$_mlpkl"  ] || echo "       missing: $_mlpkl"
    fi

    # Inv 6: progressive 1³ screen_area → vd_lod  [exp3 nocull baseline]
    run_inv "$scene" prog_1_sa     progressive "1 1 1" screen_area    "vd_lod"

    echo ""
done

echo "All scenes complete. ($(date +%H:%M:%S))"
echo "Outputs: $OUT_BASE"
