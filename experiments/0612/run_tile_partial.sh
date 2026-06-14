#!/usr/bin/env bash
# tile_partial sweep — re-adds the packing mode dropped earlier (the
# "tile_partial ~= tile_strict" finding was wrong; tile_partial partial-fills
# tiles so it avoids the giant-tile budget-starvation that sinks tile_strict at
# low budget). Produces tiled_partial_* dirs alongside the existing tiled_whole_* dirs
# in the same OUT_BASE so exp2/exp3 can plot all packing modes together.
#
# exp2: whole-tile, 4 schemes (unchanged, already on disk)
# exp3: GS-progressive (culled/no-cull) + tiled (whole/partial)
#
# Env knobs (same defaults as run_exp123_merged.sh):
#   GPU, OUT_BASE, CACHE_DIR, ORACLE_DIR, ML_DIR, LOG_DIR, SKIP_EXISTING
#
# Run detached:  GPU=0 nohup bash experiments/0612/run_tile_partial.sh > /tmp/tp.log 2>&1 &
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL="$ROOT/dlapisgs-utility"
EXP="$ROOT/exp-dataset"
SCRIPT="$UTIL/test_utility_inmem.py"

OUT_BASE="${OUT_BASE:-$UTIL/output/0606/exp123_merged}"
CACHE_DIR="${CACHE_DIR:-$UTIL/output/0606/.tiling_cache}"
ORACLE_DIR="${ORACLE_DIR:-$UTIL/output/0606/exp4_oracle_dq}"
ML_DIR="${ML_DIR:-$UTIL/output/0606/ml_models}"
LOG_DIR="${LOG_DIR:-$UTIL/logs/tile_partial}"

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

# run_case SCENE CASE_TAG PACKING GRID_3 WEIGHT SCHEMES [EXTRA_ARGS]
run_case() {
    local scene="$1" case_tag="$2" packing="$3" weight="$5" schemes="$6"
    local grid="$4"
    local grid_str="${grid// /x}"
    local extra="${7:-}"

    local ply="$EXP/$scene/point_cloud.ply"
    local trace="$EXP/$scene/sparse_views_eval.json"
    local cache="$CACHE_DIR/${scene}_${grid_str}.npz"
    local out="$OUT_BASE/$scene/$case_tag"
    local log="$LOG_DIR/${scene}_${case_tag}.log"

    if [ ! -f "$ply" ];   then echo "[SKIP] $scene/$case_tag: PLY not found: $ply";   return; fi
    if [ ! -f "$trace" ]; then echo "[SKIP] $scene/$case_tag: trace not found: $trace"; return; fi
    if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$out/metrics/summary.csv" ]; then
        echo "[SKIP-EXIST] $scene/$case_tag: summary.csv present"; return
    fi

    echo "[RUN ] $scene / $case_tag  packing=$packing grid=$grid_str weight=$weight schemes=[$schemes]"
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
    echo "[DONE] $scene / $case_tag  -> $log"
}

for scene in $SCENES; do
    echo "=============================="
    echo "Scene: $scene   ($(date +%H:%M:%S))"
    echo "=============================="

    # tile_partial 8³ screen_area → vd_lod vd_lod_w
    run_case "$scene" tiled_partial_baseline_heuristic tile_partial "8 8 8" screen_area "vd_lod vd_lod_w"

    # tile_partial 8³ screen_area → oracle_loo ml  [prereq-gated]
    _oracle="$ORACLE_DIR/$scene/oracle_dq.npz"
    _mlpkl="$ML_DIR/$scene/rf.pkl"
    if [ -f "$_oracle" ] && [ -f "$_mlpkl" ]; then
        run_case "$scene" tiled_partial_oracle_ml tile_partial "8 8 8" screen_area "oracle_loo ml" \
            "--oracle-npz $_oracle --ml-model-dir $ML_DIR/$scene --ml-model-type rf"
    else
        echo "[SKIP] $scene/tiled_partial_oracle_ml: prereqs missing"
        [ -f "$_oracle" ] || echo "       missing: $_oracle"
        [ -f "$_mlpkl"  ] || echo "       missing: $_mlpkl"
    fi

    echo ""
done

echo "All scenes complete. ($(date +%H:%M:%S))"
echo "Outputs: $OUT_BASE/*/tiled_partial_{baseline_heuristic,oracle_ml}"
