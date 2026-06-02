#!/usr/bin/env bash
# 2026-06-02: complete Exp 1 (GS weight-mode sweep) at 1600². screen_area@1600² already
# exists = output/0601/exp3_packing/progressive/{scene} (progressive is tile-agnostic, so
# the prog@8³ screen_area curve IS exp1 screen_area). This fills the missing two modes:
# volume + volume_over_d2 at prog@8³, 1600². No ML → no train/eval split; trace is matched
# to exp3 (sparse_views_100_eval.json) only so the three weight-mode curves are comparable.
# Tiling cache reused from the 0514 8³ tiling to skip re-tiling.
#
#   CUDA_VISIBLE_DEVICES=2 nohup bash experiments/0602/run_exp1_1600.sh \
#       > /tmp/exp1_1600_volmodes.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT_ROOT="${OUT_ROOT:-output/0602/exp1_1600}"
BUDGETS="${BUDGETS:-10 25 40 55 70 85 99 100}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# shellcheck disable=SC2206
WMODES=(${WMODES:-volume volume_over_d2})
SCENES=(${SCENES:-chair drums ficus hotdog materials mic})

scene_ply()   { echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/point_cloud.ply"; }
scene_trace() { echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/sparse_views_100_eval.json"; }
scene_cache() { echo "${ROOT}/dlapisgs-utility/output/0514/exp1_gs_weight_done/$1/.tiling_cache.npz"; }

for WM in "${WMODES[@]}"; do
  for SCENE in "${SCENES[@]}"; do
    PLY="$(scene_ply "$SCENE")"; TRACE="$(scene_trace "$SCENE")"; CACHE="$(scene_cache "$SCENE")"
    OUT_DIR="${OUT_ROOT}/${WM}/${SCENE}"
    [ -f "$PLY" ]   || { echo "SKIP $SCENE — no PLY";   continue; }
    [ -f "$TRACE" ] || { echo "SKIP $SCENE — no trace"; continue; }
    mkdir -p "$OUT_DIR"
    CACHE_ARG=(); [ -f "$CACHE" ] && CACHE_ARG=(--tiling-cache "$CACHE")

    echo "=== [$WM/$SCENE] selection (progressive @8³, 1600²) ==="
    conda run -n gsquic python test_utility.py \
        --ply "$PLY" --output-root "$OUT_DIR" --camera-trace "$TRACE" \
        --grid-shape 8 8 8 --budget-pct $BUDGETS \
        --schemes vd_lod --num-lod 1 --camera-index -1 \
        --packing-mode progressive --weight-mode "$WM" \
        --w-norm sum --c-norm sum --img-w 1600 --img-h 1600 "${CACHE_ARG[@]}"

    echo "=== [$WM/$SCENE] render_metrics ==="
    conda run -n gaussian_splatting python experiments/render_metrics.py \
        --output-root "$OUT_DIR" --gt-ply "$PLY" --trace "$TRACE" \
        --scene "$SCENE" --render-dir "${OUT_DIR}/renders" --delete-ply
    echo "[$WM/$SCENE] done — ${OUT_DIR}/metrics/summary.csv"
  done
done
echo "All exp1-1600 weight modes done. Output: $OUT_ROOT"
