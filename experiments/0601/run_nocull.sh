#!/usr/bin/env bash
# 2026-06-02: "no-cull floor" = progressive packing at grid 1x1x1 (single whole-scene
# tile → always visible → no FOV trim → pure global screen_area greedy). Isolates the
# value of FOV culling: compare vs progressive@8³ (cull) and tile_strict@8³ (cull+order).
# No oracle / no model needed (progressive ignores scheme; vd_lod needs neither).
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash experiments/0601/run_nocull.sh \
#       > /tmp/nocull_0601.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT_ROOT="${OUT_ROOT:-output/0601/exp3_nocull}"
BUDGETS="${BUDGETS:-10 25 40 55 70 85 99 100}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# shellcheck disable=SC2206
SCENES=(${SCENES:-chair drums ficus hotdog materials mic ship})

scene_ply()   { echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/point_cloud.ply"; }
scene_trace() { echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/sparse_views_100_eval.json"; }

mkdir -p "$OUT_ROOT"
for SCENE in "${SCENES[@]}"; do
    PLY="$(scene_ply "$SCENE")"; TRACE="$(scene_trace "$SCENE")"
    OUT_DIR="${OUT_ROOT}/${SCENE}"
    [ -f "$PLY" ]   || { echo "SKIP $SCENE — no PLY";   continue; }
    [ -f "$TRACE" ] || { echo "SKIP $SCENE — no trace"; continue; }
    mkdir -p "$OUT_DIR"

    echo "=== [$SCENE] no-cull selection (progressive @ 1x1x1) ==="
    conda run -n gsquic python test_utility.py \
        --ply "$PLY" --output-root "$OUT_DIR" --camera-trace "$TRACE" \
        --grid-shape 1 1 1 --budget-pct $BUDGETS \
        --schemes vd_lod --num-lod 1 --camera-index "$CAMERA_INDEX" \
        --packing-mode progressive --weight-mode screen_area \
        --w-norm sum --c-norm sum --img-w 1600 --img-h 1600 \
        2>&1 | tee "${OUT_DIR%/}_sel.log"

    echo "=== [$SCENE] render_metrics ==="
    conda run -n gaussian_splatting python experiments/render_metrics.py \
        --output-root "$OUT_DIR" --gt-ply "$PLY" --trace "$TRACE" \
        --scene "$SCENE" --render-dir "${OUT_DIR}/renders" --delete-ply \
        2>&1 | tee "${OUT_DIR%/}_render.log"
    echo "[$SCENE] done — ${OUT_DIR}/metrics/summary.csv"
done
echo "All no-cull scenes done. Output: $OUT_ROOT"
