#!/usr/bin/env bash
# Exp5 full timing sweep: chair + bicycle, 150 cams, 8 budgets, 7 methods.
# Design date 2026-07-02, run date 2026-07-02 -> output/0702/selection_timing/.
# Post-fix (greedy_order GPU sort + --gs-order coupling landed same session).
set -euo pipefail
cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility
export CUDA_VISIBLE_DEVICES=0
ROOT=/mnt/data1/samk/gs-quic/cs5262_tile_quic
OUT=output/0702/selection_timing

declare -A PLY=( [chair]="${ROOT}/exp-dataset/chair/checkpoint/point_cloud/iteration_30000/point_cloud.ply"
                 [bicycle]="${ROOT}/exp-dataset/bicycle/point_cloud.ply" )
declare -A CACHE=( [chair]="output/0605/exp1_gs_weights/chair/.tiling_cache.npz"
                    [bicycle]="output/0605/exp1_gs_weights/bicycle/.tiling_cache.npz" )
declare -A TRACE=( [chair]="${ROOT}/exp-dataset/chair/sparse_views_eval.json"
                    [bicycle]="${ROOT}/exp-dataset/bicycle/sparse_views_eval.json" )

for SCENE in chair bicycle; do
  echo "########## $SCENE: fast methods + oracle_online (150 cams) ##########"
  time conda run -n gaussian_splatting python time_selection.py \
    --ply "${PLY[$SCENE]}" --tiling-cache "${CACHE[$SCENE]}" --camera-trace "${TRACE[$SCENE]}" \
    --methods vd_lod heuristic progressive_screen_area progressive_vol_d2 oracle_online \
    --output-root "$OUT" --scene "$SCENE"

  echo "########## $SCENE: ml rf (150 cams) ##########"
  time conda run -n gaussian_splatting python time_selection.py \
    --ply "${PLY[$SCENE]}" --tiling-cache "${CACHE[$SCENE]}" --camera-trace "${TRACE[$SCENE]}" \
    --methods ml --ml-model-dir "output/ml_models/8/${SCENE}/AC" --ml-model-type rf \
    --output-root "$OUT" --scene "$SCENE"

  echo "########## $SCENE: ml lgbm (150 cams) ##########"
  time conda run -n gaussian_splatting python time_selection.py \
    --ply "${PLY[$SCENE]}" --tiling-cache "${CACHE[$SCENE]}" --camera-trace "${TRACE[$SCENE]}" \
    --methods ml --ml-model-dir "output/ml_models/8/${SCENE}/AC" --ml-model-type lgbm \
    --output-root "$OUT" --scene "$SCENE"
done

echo "ALL DONE"
