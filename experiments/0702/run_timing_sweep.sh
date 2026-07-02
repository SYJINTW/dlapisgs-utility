#!/usr/bin/env bash
# Exp5 full timing sweep: 10 scenes, 150 cams, 8 budgets, fast methods + oracle_online + ml_lgbm.
# Design date 2026-07-02, run date 2026-07-02 -> output/0702/selection_timing/.
# oracle_online now visibility-culls (skip LOO render of invisible tiles, floor -inf) --
# chair+bicycle oracle_online numbers get overwritten with the fixed, culled numbers.
# Split across GPU 0 / GPU 1 (deadline-driven, balanced by estimated wall time).
set -euo pipefail
cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility
ROOT=/mnt/data1/samk/gs-quic/cs5262_tile_quic
OUT=output/0702/selection_timing

declare -A PLY CACHE TRACE MLDIR
for SCENE in chair drums ficus hotdog materials mic ship bicycle garden stump; do
  PLY[$SCENE]="${ROOT}/exp-dataset/${SCENE}/point_cloud.ply"
  CACHE[$SCENE]="output/0605/exp1_gs_weights/${SCENE}/.tiling_cache.npz"
  TRACE[$SCENE]="${ROOT}/exp-dataset/${SCENE}/sparse_views_eval.json"
  MLDIR[$SCENE]="output/ml_models/8/${SCENE}/AC"
done

run_scene() {
  local SCENE=$1
  echo "########## $SCENE: fast methods + oracle_online (150 cams) ##########"
  time conda run -n gaussian_splatting python time_selection.py \
    --ply "${PLY[$SCENE]}" --tiling-cache "${CACHE[$SCENE]}" --camera-trace "${TRACE[$SCENE]}" \
    --methods vd_lod heuristic progressive_screen_area progressive_vol_d2 oracle_online \
    --output-root "$OUT" --scene "$SCENE"

  echo "########## $SCENE: ml lgbm (150 cams) ##########"
  time conda run -n gaussian_splatting python time_selection.py \
    --ply "${PLY[$SCENE]}" --tiling-cache "${CACHE[$SCENE]}" --camera-trace "${TRACE[$SCENE]}" \
    --methods ml --ml-model-dir "${MLDIR[$SCENE]}" --ml-model-type lgbm \
    --output-root "$OUT" --scene "$SCENE"
}

# Balanced by estimated wall time (2026-07-02 canary-based projection, see PLAN.md item vii).
GPU0_SCENES=(materials hotdog ship mic)
GPU1_SCENES=(ficus chair drums bicycle stump garden)

(
  export CUDA_VISIBLE_DEVICES=0
  for SCENE in "${GPU0_SCENES[@]}"; do run_scene "$SCENE"; done
  echo "GPU0 DONE"
) &
PID0=$!

(
  export CUDA_VISIBLE_DEVICES=1
  for SCENE in "${GPU1_SCENES[@]}"; do run_scene "$SCENE"; done
  echo "GPU1 DONE"
) &
PID1=$!

wait "$PID0" "$PID1"
echo "ALL DONE"
