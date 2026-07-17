#!/usr/bin/env bash
# Exp5 timing sweep, post 2026-07-17 optimization round: matmul-free project_covariance_2d
# (utility_calculation.py, 4.68x on the op) + real-budget (not identity-budget) greedy
# transfer in time_selection.py (greedy 79ms->11.8ms @ bicycle 40%). Structural reference:
# experiments/0702/run_timing_sweep.sh (do not run that script directly -- stale tiling
# cache / ML model dir paths, see CLAUDE.md "dated experiment scripts" rule).
#
# Scope: methods that actually exercise this session's two fixes (or serve as a no-op
# control) -- vd_lod (ord1-skip fix, no gaussian_weights), heuristic (both fixes, real
# weight order), progressive_screen_area (gaussian_weights, screen_area mode), ml (no
# gaussian_weights by default gs-order=ply -- control, should be near-unaffected). Default
# 8-level --budget-pct (10..100) for parity with the 0702 baseline sweep -- NOTE: this
# means the real-budget greedy fix will NOT show up in this sweep's numbers (max
# requested budget is 100%, same as the old identity-budget behavior) -- that fix's win is
# only visible in a single-small-budget call (see .claude/PLAN.md 2026-07-17 entry for
# the isolated bicycle 40% numbers). This sweep isolates the project_covariance_2d win.
#
# Design date 2026-07-17, run date 2026-07-17 -> output/0717/selection_timing_post_optimization/.
set -euo pipefail
cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility
ROOT=/mnt/data1/samk/gs-quic/cs5262_tile_quic
OUT=output/0717/selection_timing_post_optimization

declare -A PLY CACHE TRACE MLDIR
for SCENE in chair drums ficus hotdog materials mic ship; do
  PLY[$SCENE]="${ROOT}/exp-dataset/${SCENE}/checkpoint/point_cloud/iteration_30000/point_cloud.ply"
  CACHE[$SCENE]="output/oracle_tiling_cache/${SCENE}_8x8x8.npz"
  TRACE[$SCENE]="${ROOT}/exp-dataset/${SCENE}/sparse_views_eval.json"
  MLDIR[$SCENE]="output/ml_models_experimental/per_scene/${SCENE}/AC"
done
for SCENE in bicycle garden stump; do
  PLY[$SCENE]="${ROOT}/exp-dataset/${SCENE}/point_cloud.ply"
  CACHE[$SCENE]="output/oracle_tiling_cache/${SCENE}_8x8x8.npz"
  TRACE[$SCENE]="${ROOT}/exp-dataset/${SCENE}/sparse_views_eval.json"
  MLDIR[$SCENE]="output/ml_models_experimental/per_scene/${SCENE}/AC"
done

run_scene() {
  local SCENE=$1
  echo "########## $SCENE: vd_lod heuristic progressive_screen_area (150 cams) ##########"
  time conda run -n gaussian_splatting python time_selection.py \
    --ply "${PLY[$SCENE]}" --tiling-cache "${CACHE[$SCENE]}" --camera-trace "${TRACE[$SCENE]}" \
    --methods vd_lod heuristic progressive_screen_area \
    --output-root "$OUT" --scene "$SCENE"

  echo "########## $SCENE: ml lgbm (150 cams) ##########"
  time conda run -n gaussian_splatting python time_selection.py \
    --ply "${PLY[$SCENE]}" --tiling-cache "${CACHE[$SCENE]}" --camera-trace "${TRACE[$SCENE]}" \
    --methods ml --ml-model-dir "${MLDIR[$SCENE]}" --ml-model-type lgbm \
    --output-root "$OUT" --scene "$SCENE"
}

# Balanced by prior 0702 sweep's GPU split (real scenes + heaviest synthetics on GPU1).
GPU0_SCENES=(materials hotdog ship mic)
GPU1_SCENES=(ficus chair drums bicycle stump garden)

(
  export CUDA_VISIBLE_DEVICES=2
  for SCENE in "${GPU0_SCENES[@]}"; do run_scene "$SCENE"; done
  echo "GPU2 DONE"
) &
PID0=$!

(
  export CUDA_VISIBLE_DEVICES=3
  for SCENE in "${GPU1_SCENES[@]}"; do run_scene "$SCENE"; done
  echo "GPU3 DONE"
) &
PID1=$!

wait "$PID0" "$PID1"
echo "ALL DONE"
