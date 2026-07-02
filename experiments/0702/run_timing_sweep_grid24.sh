#!/usr/bin/env bash
# Exp5 grid-size retry: same 10 scenes/150 cams/8 budgets as run_timing_sweep.sh (grid8,
# output/0702/selection_timing/), rerun at grid2 and grid4 to check whether coarser tiling
# (far fewer visible tiles/cam -> fewer oracle_online LOO renders) meaningfully cuts wall time.
# See PLAN.md item (viii) + diag_visible_frac.py findings (2026-07-02): grid2 synth scenes
# have ~7 visible tiles/cam vs ~207 at grid8 -- expect oracle_online to drop sharply.
# Design date 2026-07-02, run date 2026-07-03 (user-specified) -> output/0703/.
# ml skipped: output/ml_models/8/{scene}/AC was trained on grid8 tiles, not valid for grid2/4.
# Single GPU (2) -- GPUs 0/1/3 occupied by the concurrent diag_jaccard_overlap sweep.
set -euo pipefail
cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility
ROOT=/mnt/data1/samk/gs-quic/cs5262_tile_quic
OUT_BASE=output/0703/selection_timing

declare -A PLY TRACE
for SCENE in chair drums ficus hotdog materials mic ship bicycle garden stump; do
  PLY[$SCENE]="${ROOT}/exp-dataset/${SCENE}/point_cloud.ply"
  TRACE[$SCENE]="${ROOT}/exp-dataset/${SCENE}/sparse_views_eval.json"
done

export CUDA_VISIBLE_DEVICES=2

for GRID in 2 4; do
  OUT="$OUT_BASE/grid${GRID}"
  for SCENE in chair drums ficus hotdog materials mic ship bicycle garden stump; do
    CACHE="output/0702/tiling_grid_sweep/${SCENE}/grid${GRID}/.tiling_cache.npz"
    echo "########## grid${GRID} $SCENE (150 cams) ##########"
    time conda run -n gaussian_splatting python time_selection.py \
      --ply "${PLY[$SCENE]}" --tiling-cache "$CACHE" --camera-trace "${TRACE[$SCENE]}" \
      --methods vd_lod heuristic progressive_screen_area progressive_vol_d2 oracle_online \
      --output-root "$OUT" --scene "$SCENE"
  done
done
echo "ALL DONE"
