#!/usr/bin/env bash
# Resume of run_timing_sweep_grid24.sh after CUDA OOM on ship/grid2/oracle_online
# (GPU2 contention with the concurrently-launched exp2_rerun + diag_jaccard sweeps).
# ship's other 4 methods already completed under grid2 -- only oracle_online is missing there.
set -euo pipefail
cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility
ROOT=/mnt/data1/samk/gs-quic/cs5262_tile_quic
OUT_BASE=output/0703/selection_timing

declare -A PLY TRACE
for SCENE in chair drums ficus hotdog materials mic ship bicycle garden stump; do
  PLY[$SCENE]="${ROOT}/exp-dataset/${SCENE}/point_cloud.ply"
  TRACE[$SCENE]="${ROOT}/exp-dataset/${SCENE}/sparse_views_eval.json"
done

export CUDA_VISIBLE_DEVICES=0

# --- finish grid2: ship's missing oracle_online, then bicycle/garden/stump (all methods) ---
GRID=2
OUT="$OUT_BASE/grid${GRID}"
CACHE="output/0702/tiling_grid_sweep/ship/grid${GRID}/.tiling_cache.npz"
echo "########## grid${GRID} ship oracle_online (resume) ##########"
time conda run -n gaussian_splatting python time_selection.py \
  --ply "${PLY[ship]}" --tiling-cache "$CACHE" --camera-trace "${TRACE[ship]}" \
  --methods oracle_online --output-root "$OUT" --scene ship

for SCENE in bicycle garden stump; do
  CACHE="output/0702/tiling_grid_sweep/${SCENE}/grid${GRID}/.tiling_cache.npz"
  echo "########## grid${GRID} $SCENE (150 cams) ##########"
  time conda run -n gaussian_splatting python time_selection.py \
    --ply "${PLY[$SCENE]}" --tiling-cache "$CACHE" --camera-trace "${TRACE[$SCENE]}" \
    --methods vd_lod heuristic progressive_screen_area progressive_vol_d2 oracle_online \
    --output-root "$OUT" --scene "$SCENE"
done

# --- grid4: full 10-scene sweep, never started ---
GRID=4
OUT="$OUT_BASE/grid${GRID}"
for SCENE in chair drums ficus hotdog materials mic ship bicycle garden stump; do
  CACHE="output/0702/tiling_grid_sweep/${SCENE}/grid${GRID}/.tiling_cache.npz"
  echo "########## grid${GRID} $SCENE (150 cams) ##########"
  time conda run -n gaussian_splatting python time_selection.py \
    --ply "${PLY[$SCENE]}" --tiling-cache "$CACHE" --camera-trace "${TRACE[$SCENE]}" \
    --methods vd_lod heuristic progressive_screen_area progressive_vol_d2 oracle_online \
    --output-root "$OUT" --scene "$SCENE"
done
echo "ALL DONE"
