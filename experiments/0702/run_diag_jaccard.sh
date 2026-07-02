#!/usr/bin/env bash
# Full 10-scene / 150-cam sweep for diag_jaccard_overlap.py. Round-robins across 4
# GPUs, detached (nohup, disowned). Canary (chair 3 cams, bicycle 5 cams) verified
# clean under the post-w_mode-fix script. Real scenes (bicycle/garden/stump, ~5-6M
# Gaussians each) run ~25s/cam -> ~60-70 min; synthetic scenes (~150-350K GS) ~2-3s/cam
# -> ~6-8 min.
#
# Usage: ./experiments/0702/run_diag_jaccard.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
mkdir -p logs output/diag_jaccard

SCENES=(chair drums ficus hotdog materials mic ship bicycle garden stump)
GPU=(0 1 2 3)

i=0
for SCENE in "${SCENES[@]}"; do
  G=${GPU[$((i % 4))]}
  PLY="${ROOT}/exp-dataset/${SCENE}/point_cloud.ply"
  TRACE="${ROOT}/exp-dataset/${SCENE}/sparse_views_eval.json"
  ORACLE="output/oracle/8/eval/${SCENE}/oracle_dq.npz"
  MODEL_DIR="output/ml_models/8/${SCENE}/AC"
  OUT="output/diag_jaccard/${SCENE}"
  LOG="logs/diag_jaccard_${SCENE}.log"

  nohup env CUDA_VISIBLE_DEVICES="$G" conda run -n gaussian_splatting python \
    experiments/0702/diag_jaccard_overlap.py \
    --scene "$SCENE" --ply "$PLY" --oracle-npz "$ORACLE" \
    --camera-trace "$TRACE" --ml-model-dir "$MODEL_DIR" \
    --out-dir "$OUT" \
    > "$LOG" 2>&1 &
  disown
  echo "launched $SCENE on GPU $G -> $LOG (pid $!)"
  i=$((i + 1))
done
