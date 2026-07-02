#!/usr/bin/env bash
# Full 10-scene / 150-cam sweep for diag_topk_precision.py (Open item 6 investigation).
# Launches each scene detached (nohup, disowned) round-robined across 4 GPUs so this
# survives the launching shell exiting. Canary (chair, 5 cams) already verified clean.
#
# Usage: ./experiments/0702/run_diag_topk.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
mkdir -p logs output/diag_topk

SCENES=(chair drums ficus hotdog materials mic ship bicycle garden stump)
GPU=(0 1 2 3)

i=0
for SCENE in "${SCENES[@]}"; do
  G=${GPU[$((i % 4))]}
  PLY="${ROOT}/exp-dataset/${SCENE}/point_cloud.ply"
  TRACE="${ROOT}/exp-dataset/${SCENE}/sparse_views_eval.json"
  ORACLE="output/oracle/8/eval/${SCENE}/oracle_dq.npz"
  MODEL_DIR="output/ml_models/8/${SCENE}/AC"
  OUT="output/diag_topk/${SCENE}"
  LOG="logs/diag_topk_${SCENE}.log"

  nohup env CUDA_VISIBLE_DEVICES="$G" conda run -n gaussian_splatting python \
    experiments/0702/diag_topk_precision.py \
    --scene "$SCENE" --ply "$PLY" --oracle-npz "$ORACLE" \
    --camera-trace "$TRACE" --ml-model-dir "$MODEL_DIR" \
    --out-dir "$OUT" \
    > "$LOG" 2>&1 &
  disown
  echo "launched $SCENE on GPU $G -> $LOG (pid $!)"
  i=$((i + 1))
done
