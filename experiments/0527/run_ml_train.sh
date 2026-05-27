#!/usr/bin/env bash
# Train ML models for all 8 scenes (all 4 ablations × 3 model types each).
# Output: ml/models/{scene}/{ABCD,ACD,AC,ABC}/
# ETA: ~1 hr total on 1 GPU.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash experiments/0527/run_ml_train.sh
#   (or CUDA_VISIBLE_DEVICES=3)

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ORACLE_ROOT="output/0522/exp4_oracle_dq"
MODEL_ROOT="ml/models"
LOG_DIR="output/0527/ml_train_logs"
mkdir -p "$LOG_DIR"

SCENES=(chair drums ficus hotdog materials mic ship bicycle)

for SCENE in "${SCENES[@]}"; do
    ORACLE_NPZ="${ORACLE_ROOT}/${SCENE}/oracle_dq.npz"
    OUT_DIR="${MODEL_ROOT}/${SCENE}"

    if [ ! -f "$ORACLE_NPZ" ]; then
        echo "SKIP $SCENE — oracle_dq.npz not found: $ORACLE_NPZ"
        continue
    fi

    echo "Training ${SCENE} -> ${OUT_DIR} ..."
    LOG_FILE="${LOG_DIR}/${SCENE}.log"

    conda run -n gsquic python ml/train.py \
        --oracle-npz "$ORACLE_NPZ" \
        --output-dir "$OUT_DIR" \
        --seed 0 \
        2>&1 | tee "$LOG_FILE"

    echo "Done ${SCENE}. Log: ${LOG_FILE}"
done

echo ""
echo "All scenes done. Models in ${MODEL_ROOT}/"
