#!/usr/bin/env bash
# Train RF / LGBM / XGB on clean train-camera oracle (no data snooping).
# Feature set: AC only.
# RF: fixed msl=2, mf=0.3 (from prior sweep).
# LGBM: sweep num_leaves × min_child_samples.
# XGB:  sweep max_depth × subsample.
#
# Usage:
#   conda run -n gsquic bash experiments/0527/run_ml_train_clean.sh
#
# Prereq: output/0527_clean/exp4_oracle_dq/train/{scene}/oracle_dq.npz

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ORACLE_ROOT="output/0527_clean/exp4_oracle_dq/train"
MODEL_ROOT="ml/models_clean"
LOG_DIR="output/0527_clean/ml_train_logs"
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
        --ablations AC \
        --models rf lgbm xgb \
        --cv-folds 5 \
        --seed 0 \
        --rf-min-samples-leaf 2 \
        --rf-max-features 0.3 \
        --lgbm-sweep \
        --xgb-sweep \
        2>&1 | tee "$LOG_FILE"

    echo "Done ${SCENE}. Log: ${LOG_FILE}"
done

echo ""
echo "All scenes done. Models in ${MODEL_ROOT}/"
echo "Verify: cat ${LOG_DIR}/chair.log | grep -E 'RF|LGBM|XGB|Best'"
