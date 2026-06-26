#!/usr/bin/env bash
# ML model training — RF + AC ablation (best per concluded_experiments.md).
# Skips scenes where oracle_dq.npz not yet available.
# Skips scenes where rf.pkl already exists (safe to rerun).
#
# Input:  ORACLE_ROOT/{scene}/oracle_dq.npz
# Output: ML_ROOT/{scene}/AC/rf.pkl  +  ML_ROOT/{scene}/rf.pkl (flat symlink)
#
# Env overrides:
#   ORACLE_ROOT  (default: output/0606/exp4_oracle_dq/train)
#   ML_ROOT      (default: output/0606/ml_models)
#   SCENES       (default: all 10)
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL="$ROOT/dlapisgs-utility"
ORACLE_ROOT="${ORACLE_ROOT:-$UTIL/output/0606/exp4_oracle_dq/train}"
ML_ROOT="${ML_ROOT:-$UTIL/output/0606/ml_models}"
LOG_DIR="$UTIL/logs/ml_train_0606"

SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "ML training — RF + AC ablation"
echo "ORACLE_ROOT : $ORACLE_ROOT"
echo "ML_ROOT     : $ML_ROOT"
echo "SCENES      : $SCENES"
echo "=========================================="

for scene in $SCENES; do
    oracle_npz="$ORACLE_ROOT/$scene/oracle_dq.npz"
    out_dir="$ML_ROOT/$scene"
    flat_pkl="$out_dir/rf.pkl"
    log="$LOG_DIR/${scene}.log"

    if [ ! -f "$oracle_npz" ]; then
        echo "[SKIP] $scene: oracle not ready at $oracle_npz"
        continue
    fi

    if [ -f "$flat_pkl" ]; then
        echo "[SKIP] $scene: already trained ($flat_pkl exists)"
        continue
    fi

    echo ""
    echo "--- [$scene] training RF (AC) ---  ($(date +%H:%M:%S))"
    mkdir -p "$out_dir"

    # Train in the SELECTION env (gaussian_splatting, sklearn 1.0.2) so the
    # pickled RF is loadable at selection time. Training in gsquic (sklearn 1.6)
    # yields models the py3.7 selection env cannot unpickle (tree-node dtype
    # gained `missing_go_to_left` in sklearn >=1.3).
    conda run -n gaussian_splatting python "$UTIL/ml/train.py" \
        --oracle-npz "$oracle_npz" \
        --output-dir "$out_dir" \
        --ablations AC \
        --models rf \
        --seed 0 \
        > "$log" 2>&1

    ln -sf "AC/rf.pkl" "$flat_pkl"
    ln -sf "AC/feature_names.json" "$out_dir/feature_names.json"  # predict.py needs both flat
    echo "[$scene] done -> $flat_pkl"
done

echo ""
echo "Done. Models: $ML_ROOT"
