#!/usr/bin/env bash
# ML retrain on clean float32 oracle (Step iv). RF+LGBM+XGB, AC ablation, defaults.
# Trains in gaussian_splatting env (sklearn 1.0.2) for selection-loadable pickles.
#
# Env overrides:
#   ORACLE_ROOT (default: output/oracle/8/train)
#   ML_ROOT     (default: output/ml_models/8)
#   SCENES      (default: all 10)
set -euo pipefail
cd "$(dirname "$0")/../.."

UTIL="$PWD"
ORACLE_ROOT="${ORACLE_ROOT:-$UTIL/output/oracle/8/train}"
ML_ROOT="${ML_ROOT:-$UTIL/output/ml_models/8}"
LOG_DIR="$UTIL/logs/ml_train_0630"
SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"

mkdir -p "$LOG_DIR"
echo "ORACLE_ROOT=$ORACLE_ROOT  ML_ROOT=$ML_ROOT  SCENES=$SCENES"

for scene in $SCENES; do
    oracle_npz="$ORACLE_ROOT/$scene/oracle_dq.npz"
    out_dir="$ML_ROOT/$scene"
    log="$LOG_DIR/${scene}.log"
    if [ ! -f "$oracle_npz" ]; then echo "[SKIP] $scene: no oracle"; continue; fi
    if [ -f "$out_dir/AC/rf.pkl" ]; then echo "[SKIP] $scene: already trained"; continue; fi
    echo "--- [$scene] train RF+LGBM+XGB (AC) --- $(date +%H:%M:%S)"
    mkdir -p "$out_dir"
    conda run -n gaussian_splatting python "$UTIL/ml/train.py" \
        --oracle-npz "$oracle_npz" --output-dir "$out_dir" \
        --ablations AC --models rf lgbm xgb \
        --label-mode raw --seed 0 > "$log" 2>&1
    ln -sf "AC/rf.pkl" "$out_dir/rf.pkl"
    ln -sf "AC/feature_names.json" "$out_dir/feature_names.json"
    echo "[$scene] done -> $out_dir"
done
echo "Done. Models: $ML_ROOT"
