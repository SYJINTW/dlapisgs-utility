#!/usr/bin/env bash
# Train per-grid RF models from per-grid oracle NPZs.
# Env overrides:
#   GRIDS="1 2 4"
#   SCENES="bicycle garden ..."
#   ORACLE_ROOT=output/0618/oracle_grid     # expects {ORACLE_ROOT}/{g}/{scene}/oracle_dq.npz
#   ML_OUT=output/0618/ml_models            # writes {ML_OUT}/{g}/{scene}/AC/{rf.pkl,feature_names.json,metrics.json}
#   ABLATIONS="AC"                          # passed to --ablations (default: AC only)
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
UTIL_DIR="$ROOT/dlapisgs-utility"

GRIDS="${GRIDS:-1 2 4}"
SCENES="${SCENES:-bicycle garden stump chair drums ficus hotdog materials mic ship}"
ORACLE_ROOT="${ORACLE_ROOT:-$UTIL_DIR/output/0618/oracle_grid}"
ML_OUT="${ML_OUT:-$UTIL_DIR/output/0618/ml_models}"
ABLATIONS="${ABLATIONS:-AC}"

echo "=========================================="
echo "Per-grid ML retrain (RF)"
echo "GRIDS       : $GRIDS"
echo "ORACLE_ROOT : $ORACLE_ROOT/{g}/{scene}/oracle_dq.npz"
echo "ML_OUT      : $ML_OUT/{g}/{scene}/"
echo "ABLATIONS   : $ABLATIONS"
echo "=========================================="

for g in $GRIDS; do
    echo ""
    echo "====== grid=${g}x${g}x${g} ======"
    for scene in $SCENES; do
        NPZ="$ORACLE_ROOT/$g/$scene/oracle_dq.npz"
        OUT="$ML_OUT/$g/$scene"
        if [[ ! -f "$NPZ" ]]; then
            echo "[skip] $g/$scene: no oracle_dq.npz at $NPZ"
            continue
        fi
        if [[ -f "$OUT/AC/rf.pkl" ]]; then
            echo "[skip] $g/$scene: already trained"
            continue
        fi
        echo "--- $g/$scene ---"
        mkdir -p "$OUT"
        conda run -n gaussian_splatting python "$UTIL_DIR/ml/train.py" \
            --oracle-npz "$NPZ" \
            --output-dir "$OUT" \
            --models rf \
            --ablations $ABLATIONS
    done
done

echo ""
echo "ALL ML GRID DONE $(date)"
