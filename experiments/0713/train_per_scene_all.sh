#!/usr/bin/env bash
# Part C (closing out the ML comparison): train individual per-scene LGBM/AC models for
# all 10 scenes. Third arm alongside the two already-swept pooled variants (all10-pool,
# real3/synth7-split pool) -- per-scene models were fully deleted 2026-07-05 and never
# re-benchmarked under the current (post label-fix) pipeline. output/ml_models/ doesn't
# exist anywhere on disk (confirmed) -- this is a full retrain from scratch, not a rerun.
# Oracle training labels already exist for all 10 scenes at grid8 -- no oracle recompute.
#
# CPU-only (ml/train.py's fit_model() uses n_jobs=-1 internally, same as train_pooled.py --
# per CLAUDE.md's shared-workstation rule, run sequentially, one scene at a time, niced).
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility"
cd "$ROOT"

OUT_BASE="$ROOT/output/ml_models_experimental/per_scene"
ORACLE_ROOT="$ROOT/output/oracle/8/train"

SCENES="bicycle garden stump chair drums ficus hotdog materials mic ship"

for scene in $SCENES; do
    ORACLE_NPZ="$ORACLE_ROOT/$scene/oracle_dq.npz"
    OUT_DIR="$OUT_BASE/$scene"
    if [[ -f "$OUT_DIR/AC/feature_names.json" ]]; then
        echo "[skip] $scene: already trained"
        continue
    fi
    if [[ ! -f "$ORACLE_NPZ" ]]; then
        echo "[skip] $scene: no oracle_dq.npz at $ORACLE_NPZ"
        continue
    fi
    echo ""
    echo "====== $scene ======"
    nice -n 10 conda run -n gsquic python "$ROOT/ml/train.py" \
        --oracle-npz "$ORACLE_NPZ" \
        --output-dir "$OUT_DIR" \
        --ablations AC --models lgbm --seed 0
done

echo ""
echo "All per-scene trains done."
