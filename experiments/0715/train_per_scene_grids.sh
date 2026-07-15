#!/usr/bin/env bash
# Train per-scene LGBM/AC models at grid{1,2,4,16} -- needed for Exp4's gs_order-bug rerun
# (see PLAN.md "Bug pattern found (2026-07-15)"). grid8 per-scene models already exist
# (output/ml_models_experimental/per_scene/<scene>/AC/, trained 2026-07-13) and are unaffected
# by the bug -- not retrained here. Old pooled_all10_ACG_grid{1,2,4,16} models (archived
# 2026-07-14 to ../VCUTS_backup/, retired in favor of per-scene) are stale on top of the bug --
# this brings grid1/2/4/16 up to the same per-scene-model canon grid8 already has.
#
# Oracle labels already exist at every grid, all 10 scenes (output/oracle/{g}/train/<scene>/
# oracle_dq.npz) -- no oracle regen needed, confirmed 2026-07-15.
#
# CPU-only (ml/train.py's fit_model() uses n_jobs=-1 internally) -- sequential, one
# (scene,grid) at a time, niced, per CLAUDE.md shared-workstation rule. ~36s/model measured
# (2026-07-13 per-scene grid8 run) -> 40 models ~= ~24min.
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility"
cd "$ROOT"

OUT_BASE="$ROOT/output/ml_models_experimental"
SCENES="bicycle garden stump chair drums ficus hotdog materials mic ship"
GRIDS="1 2 4 16"

for g in $GRIDS; do
    ORACLE_ROOT="$ROOT/output/oracle/$g/train"
    for scene in $SCENES; do
        ORACLE_NPZ="$ORACLE_ROOT/$scene/oracle_dq.npz"
        OUT_DIR="$OUT_BASE/per_scene_grid${g}/$scene"
        if [[ -f "$OUT_DIR/AC/feature_names.json" ]]; then
            echo "[skip] grid${g}/$scene: already trained"
            continue
        fi
        if [[ ! -f "$ORACLE_NPZ" ]]; then
            echo "[skip] grid${g}/$scene: no oracle_dq.npz at $ORACLE_NPZ"
            continue
        fi
        echo ""
        echo "====== grid${g} / $scene ======"
        nice -n 10 conda run -n gsquic python "$ROOT/ml/train.py" \
            --oracle-npz "$ORACLE_NPZ" \
            --output-dir "$OUT_DIR" \
            --ablations AC --models lgbm --seed 0
    done
done

echo ""
echo "All per-scene-per-grid trains done."
