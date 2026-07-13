#!/usr/bin/env bash
# Retrain every deployed pooled ML model after the ml/features.py label-censoring fix
# (2026-07-09): build_feature_matrix used to NaN-drop every mse_loo<=0 (cam,tile) row
# instead of flooring it, silently discarding ~90%+ of near-zero-importance training
# rows for tiny tiles on real scenes at fine grids -- traced as the root cause of a
# real-scene-only PSNR collapse at grid16 in the Exp4 rerun. Now floors at 1e-30 before
# log() instead. This script retrains every pooled model that depends on that label
# path, in place, at the same output dirs (overwrite is the normal retrain workflow for
# these "experimental" model dirs, not a mutation-for-naming issue).
#
# Runs sequentially (one training subprocess at a time) -- each fit_model() call uses
# n_jobs=-1 internally; per CLAUDE.md's shared-workstation rule, concurrent n_jobs=-1
# jobs caused a load-average incident before, so this script never launches more than
# one at once. `nice` deprioritizes it so it yields to interactive/other users' work.
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # GPU0 banned, GPU2 was free at launch time

OUT_BASE="$ROOT/output/ml_models_experimental"

run() {
    local scene_set="$1" grid="$2" out="$3"
    echo ""
    echo "====== scene-set=${scene_set} grid=${grid} -> ${out} ======"
    nice -n 10 conda run -n gsquic python "$ROOT/ml/cross_scene/train_pooled.py" \
        --scene-set "$scene_set" --ablation AC --models lgbm --seed 0 \
        --root "$ROOT/output/oracle/${grid}" --out "$out"
}

# all10, every grid (grid8 is the canonical/deployed dir; grid8 symlink already points
# pooled_all10_ACG_grid8 -> pooled_all10_ACG, so no separate grid8 target needed here)
run all10 1  "$OUT_BASE/pooled_all10_ACG_grid1"
run all10 2  "$OUT_BASE/pooled_all10_ACG_grid2"
run all10 4  "$OUT_BASE/pooled_all10_ACG_grid4"
run all10 8  "$OUT_BASE/pooled_all10_ACG"
run all10 16 "$OUT_BASE/pooled_all10_ACG_grid16"

# domain-split pooling, grid8 only (never trained at other grids)
run real3  8 "$OUT_BASE/pooled_real3_ACG"
run synth7 8 "$OUT_BASE/pooled_synth7_ACG"

echo ""
echo "All retrains done."
