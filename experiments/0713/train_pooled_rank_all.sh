#!/usr/bin/env bash
# Train pooled LGBMRanker models (candidate 1 x candidate 5, deferred 2026-07-03, revisited
# 2026-07-13) at all3 scene-set granularities, for a matched-provenance comparison against the
# already-deployed pooled regressors (pooled_all10_ACG / pooled_real3_ACG / pooled_synth7_ACG).
# All prior rank-model output was deleted in the 2026-07-05 ~113GB cleanup -- this is a full
# retrain from scratch. ml/cross_scene/train_pooled_rank.py is unmodified; the previously-found
# exp()-on-a-signed-rank-score bug already has its fix live in selection_core.py/
# test_utility_inmem.py (model_type == "lgbm_rank" skips exp() and forces --greedy-key utility)
# -- no code changes needed here, this script only trains models.
#
# CPU-only (LightGBM, no CUDA) -- sequential (not concurrent) per CLAUDE.md's shared-CPU rule,
# each fit_single_scene_ranker() call already uses n_jobs=-1 internally, same as the existing
# retrain_pooled_fixed_labels.sh script this is modeled on. nice deprioritizes so it yields to
# interactive/other users' work.
#
# Trained under gaussian_splatting env (NOT gsquic, unlike the regressor retrain script) --
# LGBMRanker's fitted state (early-stopping eval_set/eval_group -> evals_result_) pickles a
# numpy>=2.0-only scalar type; gaussian_splatting's numpy 1.21.6 can't unpickle it, and
# inference (test_utility_inmem.py) always runs under gaussian_splatting anyway. Confirmed via
# a control test that plain regressor pickles (no eval_set/early-stopping state) load fine
# cross-env -- this is specific to the ranker's early-stopping metadata, not a general issue.
set -euo pipefail

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility"
cd "$ROOT"

OUT_BASE="$ROOT/output/ml_models_experimental"

run() {
    local scene_set="$1" out="$2"
    echo ""
    echo "====== scene-set=${scene_set} -> ${out} ======"
    # ablation "ACG" is stale -- Group G (cos_cam_tile) was folded permanently into Group A
    # on 2026-07-05 (PLAN.md Open item 3); ml/features.py::ABLATION_NAMES = ("AC",) only now.
    nice -n 10 conda run -n gaussian_splatting python "$ROOT/ml/cross_scene/train_pooled_rank.py" \
        --scene-set "$scene_set" --ablation AC --seed 0 \
        --root "$ROOT/output/oracle/8" --out "$out"
}

run real3  "$OUT_BASE/pooled_rank_real3"
run synth7 "$OUT_BASE/pooled_rank_synth7"
run all10  "$OUT_BASE/pooled_rank_all10"

echo ""
echo "All rank-model retrains done."
