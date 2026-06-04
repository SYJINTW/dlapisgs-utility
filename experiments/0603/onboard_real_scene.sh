#!/usr/bin/env bash
# Onboard a new single-LOD real scene (Mip-NeRF 360 style) into the pipeline.
# Runs: gen_sparse_views -> tiling -> oracle ΔQ -> ML train -> Exp1 sweep.
#
# Usage:
#   ./experiments/0603/onboard_real_scene.sh <scene> <ply_path>
#
# Examples:
#   ./experiments/0603/onboard_real_scene.sh garden \
#       /mnt/data1/samk/gs-quic/gaussian_splatting/garden/point_cloud/iteration_30000/point_cloud.ply
#   ./experiments/0603/onboard_real_scene.sh stump \
#       /mnt/data1/samk/gs-quic/gaussian_splatting/stump/point_cloud/iteration_30000/point_cloud.ply
#
# Prerequisites: PLY already audited and present (done 2026-06-03).
# DO NOT run yet — see PLAN.md "Add 2 real scenes" entry. Run when GPU is free.

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

SCENE="${1:?usage: ./experiments/0603/onboard_real_scene.sh <scene> <ply_path>}"
PLY="${2:?usage: ./experiments/0603/onboard_real_scene.sh <scene> <ply_path>}"
[ -f "$PLY" ] || { echo "PLY not found: $PLY"; exit 1; }

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
GS_DATA_ROOT="/mnt/data1/samk/gs-quic/gaussian_splatting/${SCENE}"
OUT_BASE="output/0603/onboard_${SCENE}"
TRACE_TRAIN="${GS_DATA_ROOT}/sparse_views_100_train.json"
TRACE_EVAL="${GS_DATA_ROOT}/sparse_views_100_eval.json"
TILING_CACHE="${OUT_BASE}/.tiling_cache.npz"
ORACLE_OUT="${OUT_BASE}/oracle_dq"
ORACLE_NPZ="${ORACLE_OUT}/oracle_dq.npz"
ML_OUT="ml/models_clean/${SCENE}/AC"

echo "=== Onboarding ${SCENE} ==="
echo "PLY: $PLY"
echo "Out: $OUT_BASE"
mkdir -p "$OUT_BASE"

# ---------------------------------------------------------------------------
# Step 1: Generate train + eval camera traces (100 views each)
# ---------------------------------------------------------------------------
echo "[1/5] gen_sparse_views (train seed=42, eval seed=1)"
conda run -n gsquic python experiments/gen_sparse_views.py \
  --ply "$PLY" --out "$TRACE_TRAIN" \
  --n-views 100 --scene-type mipnerf360 \
  --fov-deg 60 --width 1600 --height 1600 \
  --seed 42

conda run -n gsquic python experiments/gen_sparse_views.py \
  --ply "$PLY" --out "$TRACE_EVAL" \
  --n-views 100 --scene-type mipnerf360 \
  --fov-deg 60 --width 1600 --height 1600 \
  --seed 1

# ---------------------------------------------------------------------------
# Step 2: Compute tiling cache (needed by oracle ΔQ)
# ---------------------------------------------------------------------------
echo "[2/5] tiling cache (grid 8x8x8, camera 0, budget 100%)"
conda run -n gaussian_splatting python test_utility_inmem.py \
  --ply "$PLY" --gt-ply "$PLY" \
  --camera-trace "$TRACE_EVAL" \
  --grid-shape 8 8 8 --camera-index 0 --budget-pct 100 \
  --schemes vd_lod --packing-mode progressive \
  --weight-mode screen_area --w-norm sum --c-norm sum \
  --img-w 1600 --img-h 1600 --num-lod 1 \
  --tiling-cache "$TILING_CACHE" \
  --output-root "${OUT_BASE}/tiling_bootstrap" \
  --scene "$SCENE"

# ---------------------------------------------------------------------------
# Step 3: Oracle ΔQ labels (LOO) — gaussian_splatting env, all 100 eval cams
# ---------------------------------------------------------------------------
echo "[3/5] oracle ΔQ (exp4, LOO, 100 eval cameras)"
conda run -n gaussian_splatting python experiments/exp4_oracle_dq.py \
  --ply "$PLY" \
  --trace "$TRACE_EVAL" \
  --tiling-cache "$TILING_CACHE" \
  --output-root "$ORACLE_OUT" \
  --width 1600 --height 1600 \
  --sh-degree 3 \
  --flush-every 5

# ---------------------------------------------------------------------------
# Step 4: ML train (RF, AC features, msl=2, mf=0.3)
# ---------------------------------------------------------------------------
echo "[4/5] ML train (RF, AC, msl=2, mf=0.3)"
conda run -n gsquic python ml/train.py \
  --oracle-npz "$ORACLE_NPZ" \
  --output-dir "$ML_OUT" \
  --ablations AC \
  --models rf \
  --rf-min-samples-leaf 2 \
  --rf-max-features 0.3 \
  --seed 42

# ---------------------------------------------------------------------------
# Step 5: Exp1 weight-mode sweep (screen_area + volume + volume_over_d2 + random)
# ---------------------------------------------------------------------------
echo "[5/5] Exp1 weight-mode sweep (1600x1600, progressive, vd_lod)"
for WMODE in screen_area volume volume_over_d2; do
  echo "  weight_mode=$WMODE"
  conda run -n gaussian_splatting python test_utility_inmem.py \
    --ply "$PLY" --gt-ply "$PLY" \
    --camera-trace "$TRACE_EVAL" \
    --grid-shape 8 8 8 --camera-index -1 \
    --budget-pct 10 25 40 55 70 85 100 \
    --schemes vd_lod --packing-mode progressive \
    --weight-mode "$WMODE" --w-norm sum --c-norm sum \
    --img-w 1600 --img-h 1600 --num-lod 1 \
    --tiling-cache "$TILING_CACHE" \
    --output-root "output/0603/exp1_1600_vol_modes/${SCENE}/${WMODE}" \
    --scene "$SCENE"
done

# Random baseline
echo "  weight_mode=random (shuffle-visible-seed=42)"
conda run -n gaussian_splatting python test_utility_inmem.py \
  --ply "$PLY" --gt-ply "$PLY" \
  --camera-trace "$TRACE_EVAL" \
  --grid-shape 8 8 8 --camera-index -1 \
  --budget-pct 10 25 40 55 70 85 100 \
  --schemes vd_lod --packing-mode progressive \
  --weight-mode screen_area --w-norm sum --c-norm sum \
  --shuffle-visible-seed 42 \
  --img-w 1600 --img-h 1600 --num-lod 1 \
  --tiling-cache "$TILING_CACHE" \
  --output-root "output/0603/exp1_1600_vol_modes/${SCENE}/random" \
  --scene "$SCENE"

echo "=== ${SCENE} onboarding complete ==="
echo "Traces:  $TRACE_TRAIN  $TRACE_EVAL"
echo "Oracle:  $ORACLE_NPZ"
echo "ML:      $ML_OUT/rf.pkl"
echo "Exp1:    output/0603/exp1_1600_vol_modes/${SCENE}/"
