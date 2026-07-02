#!/usr/bin/env bash
# One-shot debug inspect: 1 scene, 1 camera, 1 budget -> select + render + PSNR/SSIM/MSE.
# Keeps the rendered PNG (LMG-style spot-check). Output under output/debug/.
#
# Usage:
#   ./debug.sh <scene> [cam=0] [budget_pct=40] [grid=8] [packing=tile_strict] [scheme=vd_lod] [gs_order=weight]
# Examples:
#   ./debug.sh chair                      # cam 0, 40%, 8³, tile_strict, vd_lod
#   ./debug.sh chair 7 70 8 progressive   # progressive packing
#   ./debug.sh chair 7 40 1 progressive   # no-cull (1³)
#   ./debug.sh ship 12 25 8 tile_strict ml
#   ./debug.sh chair 0 40 8 tile_partial vd_lod ply   # PLY-order baseline
#
# ml/oracle_loo/oracle_loo_ssim schemes auto-wire their model-dir / oracle-npz if present.

set -euo pipefail
cd "$(dirname "$0")"

SCENE="${1:?usage: ./debug.sh <scene> [cam] [budget_pct] [grid] [packing] [scheme] [gs_order]}"
CAM="${2:-0}"; BUD="${3:-40}"; GRID="${4:-8}"; PACK="${5:-tile_strict}"; SCHEME="${6:-vd_lod}"; GS_ORDER="${7:-weight}"
ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
OUT="output/debug/${SCENE}_cam${CAM}_b${BUD}_g${GRID}_${PACK}_${SCHEME}"

case "$SCENE" in
  bicycle) PLY="${ROOT}/exp-dataset/bicycle/point_cloud.ply"
           TRACE="${ROOT}/exp-dataset/bicycle/sparse_views_eval.json" ;;
  *)       PLY="${ROOT}/exp-dataset/${SCENE}/checkpoint/point_cloud/iteration_30000/point_cloud.ply"
           TRACE="${ROOT}/exp-dataset/${SCENE}/sparse_views_eval.json" ;;
esac
[ -f "$PLY" ]   || { echo "no PLY: $PLY"; exit 1; }
[ -f "$TRACE" ] || { echo "no trace: $TRACE"; exit 1; }

EXTRA="--gs-order $GS_ORDER"
MODEL_DIR="output/ml_models/8/${SCENE}/AC"
ORACLE_NPZ="output/oracle/8/eval/${SCENE}/oracle_dq.npz"
[ "$SCHEME" = "ml" ]                                          && [ -d "$MODEL_DIR" ] && EXTRA="$EXTRA --ml-model-dir $MODEL_DIR --ml-model-type lgbm"
{ [ "$SCHEME" = "oracle_loo" ] || [ "$SCHEME" = "oracle_loo_ssim" ]; } && [ -f "$ORACLE_NPZ" ] && EXTRA="$EXTRA --oracle-npz $ORACLE_NPZ"

rm -rf "$OUT"; mkdir -p "$OUT"
echo ">> select + render + metrics  $SCENE cam=$CAM budget=${BUD}% grid=${GRID}³ packing=$PACK scheme=$SCHEME"
conda run -n gaussian_splatting python test_utility_inmem.py \
  --ply "$PLY" --gt-ply "$PLY" --output-root "$OUT" --camera-trace "$TRACE" \
  --grid-shape "$GRID" "$GRID" "$GRID" --camera-index "$CAM" --budget-pct "$BUD" \
  --schemes "$SCHEME" --num-lod 1 --packing-mode "$PACK" \
  --weight-mode screen_area --w-norm sum --c-norm sum --img-w 1600 --img-h 1600 \
  --scene "$SCENE" --lpips \
  $EXTRA >/dev/null 2>&1

MJ=$(find "${OUT}/metrics" -name "camera_$(printf '%03d' "$CAM").json" | head -1)
PNG=$(find "${OUT}/renders" -name "camera_$(printf '%03d' "$CAM").png" | head -1)
python3 -c "
import json,math
d=json.load(open('$MJ'))
p=d['psnr']; mse=10**(-p/10) if p!=float('inf') else 0.0
print(f\"  scene=$SCENE cam=$CAM budget=${BUD}% packing=$PACK scheme=$SCHEME\")
print(f\"  n_gs={d['selected_gaussians']}  tiles={d.get('n_selected_tiles','-')}\")
lp = d.get('lpips')
lp_str = f\"   LPIPS={lp:.4f}\" if lp is not None else \"\"
print(f\"  PSNR={p:.3f} dB   SSIM={d['ssim']:.4f}   MSE={mse:.3e}{lp_str}\")
"
echo "  render: $PNG"
