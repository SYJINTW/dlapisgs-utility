#!/usr/bin/env bash
# 2026-06-01: Exp 3 — packing-mode sweep. Loops {tile_partial,tile_strict,progressive}
# over the 0530 scheme set (ml vd_lod vd_lod_w oracle_loo), select + render + metrics.
# Modeled on experiments/0530/run_bicycle.sh. Packing mode is in the output path so modes
# never overwrite each other.
#
# Smoke (default): chair, 1 cam, budgets 10/55/100, GPU 0, fresh OUT_ROOT.
#   CUDA_VISIBLE_DEVICES=0 SCENES=chair CAMERA_INDEX=0 BUDGETS="10 55 100" \
#       OUT_ROOT=output/0601/exp3_smoke \
#       nohup bash experiments/0601/run_exp3_packing.sh > /tmp/exp3_smoke_0601.log 2>&1 &
#
# Full (after smoke + cost gate):
#   CUDA_VISIBLE_DEVICES=0 SCENES="chair drums ficus hotdog materials mic ship bicycle" \
#       CAMERA_INDEX=-1 BUDGETS="10 25 40 55 70 85 99 100" \
#       OUT_ROOT=output/0601/exp3_packing \
#       nohup bash experiments/0601/run_exp3_packing.sh > /tmp/exp3_full_0601.log 2>&1 &
#
# Prereqs (per scene):
#   ml/models_clean/{scene}/AC/rf.pkl
#   output/0527_clean/exp4_oracle_dq/eval/{scene}/oracle_dq.npz

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root = dlapisgs-utility/

ROOT="/mnt/data1/samk/gs-quic/cs5262_tile_quic"
ORACLE_ROOT="output/0527_clean/exp4_oracle_dq/eval"
MODEL_ROOT="ml/models_clean"
OUT_ROOT="${OUT_ROOT:-output/0601/exp3_smoke}"
BUDGETS="${BUDGETS:-10 55 100}"
CAMERA_INDEX="${CAMERA_INDEX:--1}"
MODEL_TYPE="${MODEL_TYPE:-rf}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# shellcheck disable=SC2206
SCENES=(${SCENES:-chair})
PACKING_MODES=(${PACKING_MODES:-tile_partial tile_strict progressive})

get_param() {
    local scene=$1 key=$2
    python3 -c "
import yaml, pathlib
p = pathlib.Path('${ORACLE_ROOT}/${scene}/params.yaml')
if not p.exists():
    exit(0)
with open(p) as f:
    d = yaml.safe_load(f)
print(d.get('args', {}).get('${key}', '') or '')
" 2>/dev/null
}

scene_ply() {
    case "$1" in
        bicycle) echo "${ROOT}/exp-dataset/bicycle/point_cloud.ply" ;;
        *)       echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/point_cloud.ply" ;;
    esac
}

scene_trace() {
    case "$1" in
        bicycle) echo "${ROOT}/exp-dataset/bicycle/sparse_views_100_eval.json" ;;
        *)       echo "${ROOT}/exp-dataset/$1/checkpoint/point_cloud/iteration_30000/sparse_views_100_eval.json" ;;
    esac
}

mkdir -p "$OUT_ROOT"

for PACKING in "${PACKING_MODES[@]}"; do
for SCENE in "${SCENES[@]}"; do
    ORACLE_NPZ="${ORACLE_ROOT}/${SCENE}/oracle_dq.npz"
    MODEL_DIR="${MODEL_ROOT}/${SCENE}/AC"
    PLY="$(scene_ply "$SCENE")"
    TRACE="$(scene_trace "$SCENE")"
    OUT_DIR="${OUT_ROOT}/${PACKING}/${SCENE}"

    if [ ! -f "$ORACLE_NPZ" ]; then
        echo "SKIP $SCENE — eval oracle not found: $ORACLE_NPZ"; continue
    fi
    if [ ! -d "$MODEL_DIR" ]; then
        echo "SKIP $SCENE — model dir not found: $MODEL_DIR"; continue
    fi
    if [ ! -f "$PLY" ]; then
        echo "SKIP $SCENE — PLY not found: $PLY"; continue
    fi
    if [ ! -f "$TRACE" ]; then
        echo "SKIP $SCENE — eval trace not found: $TRACE"; continue
    fi

    # Tiling cache: prefer oracle params.yaml; fall back to old exp1 cache. READ-ONLY reuse.
    TILING_CACHE="$(get_param "$SCENE" "tiling_cache")"
    if [ -z "$TILING_CACHE" ] || [ ! -f "$TILING_CACHE" ]; then
        TILING_CACHE="output/0514/exp1_gs_weight_done/${SCENE}/.tiling_cache.npz"
    fi
    TILING_ARGS=""
    [ -f "$TILING_CACHE" ] && TILING_ARGS="--tiling-cache $TILING_CACHE"

    mkdir -p "$OUT_DIR"   # parent must exist before tee opens the _sel.log

    # ── Selection ────────────────────────────────────────────────────────────
    echo ""
    echo "=== [$PACKING/$SCENE] selection ==="
    conda run -n gsquic python test_utility.py \
        --ply "$PLY" \
        --output-root "$OUT_DIR" \
        --camera-trace "$TRACE" \
        --grid-shape 8 8 8 \
        --budget-pct $BUDGETS \
        --schemes ml vd_lod vd_lod_w oracle_loo \
        --num-lod 1 \
        --camera-index "$CAMERA_INDEX" \
        --packing-mode "$PACKING" \
        --weight-mode screen_area \
        --w-norm sum --c-norm sum \
        --img-w 1600 --img-h 1600 \
        $TILING_ARGS \
        --oracle-npz "$ORACLE_NPZ" \
        --ml-model-dir "$MODEL_DIR" \
        --ml-model-type "$MODEL_TYPE" \
        2>&1 | tee "${OUT_DIR%/}_sel.log"

    echo "[$PACKING/$SCENE] selection done"

    # ── Render + metrics ─────────────────────────────────────────────────────
    echo "=== [$PACKING/$SCENE] render_metrics ==="
    conda run -n gaussian_splatting \
        python experiments/render_metrics.py \
        --output-root "$OUT_DIR" \
        --gt-ply "$PLY" \
        --trace "$TRACE" \
        --scene "$SCENE" \
        --delete-ply \
        2>&1 | tee "${OUT_DIR%/}_render.log"

    echo "[$PACKING/$SCENE] done — results: ${OUT_DIR}/metrics/summary.csv"
done
done

echo ""
echo "All packing×scene done. Output: $OUT_ROOT"
echo "Next: concat per-packing summaries, e.g."
echo "  conda run -n gsquic python experiments/concat_summaries.py \\"
echo "      --glob '${OUT_ROOT}/*/*/metrics/summary.csv' \\"
echo "      --output ${OUT_ROOT}/summary_all.csv"
