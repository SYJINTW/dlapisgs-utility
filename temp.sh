


  # GPU 2 — Exp 1 (GS weight sweep, progressive packing, vd_lod_w_c, 3 weight modes × 8 scenes × 8 budgets)
  cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility && \
    CUDA_VISIBLE_DEVICES=2 bash experiments/0514/run_exp1_gs_weight.sh 2>&1 | tee output/0514/exp1_gs_weight/run.log

  # GPU 3 — Exp 2 (Tile utility ablation, tile_strict packing, 4 schemes × 2 weight modes × 8 scenes × 8 budgets)
  cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility && \
    CUDA_VISIBLE_DEVICES=3 bash experiments/0514/run_exp2_tile_utility.sh 2>&1 | tee output/0514/exp2_tile_utility/run.log

# Once both are running, post-run verification one-liner (any cell where 100% budget doesn't hit 100/100 inf prints FAIL):

cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility && python3 - <<'PY'
import json, glob, os
root = "output/0514"
fail = 0
for summary in sorted(glob.glob(f"{root}/exp*/*/*/metrics/summary.csv")):
    cell_root = os.path.dirname(os.path.dirname(summary))
    last_budget_dir = max(glob.glob(f"{cell_root}/metrics/budget_*"),
        key=lambda p: float(os.path.basename(p).removeprefix("budget_").removesuffix("mb").replace("p", ".")))
    for scheme_dir in sorted(glob.glob(f"{last_budget_dir}/*")):
        jsons = glob.glob(f"{scheme_dir}/camera_*.json")
        n_inf = sum(1 for f in jsons if json.load(open(f))["psnr"] == float("inf"))
        if n_inf != len(jsons):
            print(f"FAIL {os.path.relpath(cell_root, root)}/{os.path.basename(scheme_dir)}: {n_inf}/{len(jsons)}")
            fail += 1
print(f"\n{'ALL PASS' if fail == 0 else f'{fail} cells failed'}")
PY


