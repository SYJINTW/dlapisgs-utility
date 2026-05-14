# 0514 Sweep — Runnable Commands

## Smoke summary

**Exp 1 (hotdog, volume_over_d2, cam 0, 3.5 MB)** — PSNR 23.67 @ 0.108s render + 0.194s compute = 302 ms stream-latency.
**Exp 2 (bicycle, volume_over_d2, cam 0, 141.9 MB, schemes={vd_lod, vd_lod_w_c})** — render 0.64–0.67s + compute 0.38s = ~1.05s stream-latency per request. Tiling at 8×8×8 takes ~160 s (one-time, cached across weight-mode runs via `--tiling-cache`).

Pipeline verified end-to-end: selection → GT/subset render → metrics → `timings_oneshot.csv` → quick PNG plots → 3×2 representative figures.

---

## Runnable commands (Phase D — full sweep)

Both can run in parallel on GPUs 2 and 3. Each is idempotent: `gt_renders/camera_NNN.png` is reused on rerun, and the per-scene `tiling.npz` is shared across weight-mode runs in Exp 1.

```bash
cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility

# Terminal 1 — Exp 1: GS-weight sweep (bicycle + hotdog + ship × 3 weight modes × 7 budgets × 100 cams)
CUDA_VISIBLE_DEVICES=2 bash experiments/0514/run_exp1_gs_weight.sh \
    2>&1 | tee output/0514_exp1.log

# Terminal 2 — Exp 2: tile-utility sweep (bicycle × 2 weight modes × 4 schemes × 7 budgets × 100 cams)
CUDA_VISIBLE_DEVICES=3 bash experiments/0514/run_exp2_tile_utility.sh \
    2>&1 | tee output/0514_exp2.log
```

### ETA estimates (from the smoke timings)

- **Exp 1**:
  - hotdog (small): ~0.5 s × 100 cams × 7 budgets × 3 weight modes ≈ **17 min**
  - ship  (mid):    ~1 s   × 100 × 7 × 3 ≈ **35 min**
  - bicycle:        ~9 s   × 100 × 7 × 3 ≈ **5.3 h**  + 3 × tiling setup ≈ **5.5 h**
  - **Exp 1 total ≈ 6.3 h** (sequential per scene; weight-modes share `tiling.npz` per scene)
- **Exp 2**:
  - bicycle, 2 weight modes × 4 schemes × 7 budgets × 100 cams × ~9 s ≈ **14 h** (+ tiling cached after run 1)

If wall time becomes a problem, drop `WEIGHT_MODES="volume_over_d2"` on Exp 2 to halve it.

### Useful overrides

```bash
# Resume / partial reruns:
SCENES="hotdog ship" bash experiments/0514/run_exp1_gs_weight.sh     # skip bicycle
WEIGHT_MODES="volume_over_d2" bash experiments/0514/run_exp2_tile_utility.sh

# Dry-run only (verify path matrix, no GPU):
DRY_RUN=1 bash experiments/0514/run_exp1_gs_weight.sh
DRY_RUN=1 bash experiments/0514/run_exp2_tile_utility.sh

# Skip render step (e.g. selection-only timing check):
SKIP_RENDER=1 bash experiments/0514/run_exp1_gs_weight.sh
```

### After the sweep finishes

Already invoked automatically inside the wrappers per cell:

- `experiments/render_metrics.py` → `<cell>/metrics/summary.csv` (+ `gt_renders/`, `renders/`).
- `experiments/aggregate_timings.py` → `<cell>/metrics/timings_{stages,setup,oneshot}.csv`.
- `experiments/concat_summaries.py` → `<sweep_root>/summary_all.csv` and per-scene `summary_all.csv`.
- `experiments/plot_metrics.py` → `<scene>/plots/{psnr,ssim}_vs_budget.png` (Exp 1) or `<cell>/plots/...` (Exp 2).
- `experiments/0514/pick_representative_views.py` → `<cell>/representative/<group>/budget_<MB>mb.png` (3×2 worst/median/best × subset/GT).

For paper figures, point `plotting/paper_plot_metrics.ipynb`'s `SUMMARY_CSV` at each sweep's CSV (with the right `GROUP_BY`) and re-execute via `jupyter nbconvert --to notebook --execute --inplace plotting/paper_plot_metrics.ipynb`.

---

## Side note on the new traces

The three already-generated `sparse_views_100.json` files don't carry the new `generation` metadata block (created before that change, and you asked not to touch them). Any future regenerations will include `generated_at`, hostname, full args, scene AABB, acceptance rate, FoV — all in a `generation` field that downstream readers ignore (only `camera_angle_x` + `frames` are consumed).

`.claude/context_experiments.md` update is still on the todo list — flag when you want it written and I'll align it to the 0514 state.
