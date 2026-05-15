# 0514 Sweep — Runnable Commands (rev 2026-05-15)

Original 0514 run died at cam 31/100 (disk full). After three patches (two-pass `progressive`, `--budget-pct`, `--delete-ply` default-on; details in `PLAN.md`), this rerun is on a clean `output/0514/`. Same scientific design as before.

---

## Runnable commands (Phase D — full sweep)

Both wrappers run in parallel on GPUs 2 and 3. Idempotent: `gt_renders/camera_NNN.png` is reused on rerun, per-scene `.tiling_cache.npz` is shared across weight-mode runs in Exp 1.

```bash
cd /mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility

# Terminal 1 — Exp 1: GS-weight sweep (bicycle + hotdog + ship × 3 weight modes × 7 budgets × 100 cams)
CUDA_VISIBLE_DEVICES=2 bash experiments/0514/run_exp1_gs_weight.sh \
    2>&1 | tee output/0514_exp1.log

# Terminal 2 — Exp 2: tile-utility sweep (bicycle × 2 weight modes × 4 schemes × 7 budgets × 100 cams)
CUDA_VISIBLE_DEVICES=3 bash experiments/0514/run_exp2_tile_utility.sh \
    2>&1 | tee output/0514_exp2.log
```

Both auto-trigger `render_metrics.py --delete-ply` per cell, so PLYs are removed as soon as their PSNR/SSIM row is written. The wrappers also concat per-scene/per-sweep summary CSVs and emit `plots/{psnr,ssim}_vs_budget.png` at the end.

### ETA (carried over from the previous smoke-timed estimates; algorithmic cost barely changed)

| Stage | Wall time |
|---|---|
| **Exp 1** total | **≈ 6.3 h** |
| ↳ hotdog (small) | ≈ 17 min |
| ↳ ship (mid) | ≈ 35 min |
| ↳ bicycle (large) + 3× tiling setup | ≈ 5.5 h |
| **Exp 2** total | **≈ 14 h** |
| ↳ bicycle × 2 weight modes × 4 schemes × 7 budgets × 100 cams | ≈ 14 h (tiling cached after run 1) |

Exp 2 can be halved by dropping one weight mode: `WEIGHT_MODES="volume_over_d2" bash …`.

### Peak disk estimate (with `--delete-ply` default-on)

- **Persistent renders** (per cell, accumulates over full sweep): ~600 MB (subset PNGs) + ~80 MB (GT PNGs, written once per scene).
  - Exp 1: 9 cells × 600 MB + 3 × 80 MB ≈ **5.6 GB**.
  - Exp 2: 2 cells × 4 schemes × 600 MB + 80 MB ≈ **5 GB**.
- **In-flight PLYs** (transient; deleted after each metric row): ≤ one camera × 7 budgets × scheme-count × max PLY size. Bicycle 100 % PLY ≈ 1.4 GB → up to ~10 GB transient if the thread pool falls behind, but typically drained within seconds.
- **Tiling caches**: ~45 MB (bicycle) + a few MB each (hotdog, ship).
- **Per-camera `camera_viz/NNN.npz`**: kilobytes each, ignorable.

**Total disk peak: ≈ 15–20 GB** for the two sweeps combined. Disk free at start: 3.0 TB. No risk.

### Useful overrides

```bash
# Resume / partial reruns:
SCENES="hotdog ship" bash experiments/0514/run_exp1_gs_weight.sh
WEIGHT_MODES="volume_over_d2" bash experiments/0514/run_exp2_tile_utility.sh

# Different budget grid:
BUDGET_PCTS="20 50 80 100" bash experiments/0514/run_exp1_gs_weight.sh

# Dry-run (verify path matrix; loads PLY once to resolve --budget-pct, no GPU):
DRY_RUN=1 bash experiments/0514/run_exp1_gs_weight.sh
DRY_RUN=1 bash experiments/0514/run_exp2_tile_utility.sh

# Selection-only timing check (no render):
SKIP_RENDER=1 bash experiments/0514/run_exp1_gs_weight.sh

# Keep PLYs (override --delete-ply; only if you have disk to burn):
KEEP_PLY=1 bash experiments/0514/run_exp1_gs_weight.sh
```

### Inspect a single (camera, budget) afterwards (under 5 s with warm tiling cache)

```bash
conda run -n gsquic python test_utility.py \
    --ply <full.ply> --output-root /tmp/inspect \
    --camera-trace <trace.json> --grid-shape 8 8 8 \
    --camera-index N --budget-pct B \
    --schemes vd_lod_w_c --packing-mode progressive \
    --weight-mode screen_area --w-norm sum --c-norm sum \
    --tiling-cache output/0514/exp1_gs_weight/<scene>/.tiling_cache.npz

conda run -n gaussian_splatting python experiments/render_metrics.py \
    --output-root /tmp/inspect --gt-ply <full.ply> --trace <trace.json> \
    --scene <name> --render-dir /tmp/inspect/renders
```

### After the sweep finishes

The wrappers already invoke per cell:

- `experiments/render_metrics.py` → `<cell>/metrics/summary.csv` (+ `gt_renders/`, `renders/`). `--delete-ply` removes PLYs after.
- `experiments/aggregate_timings.py` → `<cell>/metrics/timings_{stages,setup,oneshot}.csv`.
- `experiments/concat_summaries.py` → `<sweep_root>/summary_all.csv` and per-scene `summary_all.csv`.
- `experiments/plot_metrics.py` → `<scene>/plots/{psnr,ssim}_vs_budget.png` (95 % CI bars, 60 dB saturation line on PSNR).
- `experiments/0514/pick_representative_views.py` → `<cell>/representative/<group>/budget_<MB>mb.png` (3×2 worst/median/best × subset/GT). Note: with `--delete-ply` the source subset PLYs are gone, so representative figures use only the rendered PNGs (`renders/` + `gt_renders/`), not the PLYs.

For paper figures, point `plotting/paper_plot_metrics.ipynb`'s `SUMMARY_CSV` at each sweep's CSV (with the right `GROUP_BY`) and re-execute via `jupyter nbconvert --to notebook --execute --inplace plotting/paper_plot_metrics.ipynb`.

---

## Trace visualization

`experiments/visualize_trace_views.py` renders the 100 cameras as a 3D scatter + forward arrows, overlaid on tile AABBs (if `--tiling-npz` provided) or a single scene AABB (`--ply`). Already rendered: `plotting/traces/{bicycle,hotdog,ship}_views.png`.

```bash
python experiments/visualize_trace_views.py \
    --trace ../exp-dataset/bicycle/sparse_views_100.json \
    --ply   ../exp-dataset/bicycle/point_cloud.ply \
    --out   plotting/traces/bicycle_views.png
```

---

## Saturation criterion

A (view, budget) cell is **saturated** if `PSNR ≥ 60 dB`. SSIM is not a reliable identity check (it hits 1.0000 well before identity). The plotter draws a dotted 60 dB guide on PSNR figures.
