# Research Plan

## Next session — analyze 0514 rerun, scope Exp 3 / Exp 4

Sweeps launched on GPUs 2 (Exp 1) and 3 (Exp 2; now hotdog + bicycle, smallest first). When they finish, the next session should:

### 0. If time permits

clear old backlogs. drop obsolete ones and do important and urgent ones.

### 1. Look at the results

- `output/0514/exp1_gs_weight/<scene>/plots/{psnr,ngs,ssim}_vs_budget.png` — grouped by `weight_mode`. Confirm bicycle reproduces or modifies the hotdog/ship finding (`screen_area` > `volume_over_d2` > `volume`). The `ngs_vs_budget.png` line should sit on or near the budget line for all three modes (progressive packing → ~100 % byte-utilization).
- `output/0514/exp2_tile_utility/<scene>/<weight_mode>/plots/{psnr,ngs,ssim}_vs_budget.png` — grouped by `scheme`. Watch for: (a) does `vd_lod_w_c` beat `vd_lod` (do W_k + C_k earn their keep at the tile level?); (b) does the ngs curve drop below budget for `tile_strict` (waste due to dropped overflowing tiles)?

### 2. Possibly refine the hypothesis

The pre-fix Exp 1 results suggested `screen_area` is the right per-Gaussian weight. If bicycle confirms this, lock it as default; if bicycle contradicts (real-world scenes have very different σ distributions vs synthetic), reopen the choice. Either way, the `det_gamma_over_d2` legacy mode should be retired from future sweeps.

### 3. Exp 3 — packing-mode ablation (scaffold next session)

Compare `progressive` vs `tile_partial` vs `tile_strict` head-to-head with all other factors fixed at the best Exp 1/2 setting. Headline plot: `ngs_vs_budget.png` grouped by `packing_mode` — `tile_strict` will visibly leave bytes on the table at small-mid budgets, the other two will saturate. PSNR is the corroborating metric.

```
fixed:  scheme=vd_lod_w_c, weight_mode=<winner of Exp 1>, w_norm=sum, c_norm=sum
sweep:  packing_mode ∈ {progressive, tile_partial, tile_strict} × 7 budgets × 100 cams
scenes: bicycle (primary), hotdog (cheap synthetic baseline)
ETA:    ≈ 5.3 h bicycle + 17 min hotdog = ≈ 5.5 h
```

### 4. Exp 4 — non-greedy selection (the knapsack question)

The user's analysis is right:

- **GS level** (`progressive`): uniform cost (`bytes_per_gaussian` per Gaussian) → 0-1 knapsack with uniform costs → greedy by w_gi is optimal. Nothing to test here.
- **Tile level** (`tile_partial`/`tile_strict`): heterogeneous cost (`#GS_in_tile × bytes_per_gaussian`) and heterogeneous value (U_k) → genuine 0-1 knapsack. Greedy by U_k is **not** optimal; greedy by `U_k / cost_k` (marginal utility per byte) is optimal for the LP relaxation and within 2× of integer optimal. Worth testing as `tile_partial_density` / `tile_strict_density` variants. Beyond that:
  - **DP**: exact 0-1 knapsack on the (~hundreds of tiles, ~thousands of buckets) instance is cheap (~ms per camera). Gives the true ceiling.
  - **LP relaxation**: gives an upper bound on achievable U; gap to DP tells you how slack the integer constraint is in practice.
  - **Random-restart baseline**: sanity floor.

#### The deeper concern: how do we know tile-utility is sensibly designed when its evaluation is entangled with the packer?

Yes, this conflation is real. Build a **ground-truth tile value** by ablation rendering, then both axes can be decoupled.

##### Oracle ΔQ — empirical tile value

For each (camera c, tile k), define

```
ΔQ_k(c) = Q( render(full_scene, c) ) - Q( render(full_scene \ tile_k, c) )
```

where Q is some quality metric against the full-scene render as reference (PSNR or 1/MSE work; MSE is the cleaner choice because it composes additively — PSNR is `−10·log₁₀(MSE)` so MSE-deltas avoid log nonlinearity). Bigger ΔQ_k ⇒ tile k matters more from that camera. No formula, no heuristic — purely a measurement of what removing the tile costs.

This gives a ground-truth ranking of tiles per camera. Every proposed U_k formula can then be scored by

```
Spearman ρ ( U_k(c) , ΔQ_k(c) )    averaged over cameras
```

— a single number that says "does this utility formula correctly rank tiles by how much they actually contribute to the image?" Completely independent of which packer you use downstream.

##### Decoupling experiments

(a) **Hold packing constant, sweep utility**: fix the packer to DP-optimal, sweep U_k formulas (including ΔQ_k itself as the ceiling). Measures: among proposed formulas, which lands closest to oracle? Gap to oracle = formula error.

(b) **Hold utility constant, sweep packing**: plug ΔQ_k into the packer as the value vector. Compare DP / marginal-utility-greedy / `tile_partial` / `tile_strict` / `progressive`. Measures: given perfect values, how much PSNR does the greedy packer lose vs. DP? Gap = packer error.

If formula error ≫ packer error → invest in U_k design. If packer error ≫ formula error → invest in non-greedy selection (Exp 4 proper).

##### Caveats

- **Leave-one-out misses interactions.** Tile A and tile B individually might each have small ΔQ but together cover a big region. For LOD=1 with our tile sizes this is probably second-order, but the principled fix is Shapley-style sampling (drop random k-subsets, regress) — expensive.
- **Compute.** One extra render per tile per camera. Bicycle at 8×8×8 ≈ 512 tiles, ~1 s/render → ~8.5 min per camera. A 10-camera subset is ~85 min — affordable as a first cut. Full 100 cameras ≈ 14 h, run once and cache to `<scene>/oracle_dq.npz` keyed by (camera, tile).

#### LOD note

Greedy marginal-utility = LP-knapsack optimal only when items are independent. With LOD layers ≥ 2, picking layer ℓ of tile k makes layers < ℓ partially redundant (or required-precursor, depending on encoding). That introduces precedence constraints → it's no longer a vanilla knapsack but a more constrained selection problem (closer to a "tree-knapsack" with precedence). For LOD=1 there are no precedence constraints; that's the right setting to validate the knapsack hypothesis cleanly before re-enabling LOD.

### 5. Packet overhead in the cost model — defer

Your back-of-envelope is right: MTU ≈ 1500 B, IP+UDP+QUIC overhead ≈ 50 B, payload ≈ 1450 B. GS = 248 B as written (236 B if nx/ny/nz are stripped — they're unused by the 3DGS rasterizer and are a free ~5 % bandwidth win; file this as a backlog item). 1450 / 248 = 5.85 → 5 GS/packet at ~83 % packet utilization, OR 6 GS spanning two packets. Pick one convention and stick with it.

For now, selection-side cost stays `N × bytes_per_gaussian` and the packet overhead is tracked separately at the streaming layer (factor ≈ 1.03–1.20 depending on packing convention). Building it into the knapsack now would couple two independent design questions; defer to the QUIC streaming experiments.

### 6. Known limitation — tile independence assumption

Every tile-level scoring and packing formulation we use (U_k, marginal-utility greedy, LOO ΔQ as oracle) **assumes tiles are independent contributors to the rendered image**. This is the implicit model:

```
Q(selected_set S) ≈ Σ_{k ∈ S} contribution(k)
```

It is empirically false in at least two regimes:

- **Overlapping-footprint redundancy** — adjacent tiles whose Gaussian splats cover the same screen-space pixels (e.g. two tiles meeting at a depth boundary). Dropping one alone changes the render very little because the other covers; dropping both is catastrophic. Naive marginal value (ours) and LOO ΔQ both _underestimate_ these tiles' importance; a smart packer that picks exactly one of `{A, B}` looks lucky rather than correct.
- **Occlusion stacking** — a near-camera tile fully occludes farther tiles. The occluded tiles have near-zero ΔQ when present alongside the near tile, but become essential once the near tile is dropped. Independent scoring can't represent this conditional dependency.

**Implications:**

- Pointwise rank-correlation ρ(U_k, ΔQ_k_LOO) may understate the quality of U_k when redundancy is present. Set-level metrics (top-K rank correlation, or PSNR-of-greedy-with-U directly) are more honest evaluators.
- LOO ΔQ is the _cheap_ oracle. The honest oracle for interaction-aware evaluation is Shapley ΔQ (cost: ~50–100× LOO).
- The tile-as-independent-unit assumption is what makes the problem tractable (0-1 knapsack solves in polynomial time). Removing it pushes us into submodular/supermodular optimization, where the right structural assumption (diminishing-returns vs. complementarity) determines tractability.

**Future work directions** (in increasing cost / decreasing likelihood of getting to it):

1. **LOO ΔQ oracle** (cheap, first step if Exp 4 even happens). Per-tile leave-one-out ablation render. ~1.4 h pilot / ~14 h full on bicycle. Captures occlusion correctly; blind to redundancy.

2. **AOI ΔQ as cheap complement to LOO** (same compute as LOO, possibly faster per render because rendering one tile uses ~5k–15k GS vs. N−5k for LOO). `AOI_k = Q({k}) − Q(∅)`: how much does k contribute when *nothing else* is present.
   - LOO is the marginal at the near-saturation end of the RD curve; AOI is the marginal at the near-empty end.
   - LOO captures occlusion; AOI captures redundancy. Neither alone captures both.
   - The per-tile pair (AOI_k, LOO_k) classifies tiles by interaction regime:
     - both high → uniquely important.
     - both low → genuinely unimportant.
     - AOI high, LOO low → participates in redundancy (packer needs one of this group).
     - AOI low, LOO high → occlusion-tier role (filler alone, but removal lets occluded content leak through).
   - The gap `AOI − LOO` is a cheap proxy for "how much redundancy this tile participates in" without paying for Shapley.
   - This is ~3 steps into the future — wait for LOO results first, then decide whether to run AOI on the same pilot cameras for a paired comparison.

3. **Shapley ΔQ** (expensive, only if LOO+AOI disagree materially with U_k rankings). Truncated permutation sampling on a 10-camera pilot ≈ 28 h overnight. Plot Shapley vs. LOO per tile — if scatter is near-diagonal, independence is approximately satisfied; if wide, redundancy / occlusion are first-order.

4. **Submodular formulations** — if ΔQ is submodular in S (diminishing returns), greedy by marginal gain has a 1−1/e ≈ 63 % approximation guarantee for budget-constrained maximization (Sviridenko / Nemhauser). U_k becomes a one-shot proxy for the first-step marginal.

5. **Group-based scoring** — score connected tile groups (e.g. depth-clustered or BFS-on-adjacency) rather than individual tiles. Reduces the unit of redundancy.

6. **Coverage-aware utility** — augment U_k with a penalty proportional to overlap with already-selected tiles' screen footprints. Effectively a determinantal-point-process-style diversification term.

**Action for the immediate paper:**

- State the tile-independence assumption explicitly in the methods section.
- Report top-K rank correlation in addition to full-list ρ; the gap is informative.
- Run the Shapley pilot as a sanity check, present the LOO-vs-Shapley scatter as evidence for or against the assumption holding in our setting.
- If LOO is approximately right (scatter near-diagonal), our results stand. If LOO is materially off, frame interaction-aware utility as the obvious next step.

### 7. Backlog from these analyses

- [ ] Strip nx/ny/nz from emitted PLYs — ~5 % free bandwidth win, no quality impact.
- [ ] Retire `det_gamma_over_d2` weight mode from sweeps and CLI default if bicycle confirms `screen_area` is the winner.
- [ ] If Exp 3 shows `tile_strict` materially under-utilizes the budget but loses negligible PSNR vs `tile_partial`, that's an argument for `tile_partial` as default; if it costs PSNR too, that's an argument for fixing the packer (Exp 4 territory).
- [ ] If Exp 4's oracle-ΔPSNR experiment shows DP ≪ greedy gap is small → packer is fine, double down on utility formula design.
- [ ] **Expand real-world scene coverage.** Today bicycle is our only real-world scene (MipNeRF-360). Fetch siblings from the same dataset (bonsai, counter, garden, kitchen, room, stump, treehill, flowers) so we have ≥3 indoor + ≥3 outdoor real-world scenes to test against the 7 NeRF-Synthetic scenes. Also pull Tanks & Temples (Truck, Train, Barn, Caterpillar, Family) for a third dataset family — different capture conditions expose whether `screen_area` and our packers generalize across distributions. Pipeline: train with lapis-gs (existing), then `experiments/gen_sparse_views.py --scene-type mipnerf360` (or `--scene-type tnt` if it exists; otherwise extend gen_sparse_views.py to handle T&T's `images/` + `pose_bounds.npy` format).

---

## Status: 2026-05-18 — full-set ≠ identity: tile-strict / tile-partial packers reorder gaussians

Counterpart to the 0515 progressive packer fix. While debugging a "missing last point" in `output/0514/exp2_tile_utility/hotdog/screen_area/plots/psnr_vs_budget.png`, found that `tile_strict` and `tile_partial` packing modes still fail to render bit-identically to GT even when the full scene is selected.

**Diagnosis on hotdog @ 100% budget (35.19 MiB, all four schemes select N=148,783 = full scene):**

| scheme       | inf views / 100 |
|--------------|-----------------|
| vd_lod       | 28              |
| vd_lod_w     | 23              |
| vd_lod_c     | 22              |
| vd_lod_w_c   | 21              |

Same gaussian *set*, different inf rate — the only differentiator is the output PLY's gaussian *order*. `_greedy_order_tile_strict` writes tiles in utility rank, sorting within each tile by `w_gi` (test_utility.py:182–206); `_greedy_order` (tile_partial) does the same. The downstream `diff_gaussian_rasterization_lapisgs` is order-sensitive due to depth-sort tie-breaking + non-associative alpha compositing.

`_greedy_order_progressive` was patched on 2026-05-15 to guarantee identity at saturation (two-pass visible/invisible sort). The tile-level packers were missed.

**Companion plot bug:** `experiments/plot_metrics.py` averaged `inf` PSNR values directly, so `np.mean([…, inf, …]) = inf`, and matplotlib drops `inf` y-values from line plots. The 60 dB saturation guide was decorative. One inf view on `vd_lod` at 29.9 MiB hid the entire data point.

**Fix landed:** plotter clamps per-camera PSNR to 60 dB before averaging; `test_utility.py:_select_at_budget` sorts the selected index array ascending before PLY write. **The sort is a reproducibility policy, not a transport decision** — the rasterizer is order-sensitive, so without it, cross-scheme PSNR differences mix "set chosen" with "ordering noise." Selection rank-order is preserved inside the `_greedy_order_*` functions for any downstream consumer (priority streaming, logging) that genuinely needs it; only the final PLY-write index list is sorted.

**Counter-intuitive finding to investigate (hotdog @ 85% budget, all schemes selected exactly 126,465 gaussians but different subsets):** median PSNR — vd_lod **62.79 dB**, vd_lod_w 43.45, vd_lod_c 24.27, vd_lod_w_c 32.99. Baseline beats all W/C-augmented schemes despite identical count.

Hypothesis (unverified, from code inspection of `utility_calculation.py:188–214` and `_compute_base_scores`): with `w_norm=sum`/`c_norm=sum`, `W_k` and `C_k` are probability distributions summing to 1. Multiplying them into `base_scores = (v_k/d_k)` deflates visible-tile scores into the range of invisible-tile scores (`INVISIBLE_PRIORITY_EPS=1e-2 / d_k`). `vd_lod_w_c` lets low-W·C visible tiles drop below high-W·C invisible tiles → tile_strict fills budget with invisible content → holes in visible region. `tile_strict` lacks utility-per-byte (knapsack-ratio) normalization (see §4 Exp 4 above), compounding the problem.

**Cheap diagnostic to confirm:** for one camera at 85% budget, dump `(tile_idx, visible, W_k, C_k, score, rank)` for both `vd_lod` and `vd_lod_w_c`. Count inversions: how many invisible tiles outrank visible tiles in `vd_lod_w_c` vs `vd_lod` (expect ~0 in baseline, >0 in vd_lod_w_c). Tooling: `experiments/debug_viewer_app.py` already plots per-tile distributions; extend or repurpose for the inversion count.

If hypothesis holds, the principled fix is Exp 4 (oracle ΔQ + knapsack-ratio packer) already scoped above. Short-term workaround: try `w_norm=max` and `c_norm=max` instead of `sum` to preserve the relative magnitude of `(v/d)` vs `W·C`.

**0514 sweep expanded for this rerun:** added all 5 missing NeRF-Synthetic scenes (chair, drums, ficus, materials, mic) on top of hotdog + ship + bicycle. `experiments/0514/run_exp{1,2}_*.sh` now default to 8 scenes; new `sparse_views_100.json` traces generated next to each new PLY via `experiments/gen_sparse_views.py --scene-type synthetic`.

---

## Status: 2026-05-15 — 0514 sweep restarted on a clean slate after three correctness fixes

The first 0514 run on 2026-05-14 died mid-sweep at cam 31 (disk full) with only hotdog + ship (Exp 1) and a partial bicycle/volume completed. While investigating, three issues surfaced — all now patched. `output/0514/` was wiped to ~3 TB free; the rerun uses the patched code.

**0515 patches (all landed):**

1. **Two-pass `progressive` packer.** Old `_greedy_order_progressive` hard-masked invisible-tile Gaussians, capping selection at the visible-tile pool. So "100 % byte budget" never reached identity — we observed a 70 dB ceiling on hotdog. Fix is a two-tier sort: visible-tile GS sorted by w_gi, then (if budget remains) invisible-tile GS sorted by w_gi. Guarantees identity at byte_budget ≥ scene_size; mid-budget behavior unchanged (invisible never beats visible). Multiplicative ε softening was ruled out — `w(g_i)` spans ~30 orders of magnitude on bicycle (see `output/0513_histogram_bicycle/*.png`), no ε partitions cleanly. (`test_utility.py:_greedy_order_progressive`)
2. **`--budget-pct` flag.** Replaces hand-coded per-scene MB tables in both wrappers. Inside `test_utility.py`, percentages resolve against `N * bytes_per_gaussian` (exact full-scene byte size, not the rounded one-decimal MB number we previously baked into the wrappers). So `--budget-pct 100` strictly satisfies `max_count = N` and saturation is exact. Wrappers no longer carry `scene_full_mb()` tables.
3. **`render_metrics.py --delete-ply`.** Unlinks each `selected.ply` (and its sibling manifest) immediately after the metrics row is written. Default-on in both 0514 wrappers (override with `KEEP_PLY=1`). Without this, bicycle alone hit 511 GB per (scene, weight_mode) cell. With it, peak PLY disk is at most one camera × #budgets in flight (a few GB).

**Other plotter/methodology cleanups:**

- `experiments/plot_metrics.py` now plots **95 % CI on the mean** (`±1.96·σ/√n`, ddof=1), not `±std`. With n=100 cameras, CI bars are ~20× tighter than the old std bars; method-vs-method differences become visually unambiguous.
- PSNR plots gain a dotted **60 dB saturation guide** (MSE < 10⁻⁶ ⇒ visually identical even before 8-bit PNG quantization at ≈48 dB). Use this, not SSIM, as the saturation criterion. SSIM hits 1.0000 well before identity.
- Discovered: tile AABBs in GGSP are pure grid-cell boxes on Gaussian _centers_ — no footprint inflation. So a Gaussian whose center is in a culled tile but whose splat extends into view will leak. This is why the old `progressive` packer's hard mask broke identity. The two-pass fix routes around it at the candidate-pool level; we explicitly chose not to inflate AABBs (would defeat directional culling — see chat 2026-05-15 for the rationale).

**Pre-fix Exp 1 results (hotdog + ship, kept as a reference)** — `screen_area` dominated every non-saturated budget on both scenes by 1.6–5.4 PSNR over `volume_over_d2` and 3.6–7.5 over `volume`. The ranking is expected to be preserved post-fix (the second pass only fires near saturation; mid-budget signal is unaffected). Bicycle was not in the dataset and is the most informative scene — that's the headline value of the rerun.

## Status: 2026-05-14 — 0514 sweep scaffolded (Exp 1 GS-weight × 3 scenes; Exp 2 tile-utility × 2 weight modes)

New artifacts (implemented this session):

- `experiments/gen_sparse_views.py` — generates 100-view Blender-format trace alongside each PLY, with frustum-based void rejection (min subsampled-Gaussian visibility threshold).
- `experiments/0514/run_exp1_gs_weight.sh` — Exp 1: progressive packing × {volume, volume_over_d2, screen_area} on bicycle/hotdog/ship at 10/25/40/55/70/85/100 % budget tiers (default `CUDA_VISIBLE_DEVICES=2`).
- `experiments/0514/run_exp2_tile_utility.sh` — Exp 2: bicycle only, `tile_strict` packing, two sub-sweeps (weight-mode ∈ {volume_over_d2, screen_area}) × 4 schemes × 7 budgets (default `CUDA_VISIBLE_DEVICES=3`).
- `experiments/0514/pick_representative_views.py` — symlinks worst / median / best PSNR cameras per cell (+ GT counterpart) into `<output_root>/representative/`.
- `experiments/aggregate_timings.py` — turns the per-stage `timings.json`/`render_timings.json` into a readable `timings_summary.csv` with mean / std / 95 % CI / p50 / p95 / total and a `_per_camera_e2e` row.
- `experiments/render_metrics.py` edits: emits a persistent `<output_root>/gt_renders/camera_NNN.png` (idempotent, reused on rerun) and adds a `scene` column to `summary.csv`.

Run order:

```bash
# Phase B — generate per-scene traces (CPU only, ~minutes)
python experiments/gen_sparse_views.py --ply exp-dataset/bicycle/point_cloud.ply \
    --scene-type mipnerf360 --n-views 100 \
    --out exp-dataset/bicycle/sparse_views_100.json
python experiments/gen_sparse_views.py --ply exp-dataset/hotdog/checkpoint/point_cloud/iteration_30000/point_cloud.ply \
    --scene-type synthetic --n-views 100 \
    --out exp-dataset/hotdog/checkpoint/point_cloud/iteration_30000/sparse_views_100.json
python experiments/gen_sparse_views.py --ply exp-dataset/ship/checkpoint/point_cloud/iteration_30000/point_cloud.ply \
    --scene-type synthetic --n-views 100 \
    --out exp-dataset/ship/checkpoint/point_cloud/iteration_30000/sparse_views_100.json

# Phase B' — dry-run to verify dir layout
DRY_RUN=1 bash experiments/0514/run_exp1_gs_weight.sh
DRY_RUN=1 bash experiments/0514/run_exp2_tile_utility.sh

# Phase C — smoke (one camera, smallest budget)
CAMERA_INDEX=0 BUDGET_PCTS=10 bash experiments/0514/run_exp1_gs_weight.sh
CAMERA_INDEX=0 BUDGET_PCTS=10 bash experiments/0514/run_exp2_tile_utility.sh

# Phase D — full sweep, parallel
CUDA_VISIBLE_DEVICES=2 bash experiments/0514/run_exp1_gs_weight.sh
CUDA_VISIBLE_DEVICES=3 bash experiments/0514/run_exp2_tile_utility.sh
```

---

## Status: 2026-05-13 — instrumentation + Setup 1/2 plumbing landed; DX pass (tqdm, vectorized tile weights)

The 0513 push wired up per-stage timing, machine-readable run metadata, a Plotly Dash debug viewer, three packing modes, four weight modes, and W/C normalization knobs. Smoke-tested across all combinations. Ready for the bigger sweeps and the math-design iteration that prompted this work.

---

## Completed

### Deliverable 1 — Per-stage timing instrumentation

- [x] `_timed(name, store, **labels)` context manager in `test_utility.py` and `experiments/render_metrics.py`
- [x] Wrapped new spans: `compute_tile_weights_and_counts`, `_select_at_budget` (per scheme/budget), per-PLY-write summary stats (mean/max/total)
- [x] Dumps `<output_root>/timings.json` and `<output_root>/render_timings.json`

### Deliverable 2 — `params.yaml` run-metadata dump

- [x] `_dump_run_params` helper in both runners
- [x] Captures CLI args + timestamp + hostname + device + cuda + torch + python versions
- [x] `_yaml_safe` recursive coercion (handles torch's str subclasses and Paths)
- [x] Written as `<output_root>/params.yaml` and `<output_root>/render_params.yaml`

### Deliverable 3 — Plotly Dash interactive debug viewer

- [x] New script `experiments/debug_viewer_app.py`
- [x] Widgets: camera-index slider, weight_mode dropdown, w_norm / c_norm dropdowns, GS-subsample slider, overlay checklist
- [x] Panels: tile utility, per-tile #GS, per-tile W_k, per-Gaussian density (subsample), tile visibility, status text
- [x] Reuses `_project_points` and the shared normalize/weight helpers from `utility_calculation.py`

### Deliverable 4a — Setup 1: W/C normalization sweep

- [x] `normalize_term(x, mode)` helper supporting `{none, max, minmax, log1p, sum}`
- [x] CLI flags `--w-norm` (default `none`) and `--c-norm` (default `max`) — matches pre-0513 behavior when both are at default
- [x] `experiments/0513/run_setup1_norm_sweep.sh`
- [x] Manifest fields `w_norm`, `c_norm` added

### Deliverable 4b — Setup 2: progressive + tile_strict

- [x] `compute_gaussian_weights_v2(weight_mode=...)` with `{volume, volume_over_d2, screen_area}`
- [x] `project_covariance_2d` helper (EWA Jacobian, pure torch, GPU)
- [x] CLI flag `--packing-mode {tile_partial, tile_strict, progressive}` (default `tile_partial`)
- [x] CLI flag `--weight-mode {det_gamma_over_d2, volume, volume_over_d2, screen_area}` (default `det_gamma_over_d2`)
- [x] `_greedy_order_tile_strict` and `_greedy_order_progressive`
- [x] `experiments/0513/run_setup2_progressive.sh`
- [x] Manifest fields `packing_mode`, `weight_mode`, `gamma` added

### Smoke tests

- [x] `tile_partial` × `det_gamma_over_d2` (legacy-equivalent path) — produces same selected GS count as before
- [x] `tile_strict` — drops the overflowing tile (252 840 GS vs 253 687 for partial @ 60 MB)
- [x] `progressive` × `volume` — sorts globally, same byte count
- [x] `progressive` × `screen_area` — Jacobian path runs (~1.2 s for 6 M Gaussians on the bicycle scene)
- [x] `params.yaml` and `timings.json` (15 spans) verified on a minimal run

### Naming hygiene

- [x] Renamed `--weight-mode legacy` → `det_gamma_over_d2` (descriptive, not historical)
- [x] Confirmed `tile_partial` is **our proposed method**, not "legacy"

---

## How to use the new pieces

### Runner — `test_utility.py`

New flags (all backwards-compatible; defaults reproduce pre-0513 behavior):

```bash
conda run -n gsquic python test_utility.py \
    --ply <model.ply> --output-root <dir> --camera-trace <trace.json> \
    --grid-shape 8 8 8 --budgets-mb 60 100 200 --schemes vd_lod_w_c \
    --camera-index -1 \
    --w-norm    {none,max,minmax,log1p,sum}      # default none
    --c-norm    {none,max,minmax,log1p,sum}      # default max
    --packing-mode {tile_partial,tile_strict,progressive}   # default tile_partial
    --weight-mode  {det_gamma_over_d2,volume,volume_over_d2,screen_area}  # default det_gamma_over_d2
```

Outputs (under `<output_root>`):

- `params.yaml` — run config + execution context
- `timings.json` — per-stage wall times (one row per span; load into pandas)
- `tiling.npz`, `camera_viz/*.npz`, `ply/budget_*/<scheme>/camera_*.ply + .json`
- `utility.log` — same loguru lines as before

### Render + metrics — `experiments/render_metrics.py`

Same call signature as before. New outputs:

- `<output_root>/render_params.yaml`
- `<output_root>/render_timings.json`
- `metrics/summary.csv` now includes `w_norm`, `c_norm`, `packing_mode`, `weight_mode` columns

### Plotly Dash debug viewer — `experiments/debug_viewer_app.py`

On the server:

```bash
conda run -n gsquic python experiments/debug_viewer_app.py \
    --ply <model.ply> \
    --camera-trace <trace.json> \
    --grid-shape 4 4 4 \
    --port 8050
```

On your laptop:

```bash
ssh -L 8050:localhost:8050 <server>
# then open http://localhost:8050
```

Slide the camera index, toggle overlays, change weight_mode / w_norm / c_norm live — no re-run needed. PLY load + tiling happens once at startup.

### Sweep wrappers — `experiments/0513/`

```bash
# W/C normalization sweep (7 pairs by default)
bash experiments/0513/run_setup1_norm_sweep.sh
# defaults: bicycle, 8×8×8 grid, budgets 20/60/100/200/500 MB, all 50 cameras

# Packing-mode × weight-mode sweep (5 combos by default)
bash experiments/0513/run_setup2_progressive.sh
```

Override via env vars: `OUTPUT_ROOT`, `BUDGET_LIST`, `GRID_SHAPE`, `CAMERA_INDEX`, `PACKING_MODE`, etc.

---

## Next session

### Math design — the real reason this push exists

1. (DONE) **Run the Dash viewer first** on the bicycle scene, scrub camera index, and compare what `det_gamma_over_d2` vs `volume` vs `volume_over_d2` vs `screen_area` do to `W_k` distributions across tiles. The hypothesis (per the memory at `project_gamma_default_history.md`) is that γ=1 in the default mode weights by volume² and is the root cause of the "GS weight still sucks" symptom from 0510.
2. (RUNNING) **Setup 2 sweep:** run `run_setup2_progressive.sh`. Tells us whether progressive packing or any of the three new weight modes beats `tile_partial × det_gamma_over_d2`.
3. **Setup 1 sweep:** run `run_setup1_norm_sweep.sh`, aggregate metrics, plot PSNR vs budget for each (w_norm, c_norm) pair. The "none × max" cell is the pre-0513 baseline.

### Verification tasks not yet done

- [ x ] **Screen-area sanity check** — compare `screen_area` weights against the radii returned by a real render pass on one camera. Expect Spearman ρ > 0.9. If not, the Jacobian implementation in `project_covariance_2d` is wrong.
- [ ] **Baseline regression** — re-run 0508 budget sweep with `--w-norm none --c-norm max --packing-mode tile_partial --weight-mode det_gamma_over_d2 --gamma 1.0` and confirm PSNR/SSIM are bit-stable vs the pre-0513 run.

### Backlog (still deferred)

**DX**

- [x] add `tqdm` progress bars (cameras / schemes / budgets) with per-stage postfix so you can see which sub-stage is running
- [x] loguru routed through `tqdm.write` — no torn output
- [x] incremental `timings.json` flush once per camera (recoverable on crash; ~1 ms overhead, not per-span)
- [ ] **Resume/skip-existing** (`--resume` flag): check if `camera_{NNN}.ply` already exists for the first budget/scheme combo, skip that camera. Saves hours on reruns after a mid-sweep crash.
- [x] **`--dry-run`**: prints full (camera × scheme × budget → path) matrix, marks `[x]` for outputs that already exist, exits; no GPU work, no disk writes.
- [ ] **GPU memory logging**: append `torch.cuda.max_memory_allocated()` (MB) to each camera's timing rows; zero-cost, catches OOM risk before it strikes.
- [ ] **`_greedy_order_progressive` scheme-consistency**: it ignores `scheme` entirely and always sorts by `w_gi`. Running it under `vd` vs `vd_lod_w_c` produces identical PLYs — add a `logger.warning` or raise if `scheme != "vd"` when `packing_mode=progressive`.

**Perf / runtime**

- [x] Vectorize `compute_tile_weights_and_counts` — Python loop replaced by `scatter_add_` + `repeat_interleave` (single GPU kernel).
- [ ] Dedup tile_npz writes (test_utility_fast.py prototype exists)

**Upstream**

- [ ] `tiling_uniform_layered_gs` (GGSP/tiling.py) — triple-nested loop, 64 passes over 6M Gaussians, ~26 s on the smoke test. Fix with single-pass floor-division tile assignment + argsort grouping (~172s → seconds).
- [ ] `export_gs_to_ply` (GS-Interface/io_3dgs.py) — `list(map(tuple, ...))` allocates a Python tuple per Gaussian.

**Render + metrics**

- [ ] In-memory render pipeline: skip PLY writes entirely, render selected Gaussians directly in-memory → save PNG + metrics only. HDD is the bottleneck (~50 MB/s spinning disk).

**Research follow-ups**

1. Pick the winning (w_norm, c_norm, weight_mode, packing_mode) from the sweeps; lock as project default.
2. Algorithm design experiments (scheduler variants, LOD strategies once we re-enable num_lod > 1).
3. Multi-scene testing (beyond bicycle).
