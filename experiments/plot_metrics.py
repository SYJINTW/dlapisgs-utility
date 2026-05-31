#!/usr/bin/env python3
"""Plot PSNR and SSIM vs budget from a render+metrics sweep.

Reads metrics/summary.csv produced by render_metrics.py and generates:
  - psnr_vs_budget.png
  - ssim_vs_budget.png
Each has one line per group (controlled by --group-by) with error bars (std across cameras).
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob_mod
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Stable label/order/style for the default scheme grouping.
_SCHEME_LABELS = {
    "vd_lod":          "VD+LOD (baseline)",
    "vd_lod_w":        "VD+LOD+W (ours)",
    "ml":              "ML (ours)",
    "vd_lod_c":        "VD+LOD+C",
    "vd_lod_w_c":      "VD+LOD+W+C (proposed)",
    "vd_lod_w_sa":     "VD+LOD+W (screen_area)",
    "vd_lod_w_vd2":    "VD+LOD+W (vol/d²)",
    "oracle_loo":      "Oracle-LOO (upper bound)",
    "oracle_aoi":      "Oracle-AOI",
    "oracle_combined": "Oracle-combined",
}
_SCHEME_ORDER = [
    # baseline → ours (vd_lod_w + ml) → … → oracle (upper bound) last
    "vd_lod", "vd_lod_w", "ml", "vd_lod_w_sa", "vd_lod_w_vd2", "vd_lod_c", "vd_lod_w_c",
    "oracle_aoi", "oracle_combined", "oracle_loo",
]

# For weight_mode grouping — worst-to-best order matches the hypothesis.
_WEIGHT_MODE_ORDER = ["det_gamma_over_d2", "volume", "volume_over_d2", "screen_area"]
_WEIGHT_MODE_LABELS = {
    "det_gamma_over_d2": "det_γ/d² (original, vol²/d²)",
    "volume":            "volume (vol, view-indep.)",
    "volume_over_d2":    "volume/d² (vol/d², view-dep.)",
    "screen_area":       "screen_area (projected footprint)",
}

MARKERS = ["s", "^", "D", "o", "v", "P"]
COLORS  = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#FF8C00", "#00CED1"]

# Clamp ceiling for per-camera PSNR. The rasterizer can hit MSE=0 → PSNR=inf
# (byte-identical to GT). One inf in `psnr_vals` poisons the mean and matplotlib
# silently drops inf y-values from line plots, so we clamp pre-mean. 60 dB
# matches the existing saturation guide drawn at hline=(60, "saturated (≥60 dB)").
PSNR_SATURATION_DB = 60.0


def _data_ylim(values: list[float], metric: str) -> tuple[float, float]:
    """Tight y-limits with a small pad below the min mean — keeps dynamic range.

    Trims the dead low end (PSNR floored to nearest 5 dB below data, SSIM to
    nearest 0.05) so bars/lines fill the panel instead of starting at 0. Ported
    from the 0514 summary plotter to match its layout.
    """
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return (0.0, PSNR_SATURATION_DB) if metric == "psnr" else (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    if metric == "psnr":
        floor = max(0.0, np.floor((lo - 3) / 5) * 5)
        ceil  = min(PSNR_SATURATION_DB * 1.05, max(PSNR_SATURATION_DB * 1.02, hi + 2))
        return floor, ceil
    floor = max(0.0, np.floor((lo - 0.03) * 20) / 20)  # nearest 0.05
    ceil  = min(1.005, hi + 0.02)
    return floor, ceil


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _apply_budget_labels(rows: list[dict], budget_pcts: list[float]) -> list[dict]:
    """Map per-scene budget_mb rank → percentage label (adds '_budget_key' field in-place).

    Different scenes have different budget_mb per pct level; we rank within each scene
    so that e.g. the 4th-smallest budget_mb of every scene all map to the same label.
    """
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scene[r.get("scene", "_")].append(r)
    for scene, srows in by_scene.items():
        distinct = sorted(set(float(r["budget_mb"]) for r in srows))
        if len(distinct) != len(budget_pcts):
            raise ValueError(
                f"scene={scene!r}: {len(distinct)} distinct budget_mb values but "
                f"--budget-pcts has {len(budget_pcts)} entries"
            )
        m = {mb: f"{int(p)}%" for mb, p in zip(distinct, budget_pcts)}
        for r in srows:
            r["_budget_key"] = m[float(r["budget_mb"])]
    return rows


def _aggregate(rows: list[dict], group_by: str) -> dict[str, dict[float, dict]]:
    buckets: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = r.get(group_by) or f"missing_{group_by}"
        bk: float | str = r["_budget_key"] if "_budget_key" in r else float(r["budget_mb"])
        buckets[key][bk].append(r)

    result: dict[str, dict] = {}
    for key, budgets in buckets.items():
        result[key] = {}
        for budget_mb, entries in budgets.items():
            psnr_vals = np.minimum(
                np.array([float(e["psnr"]) for e in entries]),
                PSNR_SATURATION_DB,
            )
            ssim_vals = np.array([float(e["ssim"]) for e in entries])
            ngs_vals  = np.array([float(e.get("selected_gaussians", 0)) for e in entries])
            n = len(entries)
            n_cameras = len(set(e.get("camera_index", "") for e in entries))
            # 95% CI on the mean: ±1.96·σ/√n. Tight when n is large; this is the
            # right bar for "is method A's mean above method B's mean?"
            # std would answer "how variable is a single camera?" — not what we want.
            ci = 1.96 / max(np.sqrt(n), 1.0)
            result[key][budget_mb] = {
                "psnr_mean": float(psnr_vals.mean()),
                "psnr_ci95": float(psnr_vals.std(ddof=1) * ci) if n > 1 else 0.0,
                "ssim_mean": float(ssim_vals.mean()),
                "ssim_ci95": float(ssim_vals.std(ddof=1) * ci) if n > 1 else 0.0,
                "ngs_mean":  float(ngs_vals.mean()),
                "ngs_ci95":  float(ngs_vals.std(ddof=1) * ci) if n > 1 else 0.0,
                "n":         n,
                "n_cameras": n_cameras,
            }
    return result


def _resolve_order_and_labels(agg: dict, group_by: str) -> tuple[list[str], dict[str, str]]:
    if group_by == "scheme":
        order = _SCHEME_ORDER
        labels = _SCHEME_LABELS
    elif group_by == "weight_mode":
        order = _WEIGHT_MODE_ORDER
        labels = _WEIGHT_MODE_LABELS
    else:
        order = sorted(agg.keys())
        labels = {}
    # include any keys not in the preset order (future-proofing)
    extras = [k for k in sorted(agg.keys()) if k not in order]
    return order + extras, labels


def _plot(agg: dict[str, dict[float, dict]], order: list[str], labels: dict[str, str],
          metric: str, ylabel: str, title: str, out_path: Path,
          hline: tuple[float, str] | None = None) -> None:
    """Generic group-keyed line plot with optional horizontal reference line."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    all_means: list[float] = []
    for i, key in enumerate(k for k in order if k in agg):
        marker = MARKERS[i % len(MARKERS)]
        color  = COLORS[i % len(COLORS)]
        pts = sorted(agg[key].items())
        x   = [p[0] for p in pts]
        y   = [p[1][f"{metric}_mean"] for p in pts]
        err = [p[1][f"{metric}_ci95"] for p in pts]
        all_means.extend(y)
        ax.errorbar(x, y, yerr=err, marker=marker, color=color,
                    linewidth=1.8, markersize=6, capsize=3,
                    label=labels.get(key, key))
    ax.set_ylim(*_data_ylim(all_means, metric))
    ax.set_xlabel("Budget (MiB)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25)
    xmax = max(
        (p[0] for key in order if key in agg for p in agg[key].items()),
        default=None,
    )
    if xmax is not None:
        ax.set_xlim(right=xmax * 1.05)
    if hline is not None:
        y_ref, hlabel = hline
        ax.axhline(y_ref, color="gray", linestyle=":", linewidth=1.2, zorder=0)
        ax.text(ax.get_xlim()[1], y_ref, f" {hlabel}",
                fontsize=9, color="gray", va="bottom", ha="right")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _load_overlay_rows(args) -> list[dict]:
    """Load, filter, and optionally rename overlay rows. Returns [] if no --overlay-csv."""
    if not getattr(args, "overlay_csv", None):
        return []
    rows = _read_csv(Path(args.overlay_csv))
    if getattr(args, "overlay_schemes", None):
        keep = set(args.overlay_schemes)
        rows = [r for r in rows if r.get("scheme") in keep]
    if getattr(args, "overlay_filter", None):
        for kv in args.overlay_filter:
            col, val = kv.split("=", 1)
            rows = [r for r in rows if r.get(col) == val]
    if getattr(args, "overlay_rename", None):
        for kv in args.overlay_rename:
            col, val = kv.split("=", 1)
            for r in rows:
                r[col] = val
    return rows


def _bar_plot(agg: dict, order: list[str], labels: dict[str, str],
              metric: str, ylabel: str, title: str, out_path: Path,
              hline: tuple[float, str] | None = None) -> None:
    """Grouped bar chart: x-axis = budget levels, bar color = group."""
    # Sort budgets: pct strings ("10%") numerically, plain floats numerically.
    def _bk_sort(b: object) -> float:
        if isinstance(b, str) and b.endswith("%"):
            return float(b[:-1])
        return float(b)

    all_budgets = sorted({b for key in agg for b in agg[key]}, key=_bk_sort)
    groups = [k for k in order if k in agg]
    n_budgets = len(all_budgets)
    n_groups = len(groups)
    bar_width = 0.8 / max(n_groups, 1)
    x = np.arange(n_budgets)

    fig, ax = plt.subplots(figsize=(max(11.5, n_budgets * 1.6), 5.5))
    all_means: list[float] = []
    for i, key in enumerate(groups):
        color = COLORS[i % len(COLORS)]
        y   = [agg[key].get(b, {}).get(f"{metric}_mean", 0.0) for b in all_budgets]
        err = [agg[key].get(b, {}).get(f"{metric}_ci95",  0.0) for b in all_budgets]
        all_means.extend(v for b, v in zip(all_budgets, y) if b in agg[key])
        offset = (i - n_groups / 2 + 0.5) * bar_width
        ax.bar(x + offset, y, bar_width * 0.92, yerr=err, capsize=2,
               color=color, label=labels.get(key, key), error_kw={"linewidth": 0.8})

    ax.set_ylim(*_data_ylim(all_means, metric))
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in all_budgets], fontsize=10)
    pct_labels = bool(all_budgets) and isinstance(all_budgets[0], str) and all_budgets[0].endswith("%")
    ax.set_xlabel("Budget (% of full scene size)" if pct_labels else "Budget (MiB)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    if hline is not None:
        y_ref, hlabel = hline
        ax.axhline(y_ref, color="gray", linestyle=":", linewidth=1.2, zorder=0)
        ax.text(n_budgets - 0.5, y_ref, f" {hlabel}",
                fontsize=9, color="gray", va="bottom", ha="right")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _bk_sort(b: object) -> float:
    """Sort key for budget keys: pct strings ('10%') or plain floats."""
    if isinstance(b, str) and b.endswith("%"):
        return float(b[:-1])
    return float(b)  # type: ignore[arg-type]


def _multi_scene_plot(scenes_agg: dict[str, dict], order: list[str], labels: dict[str, str],
                      metric: str, ylabel: str, suptitle: str, out_path: Path,
                      hline: tuple[float, str] | None = None, ncols: int = 4) -> None:
    """Grid of per-scene subplots, one line per group. Shared legend at top."""
    scene_names = sorted(scenes_agg.keys())
    n = len(scene_names)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.4),
                              squeeze=False, sharey=True)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Shared, data-tight y-axis across all panels (comparable scenes).
    ylim = _data_ylim(
        [cell[f"{metric}_mean"]
         for agg in scenes_agg.values()
         for budgets in agg.values()
         for cell in budgets.values()],
        metric,
    )

    for idx, scene in enumerate(scene_names):
        ax = axes_flat[idx]
        agg = scenes_agg[scene]

        # Build x-axis positions from the first group that has data.
        x_labels = None
        for key in order:
            if key in agg:
                x_labels = sorted(agg[key].keys(), key=_bk_sort)
                break
        if x_labels is None:
            continue
        x_pos = list(range(len(x_labels)))

        for i, key in enumerate(k for k in order if k in agg):
            color  = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]
            d = agg[key]
            y   = [d.get(b, {}).get(f"{metric}_mean", float("nan")) for b in x_labels]
            err = [d.get(b, {}).get(f"{metric}_ci95",  0.0)          for b in x_labels]
            ax.errorbar(x_pos, y, yerr=err, marker=marker, color=color,
                        linewidth=1.5, markersize=4, capsize=2,
                        label=labels.get(key, key))

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(b) for b in x_labels], fontsize=8, rotation=30, ha="right")
        ax.set_title(scene, fontsize=11)
        ax.set_xlabel("Budget %", fontsize=10)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)
        if hline is not None:
            ax.axhline(hline[0], color="gray", linestyle=":", linewidth=1.0)

    # Shared legend above subplots.
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=min(len(lbls), 6),
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(suptitle, fontsize=11, y=1.03)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PSNR/SSIM vs budget")
    parser.add_argument("--summary-csv", default=None,
                        help="Path to metrics/summary.csv (single-CSV mode).")
    parser.add_argument("--multi-scene-glob", default=None, metavar="GLOB",
                        help="Glob for per-scene summary.csv files; enables subplot-grid mode.")
    parser.add_argument("--overlay-csv", default=None,
                        help="Additional CSV to merge into each scene (e.g. baseline from another exp).")
    parser.add_argument("--overlay-schemes", nargs="+", default=None, metavar="SCHEME",
                        help="Schemes to keep from --overlay-csv (default: all).")
    parser.add_argument("--row-filter", nargs="+", default=None, metavar="COL=VAL",
                        help="col=val filters applied to primary rows before aggregation.")
    parser.add_argument("--overlay-filter", nargs="+", default=None, metavar="COL=VAL",
                        help="Additional col=val filters on overlay rows (e.g. weight_mode=screen_area).")
    parser.add_argument("--overlay-rename", nargs="+", default=None, metavar="COL=VAL",
                        help="Set column to value in all overlay rows (e.g. weight_mode=oracle_loo "
                             "to show oracle as a distinct weight_mode group).")
    parser.add_argument("--ncols", type=int, default=4,
                        help="Subplot columns in multi-scene grid (default: 4).")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write PNG plots into")
    parser.add_argument("--title-suffix", default="",
                        help="Appended to plot titles (e.g. '8x8x8 grid')")
    parser.add_argument("--group-by", default="scheme",
                        help="CSV column to group lines by (default: scheme). "
                             "Use 'weight_mode' for setup2, 'w_norm' for setup1, etc.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick-inspect mode: skip NGS plot, use --out-dir as-is.")
    parser.add_argument("--bar", action="store_true",
                        help="Grouped bar chart (budgets on x-axis) instead of line plot.")
    parser.add_argument("--budget-pcts", nargs="+", type=float, default=None, metavar="PCT",
                        help="Budget percentages matching sorted distinct budget_mb values per scene. "
                             "Enables %%-labelled x-axis. Required for multi-scene aggregate bar charts.")
    args = parser.parse_args()
    if args.summary_csv is None and args.multi_scene_glob is None:
        parser.error("one of --summary-csv or --multi-scene-glob is required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""

    # ── multi-scene subplot-grid mode ─────────────────────────────────────────
    if args.multi_scene_glob:
        csv_paths = sorted(Path(p) for p in _glob_mod.glob(args.multi_scene_glob, recursive=True))
        if not csv_paths:
            raise FileNotFoundError(f"No CSVs match: {args.multi_scene_glob}")

        # Load rows per scene.
        rows_by_scene: dict[str, list[dict]] = {}
        for p in csv_paths:
            scene_rows = _read_csv(p)
            if not scene_rows:
                continue
            scene = scene_rows[0].get("scene") or p.parent.parent.name
            rows_by_scene.setdefault(scene, []).extend(scene_rows)

        # Apply primary row filter.
        if args.row_filter:
            for scene, scene_rows in rows_by_scene.items():
                for kv in args.row_filter:
                    col, val = kv.split("=", 1)
                    scene_rows = [r for r in scene_rows if r.get(col) == val]
                rows_by_scene[scene] = scene_rows

        # Apply budget labels per scene.
        if args.budget_pcts:
            for scene_rows in rows_by_scene.values():
                _apply_budget_labels(scene_rows, args.budget_pcts)

        # Load + merge overlay CSV (e.g. baselines from a different experiment).
        o_rows = _load_overlay_rows(args)
        if o_rows:
            if args.budget_pcts:
                _apply_budget_labels(o_rows, args.budget_pcts)
            for r in o_rows:
                scene = r.get("scene", "_")
                if scene in rows_by_scene:
                    rows_by_scene[scene].append(r)

        # Aggregate each scene and build unified order.
        scenes_agg: dict[str, dict] = {
            scene: _aggregate(scene_rows, args.group_by)
            for scene, scene_rows in sorted(rows_by_scene.items())
        }
        all_keys: set[str] = {k for agg in scenes_agg.values() for k in agg}
        first_agg = next(iter(scenes_agg.values()))
        order, labels = _resolve_order_and_labels(first_agg, args.group_by)
        # Ensure overlay schemes appear in order even if absent from first scene's agg.
        extras = [k for k in sorted(all_keys) if k not in order]
        order = order + extras

        n_cameras = max(
            (v["n_cameras"] for agg in scenes_agg.values() for s in agg.values() for v in s.values()),
            default=0,
        )
        n_scenes = len(scenes_agg)
        n_groups = len([k for k in order if any(k in agg for agg in scenes_agg.values())])
        title_base = (f"{n_groups} {args.group_by}s, {n_scenes} scenes, "
                      f"{n_cameras} cameras/budget{suffix}")

        _multi_scene_plot(scenes_agg, order, labels, "psnr", "PSNR (dB)",
                          f"PSNR per-scene — {title_base}",
                          out_dir / "psnr_vs_budget.png",
                          hline=(60.0, "≥60 dB"), ncols=args.ncols)
        _multi_scene_plot(scenes_agg, order, labels, "ssim", "SSIM",
                          f"SSIM per-scene — {title_base}",
                          out_dir / "ssim_vs_budget.png",
                          ncols=args.ncols)
        return

    # ── single-CSV mode ───────────────────────────────────────────────────────
    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    rows = _read_csv(summary_csv)
    if args.row_filter:
        for kv in args.row_filter:
            col, val = kv.split("=", 1)
            rows = [r for r in rows if r.get(col) == val]
    if args.budget_pcts:
        rows = _apply_budget_labels(rows, args.budget_pcts)
    o_rows = _load_overlay_rows(args)
    if o_rows:
        if args.budget_pcts:
            _apply_budget_labels(o_rows, args.budget_pcts)
        rows.extend(o_rows)
    print(f"Loaded {len(rows)} rows from {summary_csv}, grouping by '{args.group_by}'")
    agg = _aggregate(rows, args.group_by)
    order, labels = _resolve_order_and_labels(agg, args.group_by)

    n_cameras = max(v["n_cameras"] for s in agg.values() for v in s.values())
    n_groups = len([k for k in order if k in agg])

    # Scene-N reference for the gaussian-ratio plot: max selected_gaussians observed
    # across the whole CSV. Reaches true N only when at least one (group, budget)
    # cell saturates (i.e. budget >= scene size). Otherwise it's a lower bound, and
    # the plotted ratio is an upper bound (overestimates selection fraction).
    scene_n_gs = max(
        (v["ngs_mean"] for s in agg.values() for v in s.values()),
        default=0.0,
    )
    if scene_n_gs > 0:
        for s in agg.values():
            for v in s.values():
                v["ngs_mean"] /= scene_n_gs
                v["ngs_ci95"] /= scene_n_gs

    plot_fn = _bar_plot if args.bar else _plot
    ylabel_psnr = "PSNR (dB, clamped 60)" if args.bar else "PSNR (dB)"

    plot_fn(agg, order, labels, "psnr", ylabel_psnr,
            f"PSNR vs Budget — {n_groups} {args.group_by}s (mean ± 95% CI, {n_cameras} cameras/budget){suffix}",
            out_dir / "psnr_vs_budget.png",
            hline=(60.0, "saturated (≥60 dB)"))

    if not args.quick:
        _plot(agg, order, labels, "ngs", "selected / total Gaussians",
              f"Selection ratio vs Budget — {n_groups} {args.group_by}s (mean ± 95% CI, {n_cameras} cameras/budget){suffix}",
              out_dir / "ngs_vs_budget.png",
              hline=(1.0, f"full scene (≈{int(scene_n_gs):,} GS)") if scene_n_gs > 0 else None)

    plot_fn(agg, order, labels, "ssim", "SSIM",
            f"SSIM vs Budget — {n_groups} {args.group_by}s (mean ± 95% CI, {n_cameras} cameras/budget){suffix}",
            out_dir / "ssim_vs_budget.png")


if __name__ == "__main__":
    main()
