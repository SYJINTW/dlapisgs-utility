#!/usr/bin/env python3
"""Plot PSNR and SSIM vs budget from a render+metrics sweep.

Subcommands:
  single  one CSV, one scene, line plot          (x-axis: budget MiB)
  grid    multi-scene subplot grid, lines         (x-axis: budget %)
  line    cross-scene aggregate, connected line   (x-axis: budget %) -- default cross-scene view
  bar     cross-scene aggregate bars               (x-axis: budget %) -- unordered categories only
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

_SCHEME_LABELS = {
    "vd_lod":          "VD+LOD (baseline)",
    "vd_lod_w":        "VD+LOD+W (ours)",
    "v_lod_w":         "V+LOD+W (no distance)",
    "ml":              "ML (ours)",
    "ml_rf":           "ML-RF",
    "ml_lgbm":         "ML-LGBM",
    "ml_xgb":          "ML-XGB",
    "prog_no_cull":    "Progressive (no cull, 1³)",
    "prog_culled":     "Progressive (culled, 8³)",
    "oracle_tp":       "Oracle tile_partial (8³)",
    "ml_tp":           "ML tile_partial (8³)",
    "vd_lod_c":        "VD+LOD+C",
    "vd_lod_w_c":      "VD+LOD+W+C (proposed)",
    "vd_lod_w_sa":     "VD+LOD+W (screen_area)",
    "vd_lod_w_vd2":    "VD+LOD+W (vol/d²)",
    "oracle_loo":      "Oracle-LOO (upper bound)",
    "oracle_loo_ssim": "Oracle-LOO-SSIM (upper bound)",
    "oracle_aoi":      "Oracle-AOI",
    "oracle_combined": "Oracle-combined",
}
_SCHEME_ORDER = [
    "prog_no_cull", "prog_culled", "oracle_tp", "ml_tp",
    "vd_lod", "vd_lod_w", "v_lod_w", "ml", "ml_rf", "ml_lgbm", "ml_xgb",
    "vd_lod_w_sa", "vd_lod_w_vd2", "vd_lod_c", "vd_lod_w_c",
    "oracle_aoi", "oracle_combined", "oracle_loo", "oracle_loo_ssim",
]
_WEIGHT_MODE_ORDER  = ["det_gamma_over_d2", "volume", "volume_over_d2", "screen_area"]
_WEIGHT_MODE_LABELS = {
    "det_gamma_over_d2": "det_γ/d² (original)",
    "volume":            "volume (view-indep.)",
    "volume_over_d2":    "volume/d² (view-dep.)",
    "screen_area":       "screen_area (projected footprint)",
}

MARKERS = ["s", "^", "D", "o", "v", "P"]
COLORS  = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#FF8C00", "#00CED1"]
PSNR_SATURATION_DB = 60.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _bk_sort(b: object) -> float:
    if isinstance(b, str) and b.endswith("%"):
        return float(b[:-1])
    return float(b)  # type: ignore[arg-type]


def _apply_filters(rows: list[dict], filters: list[str] | None,
                   excludes: list[str] | None) -> list[dict]:
    for kv in (filters or []):
        col, val = kv.split("=", 1)
        rows = [r for r in rows if r.get(col) == val]
    for kv in (excludes or []):
        col, val = kv.split("=", 1)
        rows = [r for r in rows if r.get(col) != val]
    return rows


def _apply_budget_labels(rows: list[dict], budget_pcts: list[float]) -> list[dict]:
    """Map per-scene budget_mb rank → percentage label (adds '_budget_key' in-place)."""
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


def _load_overlay(overlay_csv: str | None, overlay_filter: list[str] | None,
                  overlay_rename: list[str] | None,
                  overlay_schemes: list[str] | None) -> list[dict]:
    if not overlay_csv:
        return []
    rows = _read_csv(Path(overlay_csv))
    if overlay_schemes:
        keep = set(overlay_schemes)
        rows = [r for r in rows if r.get("scheme") in keep]
    rows = _apply_filters(rows, overlay_filter, None)
    for kv in (overlay_rename or []):
        col, val = kv.split("=", 1)
        for r in rows:
            r[col] = val
    return rows


def _aggregate(rows: list[dict], group_by: str) -> dict[str, dict]:
    # Two-stage: cam→scene mean, then CI over n_scenes (fixes 12× too-tight CI from pooling).
    buckets: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        key = r.get(group_by) or f"missing_{group_by}"
        bk = r["_budget_key"] if "_budget_key" in r else float(r["budget_mb"])
        scene = r.get("scene", "_")
        buckets[key][bk][scene].append(r)
    result: dict[str, dict] = {}
    for key, budgets in buckets.items():
        result[key] = {}
        for bk, scenes in budgets.items():
            spsnr, sssim, sngs = [], [], []
            n_cams = 0
            for entries in scenes.values():
                spsnr.append(np.minimum(
                    np.array([float(e["psnr"]) for e in entries]), PSNR_SATURATION_DB).mean())
                sssim.append(np.array([float(e["ssim"]) for e in entries]).mean())
                sngs.append(np.array([float(e.get("selected_gaussians", 0)) for e in entries]).mean())
                n_cams += len(entries)
            spsnr = np.array(spsnr); sssim = np.array(sssim); sngs = np.array(sngs)
            n = len(spsnr)
            ci = 1.96 / max(np.sqrt(n), 1.0)
            result[key][bk] = {
                "psnr_mean": float(spsnr.mean()),
                "psnr_ci95": float(spsnr.std(ddof=1) * ci) if n > 1 else 0.0,
                "ssim_mean": float(sssim.mean()),
                "ssim_ci95": float(sssim.std(ddof=1) * ci) if n > 1 else 0.0,
                "ngs_mean":  float(sngs.mean()),
                "ngs_ci95":  float(sngs.std(ddof=1) * ci) if n > 1 else 0.0,
                "n": n,
                "n_cameras": n_cams,
            }
    return result


def _resolve_order_and_labels(agg: dict, group_by: str) -> tuple[list[str], dict[str, str]]:
    if group_by == "scheme":
        order, labels = _SCHEME_ORDER, _SCHEME_LABELS
    elif group_by == "weight_mode":
        order, labels = _WEIGHT_MODE_ORDER, _WEIGHT_MODE_LABELS
    else:
        order, labels = sorted(agg.keys()), {}
    extras = [k for k in sorted(agg.keys()) if k not in order]
    return order + extras, labels


def _data_ylim(values: list[float], metric: str) -> tuple[float, float]:
    vals = [v for v in values if v == v]
    if not vals:
        return (0.0, PSNR_SATURATION_DB) if metric == "psnr" else (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    if metric == "psnr":
        floor = max(0.0, np.floor((lo - 3) / 5) * 5)
        ceil  = min(PSNR_SATURATION_DB * 1.05, max(PSNR_SATURATION_DB * 1.05, hi + 5))
        return floor, ceil
    floor = max(0.0, np.floor((lo - 0.03) * 20) / 20)
    return floor, min(1.005, hi + 0.02)


# ── plot primitives ───────────────────────────────────────────────────────────

def _line_plot(agg: dict, order: list[str], labels: dict[str, str],
               metric: str, ylabel: str, title: str, out_path: Path,
               hline: tuple[float, str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    all_means: list[float] = []
    present = [k for k in order if k in agg]
    for i, key in enumerate(present):
        pts = sorted(agg[key].items(), key=lambda p: _bk_sort(p[0]))
        x   = [p[0] for p in pts]
        y   = [p[1][f"{metric}_mean"] for p in pts]
        err = [p[1][f"{metric}_ci95"] for p in pts]
        all_means.extend(y)
        ax.errorbar(x, y, yerr=err, marker=MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)], linewidth=1.8, markersize=6,
                    capsize=3, label=labels.get(key, key))
    ax.set_ylim(*_data_ylim(all_means, metric))
    ax.set_xlabel("Budget (MiB)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25)
    xmax = max((p[0] for k in order if k in agg for p in agg[k].items()), default=None)
    if xmax is not None:
        ax.set_xlim(right=float(xmax) * 1.05)
    if hline:
        y_ref, hlabel = hline
        ax.axhline(y_ref, color="gray", linestyle=":", linewidth=1.2, zorder=0)
        ax.text(ax.get_xlim()[1], y_ref, f" {hlabel}", fontsize=9,
                color="gray", va="bottom", ha="right")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _bar_plot(agg: dict, order: list[str], labels: dict[str, str],
              metric: str, ylabel: str, title: str, out_path: Path,
              hline: tuple[float, str] | None = None) -> None:
    all_budgets = sorted({b for k in agg for b in agg[k]}, key=_bk_sort)
    groups = [k for k in order if k in agg]
    n_budgets, n_groups = len(all_budgets), len(groups)
    bar_width = 0.8 / max(n_groups, 1)
    x = np.arange(n_budgets)

    fig, ax = plt.subplots(figsize=(max(11.5, n_budgets * 1.6), 5.5))
    all_means: list[float] = []
    for i, key in enumerate(groups):
        y   = [agg[key].get(b, {}).get(f"{metric}_mean", 0.0) for b in all_budgets]
        err = [agg[key].get(b, {}).get(f"{metric}_ci95",  0.0) for b in all_budgets]
        all_means.extend(v for b, v in zip(all_budgets, y) if b in agg[key])
        offset = (i - n_groups / 2 + 0.5) * bar_width
        ax.bar(x + offset, y, bar_width * 0.92, yerr=err, capsize=2,
               color=COLORS[i % len(COLORS)], label=labels.get(key, key),
               error_kw={"linewidth": 0.8})
    ax.set_ylim(*_data_ylim(all_means, metric))
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in all_budgets], fontsize=10)
    pct_labels = bool(all_budgets) and isinstance(all_budgets[0], str)
    _pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
    ax.set_xlabel(f"Budget ({_pct} of full scene)" if pct_labels else "Budget (MiB)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    if hline:
        y_ref, hlabel = hline
        ax.axhline(y_ref, color="gray", linestyle=":", linewidth=1.2, zorder=0)
        ax.text(n_budgets - 0.5, y_ref, f" {hlabel}", fontsize=9,
                color="gray", va="bottom", ha="right")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _agg_line_plot(agg: dict, order: list[str], labels: dict[str, str],
                    metric: str, ylabel: str, title: str, out_path: Path,
                    hline: tuple[float, str] | None = None) -> None:
    """Cross-scene aggregate, connected line (budget %% x-axis). Same data as
    `_bar_plot` but a trend across an ordered x-axis reads as a line, not bars."""
    all_budgets = sorted({b for k in agg for b in agg[k]}, key=_bk_sort)
    groups = [k for k in order if k in agg]
    x = np.arange(len(all_budgets))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    all_means: list[float] = []
    for i, key in enumerate(groups):
        y   = [agg[key].get(b, {}).get(f"{metric}_mean", float("nan")) for b in all_budgets]
        err = [agg[key].get(b, {}).get(f"{metric}_ci95",  0.0) for b in all_budgets]
        all_means.extend(v for b, v in zip(all_budgets, y) if b in agg[key])
        ax.errorbar(x, y, yerr=err, marker=MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)], linewidth=1.8, markersize=6,
                    capsize=3, label=labels.get(key, key))
    ax.set_ylim(*_data_ylim(all_means, metric))
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in all_budgets], fontsize=10)
    pct_labels = bool(all_budgets) and isinstance(all_budgets[0], str)
    _pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
    ax.set_xlabel(f"Budget ({_pct} of full scene)" if pct_labels else "Budget (MiB)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25)
    if hline:
        y_ref, hlabel = hline
        ax.axhline(y_ref, color="gray", linestyle=":", linewidth=1.2, zorder=0)
        ax.text(len(all_budgets) - 1, y_ref, f" {hlabel}", fontsize=9,
                color="gray", va="bottom", ha="right")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _grid_plot(scenes_agg: dict[str, dict], order: list[str], labels: dict[str, str],
               metric: str, ylabel: str, suptitle: str, out_path: Path,
               hline: tuple[float, str] | None = None, ncols: int = 4) -> None:
    scene_names = list(scenes_agg.keys())  # insertion order = caller-controlled
    n = len(scene_names)
    ncols = min(ncols, n)
    global_present = [k for k in order if any(k in agg for agg in scenes_agg.values())]
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.4),
                             squeeze=False, sharey=True)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

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
        x_labels = next(
            (sorted(agg[k].keys(), key=_bk_sort) for k in order if k in agg), None)
        if x_labels is None:
            continue
        x_pos = list(range(len(x_labels)))
        for key in (k for k in order if k in agg):
            i = global_present.index(key)
            d = agg[key]
            y   = [d.get(b, {}).get(f"{metric}_mean", float("nan")) for b in x_labels]
            err = [d.get(b, {}).get(f"{metric}_ci95", 0.0) for b in x_labels]
            ax.errorbar(x_pos, y, yerr=err, marker=MARKERS[i % len(MARKERS)],
                        color=COLORS[i % len(COLORS)], linewidth=1.5, markersize=4,
                        capsize=2, label=labels.get(key, key))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(b) for b in x_labels], fontsize=8, rotation=30, ha="right")
        ax.set_title(scene, fontsize=11)
        _pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
        ax.set_xlabel(f"Budget {_pct}", fontsize=10)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)
        if hline:
            ax.axhline(hline[0], color="gray", linestyle=":", linewidth=1.0)

    seen: dict[str, object] = {}
    for ax in axes_flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h
    lbls = [labels.get(l, l) for l in order if labels.get(l, l) in seen]
    handles = [seen[l] for l in lbls]
    fig.legend(handles, lbls, loc="upper center", ncol=min(len(lbls), 6),
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(suptitle, fontsize=11, y=1.03)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── subcommand handlers ───────────────────────────────────────────────────────

def cmd_single(args: argparse.Namespace) -> None:
    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    rows = _read_csv(summary_csv)
    rows = _apply_filters(rows, args.filters, args.excludes)
    o_rows = _load_overlay(args.overlay_csv, args.overlay_filter,
                           args.overlay_rename, args.overlay_schemes)
    rows.extend(o_rows)
    print(f"Loaded {len(rows)} rows from {summary_csv}, grouping by '{args.group_by}'")
    agg = _aggregate(rows, args.group_by)
    order, labels = _resolve_order_and_labels(agg, args.group_by)

    n_cameras = max(v["n_cameras"] for s in agg.values() for v in s.values())
    n_groups  = len([k for k in order if k in agg])
    suffix    = f" — {args.title_suffix}" if args.title_suffix else ""

    scene_n_gs = max((v["ngs_mean"] for s in agg.values() for v in s.values()), default=0.0)
    if scene_n_gs > 0:
        for s in agg.values():
            for v in s.values():
                v["ngs_mean"] /= scene_n_gs
                v["ngs_ci95"] /= scene_n_gs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{n_groups} {args.group_by}s (mean ± 95% CI, {n_cameras} cameras/budget){suffix}"
    _line_plot(agg, order, labels, "psnr", "PSNR (dB)",
               f"PSNR vs Budget — {base}", out_dir / "psnr_vs_budget.png",
               hline=(PSNR_SATURATION_DB, "saturated (≥60 dB)"))
    _line_plot(agg, order, labels, "ssim", "SSIM",
               f"SSIM vs Budget — {base}", out_dir / "ssim_vs_budget.png")
    if not args.quick:
        _line_plot(agg, order, labels, "ngs", "selected / total Gaussians",
                   f"Selection ratio vs Budget — {base}", out_dir / "ngs_vs_budget.png",
                   hline=(1.0, f"full scene (≈{int(scene_n_gs):,} GS)") if scene_n_gs > 0 else None)


def cmd_grid(args: argparse.Namespace) -> None:
    csv_paths = sorted(Path(p) for p in _glob_mod.glob(args.glob_pattern, recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs match: {args.glob_pattern}")

    rows_by_scene: dict[str, list[dict]] = {}
    for p in csv_paths:
        scene_rows = _read_csv(p)
        if not scene_rows:
            continue
        scene = scene_rows[0].get("scene") or p.parent.parent.name
        rows_by_scene.setdefault(scene, []).extend(scene_rows)

    for scene in list(rows_by_scene):
        rows_by_scene[scene] = _apply_filters(
            rows_by_scene[scene], args.filters, args.excludes)

    for scene_rows in rows_by_scene.values():
        _apply_budget_labels(scene_rows, args.budget_pcts)

    o_rows = _load_overlay(args.overlay_csv, args.overlay_filter,
                           args.overlay_rename, args.overlay_schemes)
    if o_rows:
        _apply_budget_labels(o_rows, args.budget_pcts)
        for r in o_rows:
            scene = r.get("scene", "_")
            if scene in rows_by_scene:
                rows_by_scene[scene].append(r)

    if args.scene_order:
        known = [s for s in args.scene_order if s in rows_by_scene]
        rest  = sorted(s for s in rows_by_scene if s not in args.scene_order)
        key_order = known + rest
    else:
        key_order = sorted(rows_by_scene.keys())

    scenes_agg = {s: _aggregate(rows_by_scene[s], args.group_by) for s in key_order}
    all_keys: set[str] = {k for agg in scenes_agg.values() for k in agg}
    order, labels = _resolve_order_and_labels({k: {} for k in all_keys}, args.group_by)

    n_cameras = max(
        (v["n_cameras"] for agg in scenes_agg.values()
         for s in agg.values() for v in s.values()), default=0)
    n_scenes  = len(scenes_agg)
    n_groups  = len([k for k in order if any(k in agg for agg in scenes_agg.values())])
    suffix    = f" — {args.title_suffix}" if args.title_suffix else ""
    base      = f"{n_groups} {args.group_by}s, {n_scenes} scenes, {n_cameras} cameras/budget{suffix}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _grid_plot(scenes_agg, order, labels, "psnr", "PSNR (dB)",
               f"PSNR per-scene — {base}", out_dir / "psnr_vs_budget.png",
               hline=(PSNR_SATURATION_DB, "≥60 dB"), ncols=args.ncols)
    _grid_plot(scenes_agg, order, labels, "ssim", "SSIM",
               f"SSIM per-scene — {base}", out_dir / "ssim_vs_budget.png",
               ncols=args.ncols)


def cmd_line(args: argparse.Namespace) -> None:
    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    rows = _read_csv(summary_csv)
    print(f"Loaded {len(rows)} rows from {summary_csv}, grouping by '{args.group_by}'")
    rows = _apply_filters(rows, args.filters, args.excludes)
    _apply_budget_labels(rows, args.budget_pcts)
    o_rows = _load_overlay(args.overlay_csv, args.overlay_filter,
                           args.overlay_rename, args.overlay_schemes)
    if o_rows:
        _apply_budget_labels(o_rows, args.budget_pcts)
        rows.extend(o_rows)
    agg = _aggregate(rows, args.group_by)
    order, labels = _resolve_order_and_labels(agg, args.group_by)

    n_cameras = max(v["n_cameras"] for s in agg.values() for v in s.values())
    n_groups  = len([k for k in order if k in agg])
    suffix    = f" — {args.title_suffix}" if args.title_suffix else ""
    base      = f"{n_groups} {args.group_by}s (mean ± 95% CI, {n_cameras} cameras/budget){suffix}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _agg_line_plot(agg, order, labels, "psnr", "PSNR (dB, clamped 60)",
                   f"PSNR vs Budget — {base}", out_dir / "psnr_vs_budget.png",
                   hline=(PSNR_SATURATION_DB, "saturated (≥60 dB)"))
    _agg_line_plot(agg, order, labels, "ssim", "SSIM",
                   f"SSIM vs Budget — {base}", out_dir / "ssim_vs_budget.png")


def cmd_bar(args: argparse.Namespace) -> None:
    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    rows = _read_csv(summary_csv)
    print(f"Loaded {len(rows)} rows from {summary_csv}, grouping by '{args.group_by}'")
    rows = _apply_filters(rows, args.filters, args.excludes)
    _apply_budget_labels(rows, args.budget_pcts)
    o_rows = _load_overlay(args.overlay_csv, args.overlay_filter,
                           args.overlay_rename, args.overlay_schemes)
    if o_rows:
        _apply_budget_labels(o_rows, args.budget_pcts)
        rows.extend(o_rows)
    agg = _aggregate(rows, args.group_by)
    order, labels = _resolve_order_and_labels(agg, args.group_by)

    n_cameras = max(v["n_cameras"] for s in agg.values() for v in s.values())
    n_groups  = len([k for k in order if k in agg])
    suffix    = f" — {args.title_suffix}" if args.title_suffix else ""
    base      = f"{n_groups} {args.group_by}s (mean ± 95% CI, {n_cameras} cameras/budget){suffix}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _bar_plot(agg, order, labels, "psnr", "PSNR (dB, clamped 60)",
              f"PSNR vs Budget — {base}", out_dir / "psnr_vs_budget.png",
              hline=(PSNR_SATURATION_DB, "saturated (≥60 dB)"))
    _bar_plot(agg, order, labels, "ssim", "SSIM",
              f"SSIM vs Budget — {base}", out_dir / "ssim_vs_budget.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", required=True)
    common.add_argument("--group-by", default="scheme",
                        help="CSV column to group by (default: scheme).")
    common.add_argument("--title-suffix", default="")
    common.add_argument("--quick", action="store_true",
                        help="Skip NGS plot; use --out-dir as-is.")
    common.add_argument("--filter", nargs="+", default=None, metavar="COL=VAL",
                        dest="filters", help="Keep rows matching col=val.")
    common.add_argument("--exclude", nargs="+", default=None, metavar="COL=VAL",
                        dest="excludes", help="Drop rows matching col=val.")
    common.add_argument("--overlay-csv", default=None)
    common.add_argument("--overlay-filter", nargs="+", default=None, metavar="COL=VAL")
    common.add_argument("--overlay-rename", nargs="+", default=None, metavar="COL=VAL")
    common.add_argument("--overlay-schemes", nargs="+", default=None)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("single", parents=[common],
                       help="Line plot for one scene CSV.")
    p.add_argument("--summary-csv", required=True)

    p = sub.add_parser("grid", parents=[common],
                       help="Per-scene subplot grid (budget %% x-axis).")
    p.add_argument("--glob", required=True, dest="glob_pattern",
                   help="Glob for per-scene summary CSVs.")
    p.add_argument("--budget-pcts", nargs="+", type=float, required=True, metavar="PCT")
    p.add_argument("--scene-order", nargs="+", default=None, metavar="SCENE",
                   help="Real scenes first; rest appended alphabetically.")
    p.add_argument("--ncols", type=int, default=4)

    p = sub.add_parser("line", parents=[common],
                       help="Aggregate connected-line plot across scenes (budget %% x-axis). "
                            "Default cross-scene view — use this unless the groups are "
                            "unordered categories.")
    p.add_argument("--summary-csv", required=True,
                   help="Combined summary_all.csv covering all scenes.")
    p.add_argument("--budget-pcts", nargs="+", type=float, required=True, metavar="PCT")

    p = sub.add_parser("bar", parents=[common],
                       help="Aggregate bar chart across scenes (budget %% x-axis). "
                            "Only for unordered categorical comparisons; a budget sweep "
                            "should use 'line' instead.")
    p.add_argument("--summary-csv", required=True,
                   help="Combined summary_all.csv covering all scenes.")
    p.add_argument("--budget-pcts", nargs="+", type=float, required=True, metavar="PCT")

    args = parser.parse_args()
    {"single": cmd_single, "grid": cmd_grid, "line": cmd_line, "bar": cmd_bar}[args.cmd](args)


if __name__ == "__main__":
    main()
