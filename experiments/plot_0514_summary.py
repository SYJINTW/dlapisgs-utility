#!/usr/bin/env python3
"""Aggregate 0514 sweep results across scenes and produce summary plots.

Reads:
  output/0514/exp1_gs_weight/summary_all.csv                    (exp1)
  output/0514/exp2_tile_utility/<scene>/screen_area/metrics/summary.csv  (exp2)

Writes (under output/0514/summary_plots/):
  exp1_bar_psnr.png, exp1_bar_ssim.png        — bars grouped by weight_mode, x=budget%
  exp1_lines_psnr.png, exp1_lines_ssim.png    — per-scene grid of line plots
  exp2_bar_psnr.png, exp2_bar_ssim.png        — bars grouped by scheme @ screen_area
  exp2_lines_psnr.png, exp2_lines_ssim.png    — per-scene grid of line plots

Notes:
- Per-scene budgets are absolute MiB and differ across scenes; we normalize
  to budget% via budget_mb / max(budget_mb per scene).
- Per-camera PSNR is clamped to 60 dB before averaging (matches plot_metrics.py).
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/data1/samk/gs-quic/cs5262_tile_quic/dlapisgs-utility/output/0514")
OUT_DIR = ROOT / "summary_plots"
OUT_DIR.mkdir(exist_ok=True)

PSNR_SAT = 60.0

WEIGHT_MODE_ORDER = ["det_gamma_over_d2", "volume", "volume_over_d2", "screen_area"]
WEIGHT_MODE_LABEL = {
    "det_gamma_over_d2": "det_γ/d²",
    "volume":            "volume",
    "volume_over_d2":    "volume/d²",
    "screen_area":       "screen_area",
}
SCHEME_ORDER = ["vd_lod", "vd_lod_w", "vd_lod_c", "vd_lod_w_c"]
SCHEME_LABEL = {
    "vd_lod":     "VD+LOD",
    "vd_lod_w":   "VD+LOD+W",
    "vd_lod_c":   "VD+LOD+C",
    "vd_lod_w_c": "VD+LOD+W+C",
}
COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
MARKERS = ["s", "^", "D", "o"]
SCENES = ["chair", "drums", "ficus", "hotdog", "materials", "mic", "ship", "bicycle"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def attach_budget_pct(rows: list[dict]) -> None:
    """Add budget_pct (rounded to nearest int) per row, per-scene normalized."""
    max_by_scene: dict[str, float] = defaultdict(float)
    for r in rows:
        b = float(r["budget_mb"])
        if b > max_by_scene[r["scene"]]:
            max_by_scene[r["scene"]] = b
    for r in rows:
        pct = 100.0 * float(r["budget_mb"]) / max_by_scene[r["scene"]]
        r["budget_pct"] = int(round(pct))


def agg_per_cell(rows: list[dict], group_keys: tuple[str, ...]
                 ) -> dict[tuple, dict]:
    """Aggregate rows by (group_keys) → mean PSNR/SSIM across cameras (PSNR clamped)."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        buckets[key].append(r)
    out: dict[tuple, dict] = {}
    for key, entries in buckets.items():
        psnr = np.minimum(np.array([float(e["psnr"]) for e in entries]), PSNR_SAT)
        ssim = np.array([float(e["ssim"]) for e in entries])
        out[key] = {"psnr": float(psnr.mean()),
                    "ssim": float(ssim.mean()),
                    "n": len(entries)}
    return out


# ---------- Load data ----------

print("Loading exp1...")
exp1_rows = read_csv(ROOT / "exp1_gs_weight" / "summary_all.csv")
attach_budget_pct(exp1_rows)
exp1_cell = agg_per_cell(exp1_rows, ("scene", "weight_mode", "budget_pct"))

print("Loading exp2 (screen_area + volume_over_d2)...")
exp2_cells: dict[str, dict] = {}
for wm in ("screen_area", "volume_over_d2"):
    rows: list[dict] = []
    for scene in SCENES:
        p = ROOT / "exp2_tile_utility" / scene / wm / "metrics" / "summary.csv"
        if p.exists():
            rows.extend(read_csv(p))
    attach_budget_pct(rows)
    exp2_cells[wm] = agg_per_cell(rows, ("scene", "scheme", "budget_pct"))

# Sorted budget% set used by both experiments
BUDGETS = sorted({k[-1] for k in exp1_cell})
print("Budget%:", BUDGETS)


# ---------- Plotters ----------

def _data_ylim(values, metric: str) -> tuple[float, float]:
    """Tight y-limits with a small pad below the minimum mean — keeps dynamic range."""
    if not values:
        return (0.0, PSNR_SAT) if metric == "psnr" else (0.0, 1.0)
    lo, hi = min(values), max(values)
    if metric == "psnr":
        floor = max(0.0, np.floor((lo - 3) / 5) * 5)  # round down to nearest 5 dB
        ceil  = min(PSNR_SAT * 1.05, max(PSNR_SAT * 1.02, hi + 2))
        return floor, ceil
    # ssim: pad ±0.02, never escape [0, 1]
    floor = max(0.0, np.floor((lo - 0.03) * 20) / 20)  # nearest 0.05
    ceil  = min(1.005, hi + 0.02)
    return floor, ceil


def _ylim_for_grid(cell, series_keys, metric):
    """Collect every scene/series/budget mean so per-scene panels share a tight axis."""
    vals = [cell[k][metric] for k in cell if k[1] in series_keys]
    return _data_ylim(vals, metric)


def bar_plot(cell: dict[tuple, dict],
             series_keys: list[str],
             series_label: dict[str, str],
             metric: str,
             ylabel: str,
             title: str,
             out_path: Path) -> None:
    """Grouped bars: x=budget_pct, groups=series_keys; y=mean across scenes."""
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    width = 0.8 / max(len(series_keys), 1)
    xs = np.arange(len(BUDGETS))

    all_means: list[float] = []
    for i, sk in enumerate(series_keys):
        ys, ers = [], []
        for b in BUDGETS:
            vals = []
            for scene in SCENES:
                hit = cell.get((scene, sk, b))
                if hit is not None:
                    vals.append(hit[metric])
            if vals:
                arr = np.array(vals)
                ys.append(arr.mean())
                ers.append(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)
            else:
                ys.append(np.nan)
                ers.append(0.0)
        all_means.extend(y for y in ys if not np.isnan(y))
        ax.bar(xs + (i - (len(series_keys) - 1) / 2) * width, ys, width,
               yerr=ers, capsize=2.5, color=COLORS[i % len(COLORS)],
               label=series_label.get(sk, sk))

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}%" for b in BUDGETS])
    ax.set_xlabel("Budget (% of full scene size)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title + f"  —  mean across {len(SCENES)} scenes", fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    ymin, ymax = _data_ylim(all_means, metric)
    ax.set_ylim(ymin, ymax)
    if metric == "psnr":
        ax.axhline(PSNR_SAT, color="gray", linestyle=":", linewidth=1.0, zorder=0)
        ax.text(xs[-1], PSNR_SAT, "  60 dB (saturated)",
                fontsize=8, color="gray", va="bottom", ha="right")
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def per_scene_grid(cell: dict[tuple, dict],
                   series_keys: list[str],
                   series_label: dict[str, str],
                   metric: str,
                   ylabel: str,
                   title: str,
                   out_path: Path) -> None:
    """4x2 grid of line plots, one panel per scene; x=budget%, lines per series."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharey=True)
    ymin, ymax = _ylim_for_grid(cell, series_keys, metric)
    for ax, scene in zip(axes.flat, SCENES):
        for i, sk in enumerate(series_keys):
            ys = []
            for b in BUDGETS:
                hit = cell.get((scene, sk, b))
                ys.append(hit[metric] if hit else np.nan)
            ax.plot(BUDGETS, ys,
                    marker=MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)],
                    linewidth=1.6, markersize=5,
                    label=series_label.get(sk, sk))
        ax.set_title(scene, fontsize=11)
        ax.set_xlabel("Budget %", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_ylim(ymin, ymax)
        if metric == "psnr":
            ax.axhline(PSNR_SAT, color="gray", linestyle=":", linewidth=0.8, zorder=0)
    axes[0, 0].set_ylabel(ylabel, fontsize=11)
    axes[1, 0].set_ylabel(ylabel, fontsize=11)
    # one legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(series_keys), bbox_to_anchor=(0.5, 1.0),
               fontsize=11, frameon=False)
    fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------- Exp1: weight_mode ablation ----------

exp1_series = [w for w in WEIGHT_MODE_ORDER
               if any(k[1] == w for k in exp1_cell)]
print("Exp1 weight modes present:", exp1_series)

bar_plot(exp1_cell, exp1_series, WEIGHT_MODE_LABEL,
         "psnr", "PSNR (dB, clamped 60)",
         "Exp 1 — per-Gaussian weight ablation (scheme=vd_lod_w_c, packing=progressive)",
         OUT_DIR / "exp1_bar_psnr.png")
bar_plot(exp1_cell, exp1_series, WEIGHT_MODE_LABEL,
         "ssim", "SSIM",
         "Exp 1 — per-Gaussian weight ablation (scheme=vd_lod_w_c, packing=progressive)",
         OUT_DIR / "exp1_bar_ssim.png")
per_scene_grid(exp1_cell, exp1_series, WEIGHT_MODE_LABEL,
               "psnr", "PSNR (dB)",
               "Exp 1 per-scene — PSNR vs budget % by weight_mode  (packing=progressive)",
               OUT_DIR / "exp1_lines_psnr.png")
per_scene_grid(exp1_cell, exp1_series, WEIGHT_MODE_LABEL,
               "ssim", "SSIM",
               "Exp 1 per-scene — SSIM vs budget % by weight_mode  (packing=progressive)",
               OUT_DIR / "exp1_lines_ssim.png")


# ---------- Exp2: scheme ablation (weight=screen_area) ----------

for wm, exp2_cell in exp2_cells.items():
    exp2_series = [s for s in SCHEME_ORDER if any(k[1] == s for k in exp2_cell)]
    print(f"Exp2 schemes present (weight={wm}):", exp2_series)
    tag = f"weight={wm}, packing=tile_strict"
    bar_plot(exp2_cell, exp2_series, SCHEME_LABEL,
             "psnr", "PSNR (dB, clamped 60)",
             f"Exp 2 — tile utility scheme ablation ({tag})",
             OUT_DIR / f"exp2_{wm}_bar_psnr.png")
    bar_plot(exp2_cell, exp2_series, SCHEME_LABEL,
             "ssim", "SSIM",
             f"Exp 2 — tile utility scheme ablation ({tag})",
             OUT_DIR / f"exp2_{wm}_bar_ssim.png")
    per_scene_grid(exp2_cell, exp2_series, SCHEME_LABEL,
                   "psnr", "PSNR (dB)",
                   f"Exp 2 per-scene — PSNR vs budget % by scheme  ({tag})",
                   OUT_DIR / f"exp2_{wm}_lines_psnr.png")
    per_scene_grid(exp2_cell, exp2_series, SCHEME_LABEL,
                   "ssim", "SSIM",
                   f"Exp 2 per-scene — SSIM vs budget % by scheme  ({tag})",
                   OUT_DIR / f"exp2_{wm}_lines_ssim.png")


# ---------- Console summary table ----------

def print_table(cell: dict[tuple, dict], series_keys: list[str],
                series_label: dict[str, str], header: str) -> None:
    print(f"\n=== {header} — mean PSNR (dB) across {len(SCENES)} scenes ===")
    head = "budget% | " + " | ".join(f"{series_label[s]:>14}" for s in series_keys)
    print(head)
    print("-" * len(head))
    for b in BUDGETS:
        cells = []
        for s in series_keys:
            vals = [cell[(sc, s, b)]["psnr"] for sc in SCENES
                    if (sc, s, b) in cell]
            cells.append(f"{np.mean(vals):>14.2f}" if vals else f"{'—':>14}")
        print(f"{b:>6}% | " + " | ".join(cells))

print_table(exp1_cell, exp1_series, WEIGHT_MODE_LABEL, "Exp 1 weight_mode")
for wm, exp2_cell in exp2_cells.items():
    series = [s for s in SCHEME_ORDER if any(k[1] == s for k in exp2_cell)]
    print_table(exp2_cell, series, SCHEME_LABEL, f"Exp 2 scheme @ {wm}")
