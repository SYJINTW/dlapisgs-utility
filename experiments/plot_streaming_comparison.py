#!/usr/bin/env python3
"""Generic streaming_sim cross-config comparison plot: PSNR/SSIM/VMAF vs elapsed_sec,
one subplot per method, one line per named panel. Panel identity is always an explicit
CLI label=glob pair -- never inferred from a CSV column -- so the same tool covers any
comparison axis (pruning on/off, track_schedule x n_tracks, a future drift-reorder
on/off, ...) without a dedicated script per axis. Supersedes plot_pruning_comparison.py,
plot_track_schedule_comparison.py, plot_n_tracks_comparison.py (all deleted 2026-07-16 --
same ~130-line skeleton copy-pasted 3x, only the config-labeling logic differed).

Usage:
  python experiments/plot_streaming_comparison.py \\
      --panel "no pruning=output/.../noprune/user*/metrics/summary.csv" \\
      --panel "50% per-tile prune=output/.../prune50/user*/metrics/summary.csv" \\
      --out-dir output/.../agg_plots

  # Track-schedule x n_tracks comparison (one glob per config, explicit > implicit):
  python experiments/plot_streaming_comparison.py \\
      --panel "single_track=output/.../n_tracks1/user*/metrics/summary.csv" \\
      --panel "round_robin_n4=output/.../round_robin_n4/user*/metrics/summary.csv" \\
      --panel "greedy_n4=output/.../greedy_n4/user*/metrics/summary.csv" \\
      --color "single_track=#000000" --color "greedy_n4=#2ca02c" \\
      --out-dir output/.../agg_plots
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_streaming_sim import _METHOD_CONFIG, _METRIC_YLABEL, PSNR_SATURATION_DB, csfont, DPI

try:
    plt.rc("text", usetex=True)
    plt.rc("font", **csfont)
    plt.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}"
    fig = plt.figure()
    fig.text(0.5, 0.5, r"test $x^2$")
    fig.canvas.draw()
    plt.close(fig)
except Exception:
    plt.rc("text", usetex=False)
    plt.rc("font", **csfont)

# Assigned to panels in --panel order unless overridden by --color; stable, colorblind-safe.
_DEFAULT_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]


def _tex_escape(label: str) -> str:
    # usetex renders legend/title text as raw LaTeX source -- "%" is a comment char there
    # and silently kills the whole render (matplotlib exits 0 with a blank/missing figure
    # unless you're watching stderr). Underscores are fine as-is (matplotlib's texmanager
    # tolerates them in text mode; confirmed empirically against committed track-schedule
    # figures using labels like "round_robin_n4").
    return label.replace("%", r"\%")


def _parse_kv(items: list[str]) -> dict:
    out = {}
    for item in items:
        label, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"expected LABEL=VALUE, got: {item!r}")
        out[label] = value
    return out


def _read_rows(path: Path, panel: str) -> list[dict]:
    with path.open(newline="") as f:
        return [dict(r, _panel=panel) for r in csv.DictReader(f)]


def plot_metric(pooled: dict, metric: str, methods: list[str], panels: list[str],
                 colors: dict, bw: str, out_path: Path):
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 4.8),
                              sharey=True, constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        cfg_style = _METHOD_CONFIG.get(method, {})
        for panel in panels:
            key = (method, panel)
            if key not in pooled:
                continue
            elapsed_to_vals = pooled[key]
            xs = sorted(elapsed_to_vals)
            means, los, his = [], [], []
            for x in xs:
                vals = np.asarray(elapsed_to_vals[x])
                if metric == "psnr":
                    vals = np.clip(vals, None, PSNR_SATURATION_DB)
                m = vals.mean()
                ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                means.append(m)
                los.append(m - ci)
                his.append(m + ci)
            color = colors[panel]
            ax.plot(xs, means, color=color, linewidth=2.0, label=_tex_escape(panel), zorder=2)
            ax.fill_between(xs, los, his, color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.set_title(f"{cfg_style.get('label', method)} ({bw} Mbps)", fontsize=15)
        ax.set_xlabel("Time since cadence tick (sec)", fontsize=14)
        ax.tick_params(labelsize=12, direction="out", which="both", top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.legend(loc="best", framealpha=0.9, fontsize=11)
    axes[0].set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=15)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True, action="append",
                    help="LABEL=GLOB, repeatable -- one glob per compared config")
    p.add_argument("--color", action="append", default=[],
                    help="LABEL=HEXCOLOR override, repeatable (default: stable palette "
                         "in --panel order)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "vmaf"])
    args = p.parse_args()

    panel_globs = _parse_kv(args.panel)
    panels = list(panel_globs)  # preserves --panel order
    color_overrides = _parse_kv(args.color)
    colors = {panel: color_overrides.get(panel, _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)])
              for i, panel in enumerate(panels)}

    all_rows = []
    for panel, pattern in panel_globs.items():
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"no files matched for panel {panel!r}: {pattern}")
        print(f"{panel}: pooling {len(paths)} file(s)")
        for path in paths:
            all_rows.extend(_read_rows(Path(path), panel))

    methods = [m for m in _METHOD_CONFIG if m in {r["method"] for r in all_rows}]
    bandwidths = sorted({r["bandwidth_mbps"] for r in all_rows}, key=float)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bw in bandwidths:
        rows_bw = [r for r in all_rows if r["bandwidth_mbps"] == bw]
        for metric in args.metrics:
            if not any(r.get(metric) for r in rows_bw):
                continue
            pooled: dict = defaultdict(lambda: defaultdict(list))
            for r in rows_bw:
                if not r.get(metric):
                    continue
                key = (r["method"], r["_panel"])
                pooled[key][round(float(r["elapsed_sec"]), 6)].append(float(r[metric]))
            out_path = out_dir / f"{metric}_vs_elapsed_bw{bw}.png"
            plot_metric(pooled, metric, methods, panels, colors, bw, out_path)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
