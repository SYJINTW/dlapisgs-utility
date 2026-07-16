#!/usr/bin/env python3
"""Compare PSNR/SSIM vs elapsed_sec across --n-tracks values (glob one or more
summary.csv, grouped by their own n_tracks column). One subplot per method,
one line per n_tracks value.

Usage:
  python experiments/plot_n_tracks_comparison.py \
      --glob "output/.../n_tracks*/user*/metrics/summary.csv" \
      --out-dir output/.../plots
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

_N_TRACKS_STYLE = {1: {"linestyle": "dotted"}, 4: {"linestyle": "dashed"}, 8: {"linestyle": "solid"}}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_metric(pooled: dict, metric: str, methods: list[str], n_tracks_vals: list[int],
                bw: str, out_path: Path):
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 4.8),
                              sharey=True, constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        cfg = _METHOD_CONFIG.get(method, {})
        for n in n_tracks_vals:
            key = (method, n)
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
            style = _N_TRACKS_STYLE.get(n, {})
            ax.plot(xs, means, color=cfg.get("color"), linewidth=2.0,
                    label=f"n_tracks={n}", zorder=2, **style)
            ax.fill_between(xs, los, his, color=cfg.get("color"), alpha=0.15, linewidth=0, zorder=1)
        ax.set_title(f"{cfg.get('label', method)} ({bw} Mbps)", fontsize=15)
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
    p.add_argument("--glob", required=True, help="glob matching summary.csv files (must have an n_tracks column)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matched: {args.glob}")
    print(f"pooling {len(paths)} file(s)")

    all_rows = []
    for path in paths:
        all_rows.extend(_read_rows(Path(path)))

    methods = [m for m in _METHOD_CONFIG if m in {r["method"] for r in all_rows}]
    n_tracks_vals = sorted({int(r["n_tracks"]) for r in all_rows})
    bandwidths = sorted({r["bandwidth_mbps"] for r in all_rows}, key=float)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bw in bandwidths:
        rows_bw = [r for r in all_rows if r["bandwidth_mbps"] == bw]
        for metric in ("psnr", "ssim", "vmaf"):
            if not any(r.get(metric) for r in rows_bw):
                continue
            pooled: dict = defaultdict(lambda: defaultdict(list))
            for r in rows_bw:
                if not r.get(metric):
                    continue
                key = (r["method"], int(r["n_tracks"]))
                pooled[key][round(float(r["elapsed_sec"]), 6)].append(float(r[metric]))
            out_path = out_dir / f"{metric}_vs_elapsed_bw{bw}.png"
            plot_metric(pooled, metric, methods, n_tracks_vals, bw, out_path)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
