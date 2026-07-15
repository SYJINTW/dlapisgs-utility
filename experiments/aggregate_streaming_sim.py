#!/usr/bin/env python3
"""Aggregate streaming_sim.py summary.csv across multiple traces, pooled by
elapsed_sec (time since last cadence reorder) -- NOT raw t_sec, since traces
are independent sessions with no shared clock. One point per (bandwidth,
method, elapsed_sec) = mean +/- 95% CI over all traces x cadence windows
landing on that elapsed_sec tick.

Usage:
  python experiments/aggregate_streaming_sim.py \
      --glob "output/0714/streaming_sim/all24_bw600/*/metrics/summary.csv" \
      --out-dir output/0714/streaming_sim/all24_bw600/agg
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


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_metric(pooled: dict, metric: str, bandwidths: list[str], methods: list[str],
                out_path: Path):
    fig, axes = plt.subplots(1, len(bandwidths), figsize=(5.5 * len(bandwidths), 4.8),
                              sharey=True, constrained_layout=True)
    if len(bandwidths) == 1:
        axes = [axes]

    for ax, bw in zip(axes, bandwidths):
        for method in methods:
            cfg = _METHOD_CONFIG.get(method, {})
            key = (bw, method)
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
            ax.plot(xs, means, color=cfg.get("color"), linewidth=2.0,
                    label=cfg.get("label", method), zorder=2)
            ax.fill_between(xs, los, his, color=cfg.get("color"), alpha=0.2, linewidth=0, zorder=1)
        ax.set_title(f"{bw} Mbps", fontsize=17)
        ax.set_xlabel("Time since cadence tick (sec)", fontsize=15)
        ax.tick_params(labelsize=13, direction="out", which="both", top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
    axes[0].set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=17)
    axes[0].legend(loc="best", framealpha=0.9, fontsize=13)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="glob matching summary.csv files across traces")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matched: {args.glob}")
    print(f"pooling {len(paths)} trace(s)")

    all_rows = []
    for path in paths:
        all_rows.extend(_read_rows(Path(path)))

    bandwidths = sorted({r["bandwidth_mbps"] for r in all_rows}, key=float)
    methods = [m for m in _METHOD_CONFIG if m in {r["method"] for r in all_rows}]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("psnr", "ssim", "vmaf"):
        if not any(r.get(metric) for r in all_rows):
            continue
        # pooled[(bandwidth, method)][elapsed_sec] = [values across traces/windows]
        pooled: dict = defaultdict(lambda: defaultdict(list))
        for r in all_rows:
            if not r.get(metric):
                continue
            key = (r["bandwidth_mbps"], r["method"])
            pooled[key][float(r["elapsed_sec"])].append(float(r[metric]))
        out_path = out_dir / f"{metric}_vs_elapsed_pooled.png"
        plot_metric(pooled, metric, bandwidths, methods, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
