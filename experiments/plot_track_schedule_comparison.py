#!/usr/bin/env python3
"""Compare PSNR/SSIM/VMAF vs elapsed_sec across (track_schedule, n_tracks) configs
(glob one or more summary.csv, grouped by the CSV's own track_schedule+n_tracks columns,
NOT just n_tracks -- plot_n_tracks_comparison.py pools schedule types together since it
only keys on n_tracks, which would silently conflate e.g. round_robin_n2 and greedy_n2).
One subplot per method, one line per (track_schedule, n_tracks) config.

Usage:
  python experiments/plot_track_schedule_comparison.py \
      --glob "output/.../{round_robin,greedy}_n*/user*/metrics/summary.csv" \
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

_CONFIG_COLOR = {
    "single_track": "#000000",
    "round_robin_n2": "#d62728",
    "round_robin_n4": "#ff7f0e",
    "greedy_n2": "#1f77b4",
    "greedy_n4": "#2ca02c",
}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _config_label(r: dict) -> str:
    # n_tracks=1 makes track_schedule a no-op (round_robin/greedy/single-stream are all the
    # same schedule by construction with one track) -- call it what it is, not "round_robin_n1".
    if int(r["n_tracks"]) == 1:
        return "single_track"
    # pre-2026-07-16 summary.csv files predate --track-schedule and have no such column --
    # round_robin (via partition_into_tracks) was the only code path back then, so that's
    # the correct implicit value, not a guess.
    sched = r.get("track_schedule") or "round_robin"
    return f"{sched}_n{r['n_tracks']}"


def plot_metric(pooled: dict, metric: str, methods: list[str], configs: list[str],
                bw: str, out_path: Path):
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 4.8),
                              sharey=True, constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        cfg_style = _METHOD_CONFIG.get(method, {})
        for cfg_name in configs:
            key = (method, cfg_name)
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
            color = _CONFIG_COLOR.get(cfg_name, "gray")
            ax.plot(xs, means, color=color, linewidth=2.0, label=cfg_name, zorder=2)
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
    p.add_argument("--glob", required=True, nargs="+",
                   help="one or more globs matching summary.csv files (must have "
                        "track_schedule + n_tracks columns, or predate that column -- "
                        "see _config_label's round_robin fallback)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    paths = sorted({p for g in args.glob for p in glob.glob(g)})
    if not paths:
        raise SystemExit(f"no files matched: {args.glob}")
    print(f"pooling {len(paths)} file(s)")

    all_rows = []
    for path in paths:
        all_rows.extend(_read_rows(Path(path)))

    methods = [m for m in _METHOD_CONFIG if m in {r["method"] for r in all_rows}]
    configs = sorted({_config_label(r) for r in all_rows})
    bandwidths = sorted({r["bandwidth_mbps"] for r in all_rows}, key=float)
    print(f"configs found: {configs}")

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
                key = (r["method"], _config_label(r))
                pooled[key][round(float(r["elapsed_sec"]), 6)].append(float(r[metric]))
            out_path = out_dir / f"{metric}_vs_elapsed_bw{bw}.png"
            plot_metric(pooled, metric, methods, configs, bw, out_path)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
