#!/usr/bin/env python3
"""Overlay default vmaf_v0.6.1 (summary.csv) vs vmaf_4k_v0.6.1 (summary_vmaf4k.csv) on the
same axes, per bandwidth: 2 rows (profile) x 3 cols (method), one line per n_tracks, shared
y-axis so the absolute-score shift and the (un)changed ranking are both visible directly.

Usage:
  python experiments/plot_vmaf_profile_comparison.py \
      --default-glob "output/0715/streaming_sim_multitrack/n_tracks*/user*/metrics/summary.csv" \
      --vmaf4k-glob "output/0715/streaming_sim_multitrack/n_tracks*/user*/metrics/summary_vmaf4k.csv" \
      --out-dir output/0715/streaming_sim_multitrack/agg_plots
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

from plot_streaming_sim import _METHOD_CONFIG, csfont, DPI

plt.rc("text", usetex=False)
plt.rc("font", **csfont)

_N_TRACKS_STYLE = {1: {"linestyle": "dotted"}, 4: {"linestyle": "dashed"}, 8: {"linestyle": "solid"}}
_PROFILE_LABEL = {"default": "vmaf_v0.6.1 (default, 1080p-calibrated)",
                   "vmaf4k": "vmaf_4k_v0.6.1 (4K-calibrated)"}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--default-glob", required=True)
    p.add_argument("--vmaf4k-glob", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    default_paths = sorted(glob.glob(args.default_glob))
    vmaf4k_paths = sorted(glob.glob(args.vmaf4k_glob))
    if not default_paths or not vmaf4k_paths:
        raise SystemExit(f"no files matched (default={len(default_paths)}, vmaf4k={len(vmaf4k_paths)})")
    print(f"pooling {len(default_paths)} default + {len(vmaf4k_paths)} vmaf4k file(s)")

    rows_by_profile = {"default": [], "vmaf4k": []}
    for path in default_paths:
        rows_by_profile["default"].extend(_read_rows(Path(path)))
    for path in vmaf4k_paths:
        rows_by_profile["vmaf4k"].extend(_read_rows(Path(path)))

    all_rows = rows_by_profile["default"] + rows_by_profile["vmaf4k"]
    methods = [m for m in _METHOD_CONFIG if m in {r["method"] for r in all_rows}]
    n_tracks_vals = sorted({int(r["n_tracks"]) for r in all_rows})
    bandwidths = sorted({r["bandwidth_mbps"] for r in all_rows}, key=float)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bw in bandwidths:
        fig, axes = plt.subplots(2, len(methods), figsize=(5.5 * len(methods), 9.0),
                                  sharey=True, sharex=True, constrained_layout=True)
        for row, profile in enumerate(("default", "vmaf4k")):
            rows_bw = [r for r in rows_by_profile[profile] if r["bandwidth_mbps"] == bw]
            pooled: dict = defaultdict(lambda: defaultdict(list))
            for r in rows_bw:
                if not r.get("vmaf"):
                    continue
                key = (r["method"], int(r["n_tracks"]))
                pooled[key][round(float(r["elapsed_sec"]), 6)].append(float(r["vmaf"]))

            for col, method in enumerate(methods):
                ax = axes[row][col]
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
                        m = vals.mean()
                        ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                        means.append(m)
                        los.append(m - ci)
                        his.append(m + ci)
                    style = _N_TRACKS_STYLE.get(n, {})
                    ax.plot(xs, means, color=cfg.get("color"), linewidth=2.0,
                            label=f"n_tracks={n}", zorder=2, **style)
                    ax.fill_between(xs, los, his, color=cfg.get("color"), alpha=0.15, linewidth=0, zorder=1)
                if row == 0:
                    ax.set_title(f"{cfg.get('label', method)}", fontsize=15)
                if col == 0:
                    ax.set_ylabel(f"VMAF\n({_PROFILE_LABEL[profile]})", fontsize=12)
                if row == 1:
                    ax.set_xlabel("Time since cadence tick (sec)", fontsize=14)
                ax.tick_params(labelsize=11, direction="out", which="both", top=False, right=False)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                ax.legend(loc="best", framealpha=0.9, fontsize=10)

        fig.suptitle(f"VMAF profile comparison @ {bw} Mbps", fontsize=17)
        out_path = out_dir / f"vmaf_profile_compare_bw{bw}.png"
        fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
        fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
