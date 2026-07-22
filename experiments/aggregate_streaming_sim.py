#!/usr/bin/env python3
"""Aggregate streaming_sim.py summary.csv across multiple traces, pooled by raw session
t_sec (2026-07-16, stateful rewrite -- supersedes the old elapsed_sec/"time since last
cadence reorder" pooling key, which only made sense under the retired per-window-reset
model). All EyeNavGS bicycle traces are genuinely ~60s recordings sharing a comparable
absolute session clock, so pooling directly by t_sec across traces is valid, not an
approximation.

Single entry point for every streaming-sim cross-trace aggregation this paper needs --
grouped by method (utility-model comparison, n_tracks fixed) or by n_tracks (scheduler
comparison, method fixed), rendered as a dense line+CI plot or a sparse grouped-bar plot,
plus an optional best/median/worst representative-trace pick. Kept as one file
(2026-07-20, consolidated from 3 short-lived one-off scripts -- plot_streaming_bmw_and_bar.py,
plot_streaming_ntracks_bar.py, plot_streaming_ntracks_line.py -- that each reimplemented
the same CI/pooling/snapping logic; see .claude/PLAN.md) instead of one script per
group-by x plot-mode combination.

Usage (method comparison, n_tracks fixed -- original/default behavior):
  python experiments/aggregate_streaming_sim.py \
      --glob "output/0717/streaming_sim_stateful/n_tracks4/user*/metrics/summary.csv" \
      --out-dir output/0717/streaming_sim_stateful/_bmw_and_bar/by_model \
      --paper-dir plotting/paper/streaming_sim/by_model

Usage (scheduler comparison, method fixed -- glob needs a literal '{}' for n_tracks):
  python experiments/aggregate_streaming_sim.py \
      --glob "output/0717/streaming_sim_stateful/n_tracks{}/user*/metrics/summary.csv" \
      --group-by n_tracks --fix-method ml --plot-mode line \
      --out-dir output/0717/streaming_sim_stateful/_bmw_and_bar/by_ntracks \
      --paper-dir plotting/paper/streaming_sim/by_ntracks

Usage (sparse grouped-bar instead of dense line, either group-by):
  ... --plot-mode bar --bar-time-marks 10 20 30 40 50 60

Usage (best/median/worst trace pick, independent of plot-mode/group-by):
  ... --bmw --bmw-method ml --bmw-metric vmaf
"""
from __future__ import annotations

import argparse
import csv
import glob
import shutil
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

# Matches plotting/paper_plot_metrics.ipynb cell 2's color_palette/errorbar style exactly,
# so every streaming-sim figure shares the same look as the rest of the paper.
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
_ERR_LW, _ERR_CAPSIZE, _ERR_CAPTHICK = 1.5, 4, 1.5

NTRACKS_ORDER = [1, 2, 3, 4]
# Ordinal track-count palette, style intent matches plotting/paper/plot_format_ref/
# streaming/*.png's n_tracks legend (No Tile/2/4/16 -> blue/orange/green/red). 5-8 (2026-07-20
# optimal-#tracks sweep) extend the cycle with tab10's remaining colors.
_NTRACKS_CONFIG = {
    1: {"label": "n=1", "color": "#1f77b4"},
    2: {"label": "n=2", "color": "#ff7f0e"},
    3: {"label": "n=3", "color": "#2ca02c"},
    4: {"label": "n=4", "color": "#d62728"},
    5: {"label": "n=5", "color": "#9467bd"},
    6: {"label": "n=6", "color": "#8c564b"},
    7: {"label": "n=7", "color": "#e377c2"},
    8: {"label": "n=8", "color": "#7f7f7f"},
}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _ci95(vals: np.ndarray) -> float:
    return 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0


def _load_rows(args) -> list[dict]:
    """Group-by=method: single glob, all methods, n_tracks whatever the dir already is.
    Group-by=n_tracks: glob must contain a literal '{}', formatted with each of
    NTRACKS_ORDER, rows filtered to --fix-method, and n_tracks column cross-checked."""
    all_rows = []
    if args.group_by == "method":
        paths = sorted(glob.glob(args.glob))
        if not paths:
            raise SystemExit(f"no files matched: {args.glob}")
        print(f"pooling {len(paths)} trace(s)")
        for path in paths:
            all_rows.extend(_read_rows(Path(path)))
    else:  # n_tracks
        if "{}" not in args.glob:
            raise SystemExit("--group-by n_tracks requires a literal '{}' in --glob")
        if not args.fix_method:
            raise SystemExit("--group-by n_tracks requires --fix-method")
        for nt in args.n_tracks_values:
            pattern = args.glob.format(nt)
            paths = sorted(glob.glob(pattern))
            if not paths:
                raise SystemExit(f"no files matched: {pattern}")
            n_rows = 0
            for path in paths:
                rows = [r for r in _read_rows(Path(path)) if r["method"] == args.fix_method]
                # summary.csv's n_tracks column is a CSV string (DictReader), compared
                # against str(nt) deliberately -- not a type-mismatch accident.
                bad_nt = {r["n_tracks"] for r in rows} - {str(nt)}
                if bad_nt:
                    raise SystemExit(f"{path}: expected n_tracks=={nt} only, found {bad_nt}")
                all_rows.extend(rows)
                n_rows += len(rows)
            print(f"n_tracks={nt}: {len(paths)} traces, {n_rows} rows (method={args.fix_method})")
    return all_rows


def _group_vals_and_cfg(args, all_rows: list[dict]):
    if args.group_by == "method":
        vals = [m for m in _METHOD_CONFIG if m in {r["method"] for r in all_rows}]
        return vals, _METHOD_CONFIG, "method"
    return args.n_tracks_values, _NTRACKS_CONFIG, "n_tracks"


def plot_line(pooled: dict, metric: str, bandwidths: list[str], group_vals: list,
              group_cfg: dict, out_path: Path):
    fig, axes = plt.subplots(1, len(bandwidths), figsize=(5.5 * len(bandwidths), 4.8),
                              sharey=True, constrained_layout=True)
    if len(bandwidths) == 1:
        axes = [axes]

    for ax, bw in zip(axes, bandwidths):
        for gv in group_vals:
            cfg = group_cfg.get(gv, {})
            key = (bw, gv)
            if key not in pooled:
                continue
            t_to_vals = pooled[key]
            xs = sorted(t_to_vals)
            means, los, his = [], [], []
            for x in xs:
                vals = np.asarray(t_to_vals[x])
                if metric == "psnr":
                    vals = np.clip(vals, None, PSNR_SATURATION_DB)
                m = vals.mean()
                ci = _ci95(vals)
                means.append(m)
                los.append(m - ci)
                his.append(m + ci)
            ax.plot(xs, means, color=cfg.get("color"), linewidth=2.0,
                    label=cfg.get("label", str(gv)), zorder=2)
            ax.fill_between(xs, los, his, color=cfg.get("color"), alpha=0.2, linewidth=0, zorder=1)
        ax.set_title(f"{bw} Mbps", fontsize=17)
        ax.set_xlabel("Time (sec)", fontsize=15)
        ax.tick_params(labelsize=13, direction="out", which="both", top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
    axes[0].set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=17)
    axes[0].legend(loc="best", framealpha=0.9, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
    plt.close(fig)


def _pool_at_marks(all_rows: list[dict], metric: str, group_key: str,
                    time_marks: list[int]) -> dict:
    """pooled[bw][group_val][mark_sec] = [values across traces], snapped to the nearest
    actual t_sec per mark. Shared by plot_bar and plot_line_marks -- same sparse-sampling
    semantics, different rendering."""
    all_t = sorted({round(float(r["t_sec"]), 6) for r in all_rows})
    mark_to_t = {mark: min(all_t, key=lambda t: abs(t - mark)) for mark in time_marks}
    pooled: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_rows:
        if not r.get(metric):
            continue
        t = round(float(r["t_sec"]), 6)
        gv = r[group_key] if group_key == "method" else int(r[group_key])
        for mark, snapped_t in mark_to_t.items():
            if t == snapped_t:
                pooled[r["bandwidth_mbps"]][gv][mark].append(float(r[metric]))
    return pooled


def plot_line_marks(all_rows: list[dict], metric: str, bandwidths: list[str],
                     group_vals: list, group_cfg: dict, group_key: str,
                     time_marks: list[int], out_path: Path):
    """Sparse line plot at time_marks (default 6: 10/20/.../60s) with real 95% CI error
    bars (ax.errorbar) instead of plot_line's dense-per-frame shaded-band CI, and instead
    of plot_bar's grouped bars -- same pooling as plot_bar, connected-line rendering."""
    pooled = _pool_at_marks(all_rows, metric, group_key, time_marks)

    fig, axes = plt.subplots(1, len(bandwidths), figsize=(5.5 * len(bandwidths), 4.8),
                              sharey=True, constrained_layout=True)
    if len(bandwidths) == 1:
        axes = [axes]

    for ax, bw in zip(axes, bandwidths):
        for gv in group_vals:
            cfg = group_cfg.get(gv, {})
            means, cis = [], []
            for mark in time_marks:
                vals = np.asarray(pooled[bw][gv][mark])
                if metric == "psnr":
                    vals = np.clip(vals, None, PSNR_SATURATION_DB)
                means.append(vals.mean() if len(vals) else np.nan)
                cis.append(_ci95(vals) if len(vals) else 0.0)
            ax.errorbar(time_marks, means, yerr=cis, marker="o", capsize=4, linewidth=2.0,
                        color=cfg.get("color"), label=cfg.get("label", str(gv)), zorder=2)
        ax.set_title(f"{bw} Mbps", fontsize=17)
        ax.set_xlabel("Time (sec)", fontsize=15)
        ax.set_xticks(time_marks)
        ax.tick_params(labelsize=13, direction="out", which="both", top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
    axes[0].set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=17)
    axes[0].legend(loc="best", framealpha=0.9, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
    plt.close(fig)


def plot_bar(all_rows: list[dict], metric: str, bandwidths: list[str], group_vals: list,
             group_cfg: dict, group_key: str, time_marks: list[int], out_path: Path):
    pooled = _pool_at_marks(all_rows, metric, group_key, time_marks)

    n_bars = len(group_vals)
    width = 0.8 / n_bars
    x = np.arange(len(time_marks))

    fig, axes = plt.subplots(1, len(bandwidths), figsize=(6.0 * len(bandwidths), 4.8),
                              sharey=True, constrained_layout=True)
    if len(bandwidths) == 1:
        axes = [axes]

    for ax, bw in zip(axes, bandwidths):
        for i, gv in enumerate(group_vals):
            cfg = group_cfg.get(gv, {})
            means, cis = [], []
            for mark in time_marks:
                vals = np.asarray(pooled[bw][gv][mark])
                if metric == "psnr":
                    vals = np.clip(vals, None, PSNR_SATURATION_DB)
                means.append(vals.mean() if len(vals) else np.nan)
                cis.append(_ci95(vals) if len(vals) else 0.0)
            offset = (i - (n_bars - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=cis, capsize=3,
                   color=cfg.get("color"), label=cfg.get("label", str(gv)), zorder=2)
        ax.set_title(f"{bw} Mbps", fontsize=17)
        ax.set_xlabel("Time (sec)", fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in time_marks])
        ax.tick_params(labelsize=13, direction="out", which="both", top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
    axes[0].set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=17)
    axes[0].legend(loc="best", framealpha=0.9, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
    plt.close(fig)


def plot_scalar(all_rows: list[dict], metric: str, bandwidths: list[str], group_vals: list,
                 out_path: Path):
    """Whole-session aggregate metric vs. #tracks -- one point per (bandwidth, n_tracks),
    pooling every trace and every t_sec together (not faceted by time like plot_line/
    plot_bar). Answers "does adding more tracks keep helping, or does it plateau/regress"
    at a glance, instead of eyeballing a per-time-mark bar chart."""
    # pooled[bw][group_val] = [values across all traces, all t_sec]
    pooled: dict = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        if not r.get(metric):
            continue
        vals = pooled[r["bandwidth_mbps"]][int(r["n_tracks"])]
        v = float(r[metric])
        if metric == "psnr":
            v = min(v, PSNR_SATURATION_DB)
        vals.append(v)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    for i, bw in enumerate(bandwidths):
        xs, means, cis = [], [], []
        for gv in group_vals:
            vals = np.asarray(pooled[bw].get(gv, []))
            if len(vals) == 0:
                continue
            xs.append(gv)
            means.append(vals.mean())
            cis.append(_ci95(vals))
        color = _PALETTE[i % len(_PALETTE)]
        ax.errorbar(xs, means, yerr=cis, marker="o", capsize=_ERR_CAPSIZE,
                     elinewidth=_ERR_LW, capthick=_ERR_CAPTHICK, linewidth=2.0,
                     color=color, label=f"{bw} Mbps", zorder=2)
    ax.set_xlabel("Number of tracks", fontsize=15)
    ax.set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=17)
    ax.set_xticks(group_vals)
    ax.tick_params(labelsize=13, direction="out", which="both", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.legend(loc="best", framealpha=0.9, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
    plt.close(fig)


def pick_bmw(all_rows: list[dict], args, bandwidths: list[str], out_dir: Path,
             paper_dir: Path | None):
    """Best/Median/Worst trace per bandwidth, ranked by --bmw-method's whole-session mean
    --bmw-metric per trace (outline: "Representative Traces that gives Our Scheduler using
    ML model with Best-Median-Worst average VMAF result")."""
    by_trace: dict = defaultdict(list)
    for r in all_rows:
        by_trace[r["_trace_id"]].append(r)

    csv_rows = []
    for bw in bandwidths:
        per_trace_mean = {}
        for trace_id, rows in by_trace.items():
            vals = [float(r[args.bmw_metric]) for r in rows
                    if r["method"] == args.bmw_method and r["bandwidth_mbps"] == bw
                    and r.get(args.bmw_metric)]
            if vals:
                per_trace_mean[trace_id] = float(np.mean(vals))
        if not per_trace_mean:
            continue
        ranked = sorted(per_trace_mean.items(), key=lambda kv: kv[1])
        worst, best = ranked[0], ranked[-1]
        mid = len(ranked) // 2
        median = ranked[mid] if len(ranked) % 2 == 1 else ranked[mid - 1]
        for role, (trace_id, mean_val) in (("best", best), ("median", median), ("worst", worst)):
            csv_rows.append({"bandwidth_mbps": bw, "role": role, "trace": trace_id,
                              f"mean_{args.bmw_metric}_{args.bmw_method}_method": mean_val})
        print(f"bw={bw}: n_traces={len(ranked)} "
              f"best={best[0]}({best[1]:.2f}) median={median[0]}({median[1]:.2f}) "
              f"worst={worst[0]}({worst[1]:.2f})")

    out_path = out_dir / "bmw_traces.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {out_path}")
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, paper_dir / "bmw_traces.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True,
                    help="glob matching summary.csv files. For --group-by n_tracks, must "
                         "contain a literal '{}' where the n_tracks value goes.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--paper-dir", default=None,
                    help="Optional second copy of every output, for the paper (mirrors "
                         "--out-dir). Omit to write --out-dir only.")
    p.add_argument("--group-by", choices=["method", "n_tracks"], default="method")
    p.add_argument("--fix-method", default=None,
                    help="Required when --group-by n_tracks: which method's rows to load.")
    p.add_argument("--n-tracks-values", type=int, nargs="+", default=NTRACKS_ORDER,
                    help="Only used when --group-by n_tracks: which n_tracks values to "
                         "expect/glob for. Defaults to 1-4 (today's full sweep) -- set "
                         "explicitly for a partial sweep (e.g. '1 2') so a missing value "
                         "doesn't hard-fail the glob.")
    p.add_argument("--plot-mode", choices=["line", "bar", "scalar", "line_marks"],
                    default="line",
                    help="line: dense per-frame line + shaded-band CI. bar: grouped bars at "
                         "--bar-time-marks. line_marks: connected line + errorbar 95%% CI at "
                         "--bar-time-marks (sparse line, real error bars, no shading). "
                         "scalar: whole-session aggregate metric vs. group value (x-axis is "
                         "n_tracks/method, not time) -- for --group-by n_tracks, answers "
                         "where more tracks stops helping.")
    p.add_argument("--bar-time-marks", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60],
                    help="Used by --plot-mode bar and line_marks.")
    p.add_argument("--bmw", action="store_true",
                    help="Also write bmw_traces.csv (best/median/worst trace per "
                         "bandwidth), independent of --plot-mode/--group-by.")
    p.add_argument("--bmw-method", default="ml")
    p.add_argument("--bmw-metric", default="vmaf")
    p.add_argument("--extra-glob", default=None,
                    help="Second sweep tree to pull additional bandwidth line(s) from, e.g. "
                         "a different-tier sweep with a fill-in n_tracks value the primary "
                         "--glob's tree doesn't have. Same '{}' rule as --glob under "
                         "--group-by n_tracks. Rows are filtered to --extra-bandwidths before "
                         "merging -- NOT a blind union, since two independently-run sweeps can "
                         "legitimately share a bandwidth value (e.g. both include 120Mbps) and "
                         "merging that would silently double-count one bandwidth's trace pool.")
    p.add_argument("--extra-bandwidths", nargs="+", default=None,
                    help="Which bandwidth_mbps values to keep from --extra-glob. Required with "
                         "--extra-glob.")
    args = p.parse_args()
    if args.extra_glob and not args.extra_bandwidths:
        raise SystemExit("--extra-glob requires --extra-bandwidths")

    all_rows = _load_rows(args)
    if args.extra_glob:
        if args.group_by != "n_tracks":
            raise SystemExit("--extra-glob only supported with --group-by n_tracks")
        keep_bw = set(args.extra_bandwidths)
        extra_rows = []
        for nt in args.n_tracks_values:
            for path in sorted(glob.glob(args.extra_glob.format(nt))):
                extra_rows.extend(
                    r for r in _read_rows(Path(path))
                    if r["method"] == args.fix_method and r["bandwidth_mbps"] in keep_bw
                       and r["n_tracks"] == str(nt))
        if not extra_rows:
            raise SystemExit(f"--extra-glob matched but no rows survived the "
                              f"bandwidth={keep_bw} filter -- check --extra-bandwidths")
        print(f"extra: {len(extra_rows)} rows from --extra-glob "
              f"(bandwidths={sorted(keep_bw, key=float)})")
        all_rows.extend(extra_rows)
    for r in all_rows:
        r.setdefault("_trace_id", None)  # filled in below only if --bmw needs it
    bandwidths = sorted({r["bandwidth_mbps"] for r in all_rows}, key=float)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path(args.paper_dir) if args.paper_dir else None
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)

    if args.bmw:
        # Re-load with trace ids attached (cheap; keeps _load_rows agnostic of bmw).
        if args.group_by != "method":
            raise SystemExit("--bmw expects a single n_tracks value: use --group-by method")
        paths = sorted(glob.glob(args.glob))
        bmw_rows = []
        for path in paths:
            trace_id = Path(path).parent.parent.name  # .../userN/metrics/summary.csv -> userN
            for r in _read_rows(Path(path)):
                r["_trace_id"] = trace_id
                bmw_rows.append(r)
        pick_bmw(bmw_rows, args, bandwidths, out_dir, paper_dir)

    group_vals, group_cfg, group_key = _group_vals_and_cfg(args, all_rows)

    for metric in ("psnr", "ssim", "vmaf"):
        if not any(r.get(metric) for r in all_rows):
            print(f"skip {metric}: no data")
            continue
        suffix = {"line": "line_vs_time", "bar": "grouped_bar_by_time",
                  "line_marks": "line_marks_vs_time", "scalar": "vs_ntracks"}[args.plot_mode]
        by = "method" if args.group_by == "method" else "ntracks"
        out_path = out_dir / f"{metric}_{suffix}_by_{by}.png"
        if args.plot_mode == "scalar":
            plot_scalar(all_rows, metric, bandwidths, group_vals, out_path)
        elif args.plot_mode == "line_marks":
            plot_line_marks(all_rows, metric, bandwidths, group_vals, group_cfg, group_key,
                             args.bar_time_marks, out_path)
        elif args.plot_mode == "line":
            # pooled[(bandwidth, group_val)][t_sec] = [values]. round(): matches render
            # marks' fixed step exactly, but floats accumulate drift across many
            # summary.csv files -- rounding avoids fragmenting one nominal tick into
            # several near-duplicate x-positions (.claude/PLAN.md "Cross-trace
            # aggregator bug").
            pooled: dict = defaultdict(lambda: defaultdict(list))
            for r in all_rows:
                if not r.get(metric):
                    continue
                gv = r[group_key] if group_key == "method" else int(r[group_key])
                pooled[(r["bandwidth_mbps"], gv)][round(float(r["t_sec"]), 6)].append(float(r[metric]))
            plot_line(pooled, metric, bandwidths, group_vals, group_cfg, out_path)
        else:
            plot_bar(all_rows, metric, bandwidths, group_vals, group_cfg, group_key,
                     args.bar_time_marks, out_path)
        print(f"wrote {out_path}")
        if paper_dir is not None:
            for ext in (".png", ".eps"):
                shutil.copy(out_path.with_suffix(ext), paper_dir / out_path.with_suffix(ext).name)


if __name__ == "__main__":
    main()
