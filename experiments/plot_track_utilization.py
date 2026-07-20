#!/usr/bin/env python3
"""Plot per-track bandwidth utilization from a streaming_sim.py run's schedule/*.json dump.

Two diagnostics per (method, bandwidth): a Gantt chart (one row per track, one bar per
tile it sent) and an achieved-throughput-vs-nominal-bandwidth curve (bytes delivered per
time bucket, summed across tracks, vs the nominal bandwidth). Sanity-checks that
build_session_schedule() (streaming_sim.py) isn't pathologically stalling a track while
unclaimed tiles remain -- exact schedule data, not sampled/reconstructed from the render
loop (2026-07-16, user request accompanying the stateful streaming_sim.py rewrite).

Usage:
  python experiments/plot_track_utilization.py \\
      --schedule-json <output_root>/schedule/vd_lod_bw300.0mbps.json \\
      --out-dir <output_root>/plots/utilization
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300


def plot_gantt(data: dict, out_path: Path, title: str) -> None:
    entries = data["entries"]
    n_tracks = data["n_tracks"]
    session_dur = data["session_duration_sec"]
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 0.6 * n_tracks + 1.5))
    for e in entries:
        ax.barh(e["track"], e["end_sec"] - e["start_sec"], left=e["start_sec"],
                height=0.6, color=cmap(e["track"] % 10), edgecolor="none")
    ax.set_yticks(range(n_tracks))
    ax.set_yticklabels([f"track {k}" for k in range(n_tracks)])
    ax.set_xlabel("Time (sec)")
    ax.set_xlim(0, session_dur)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI)
    plt.close(fig)


def plot_gantt_comparison(configs: list[tuple[str, dict]], out_path: Path,
                           suptitle: str) -> None:
    """Stack one gantt subplot per config (e.g. n_tracks=1/2/3/4 for the same trace/method/
    bandwidth) in one figure, sharing the x-axis -- so a straggler tile in one config lines
    up visually against the others. Bars colored by tile_idx (not track) so a single big
    tile is visually traceable across configs; bar length = real send duration from the
    schedule JSON, not synthetic. Mirrors the toy gantt_best_in_family_comparison.png
    layout (quickplots/0716_track_scheduling_example_gantt/), but from real streaming_sim.py
    schedule/*.json data instead of a hand-built example."""
    cmap = plt.get_cmap("tab20")
    session_dur = max(d["session_duration_sec"] for _, d in configs)

    fig, axes = plt.subplots(len(configs), 1, figsize=(11, sum(0.5 * d["n_tracks"] + 1.3
                                                                 for _, d in configs)),
                              constrained_layout=True)
    if len(configs) == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, configs):
        n_tracks = data["n_tracks"]
        makespan = max((e["end_sec"] for e in data["entries"]), default=0.0)
        for e in data["entries"]:
            ax.barh(e["track"], e["end_sec"] - e["start_sec"], left=e["start_sec"],
                     height=0.6, color=cmap(e["tile_idx"] % 20), edgecolor="none")
        ax.set_yticks(range(n_tracks))
        ax.set_yticklabels([f"track {k}" for k in range(n_tracks)])
        ax.set_xlim(0, session_dur)
        ax.set_title(f"{label} (N={n_tracks} tracks) | makespan={makespan:.2f}s | "
                      f"{len(data['entries'])} tiles sent", fontsize=12)
        ax.tick_params(labelsize=10)
    axes[-1].set_xlabel("Time (sec)", fontsize=12)
    fig.suptitle(suptitle, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=DPI)
    plt.close(fig)


def plot_throughput(data: dict, out_path: Path, title: str, bucket_sec: float = 0.5) -> None:
    """Achieved aggregate throughput (bytes/bucket summed across all tracks, using each
    claimed tile's own send rate = bytes/duration) vs the nominal total bandwidth. Flat at
    ~nominal until full coverage, then drops to 0 -- any other shape (dips, spikes) while
    unclaimed tiles remain indicates a scheduling pathology."""
    entries = data["entries"]
    session_dur = data["session_duration_sec"]
    bandwidth_mbps = data["bandwidth_mbps"]
    nominal_mbps = bandwidth_mbps

    n_buckets = int(np.ceil(session_dur / bucket_sec)) + 1
    bucket_bytes = np.zeros(n_buckets)
    for e in entries:
        st, en, nb = e["start_sec"], e["end_sec"], e["bytes"]
        dur = en - st
        if dur <= 0:
            continue
        rate = nb / dur  # bytes/sec this tile was sent at
        b0, b1 = int(st // bucket_sec), int(en // bucket_sec)
        for b in range(b0, min(b1, n_buckets - 1) + 1):
            lo, hi = max(st, b * bucket_sec), min(en, (b + 1) * bucket_sec)
            if hi > lo:
                bucket_bytes[b] += rate * (hi - lo)

    xs = (np.arange(n_buckets) + 0.5) * bucket_sec
    achieved_mbps = bucket_bytes / bucket_sec * 8 / 1e6

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, achieved_mbps, label="achieved throughput", color="#1f77b4", linewidth=1.5)
    ax.axhline(nominal_mbps, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"nominal bandwidth ({bandwidth_mbps:g} Mbps)")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_xlim(0, session_dur)
    ax.set_ylim(0, None)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule-json", required=True, nargs="+",
                    help="One or more schedule/{method}_bw{bw}mbps.json paths.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bucket-sec", type=float, default=0.5)
    p.add_argument("--compare-out", default=None,
                    help="If set, also write one combined gantt (one stacked subplot per "
                         "--schedule-json, in the order given) to this path -- e.g. "
                         "comparing n_tracks 1/2/3/4 for the same trace/method/bandwidth.")
    p.add_argument("--compare-title", default="",
                    help="Suptitle for --compare-out (e.g. trace/method/bandwidth info).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_out:
        configs = []
        for path_str in args.schedule_json:
            path = Path(path_str)
            data = json.loads(path.read_text())
            label = path.parent.parent.parent.name  # .../n_tracksN/{trace}/schedule/foo.json -> n_tracksN
            configs.append((label, data))
        plot_gantt_comparison(configs, Path(args.compare_out), args.compare_title)
        print(f"wrote {args.compare_out}")

    for path_str in args.schedule_json:
        path = Path(path_str)
        data = json.loads(path.read_text())
        # schedule/ dir's grandparent (the run's output_root name) disambiguates otherwise-
        # identical stems (e.g. "vd_lod_bw300.0mbps") when plotting configs from different runs
        # in one --out-dir call.
        stem = f"{path.parent.parent.name}_{path.stem}"
        title = f"{stem} ({data['n_tracks']} tracks)"
        plot_gantt(data, out_dir / f"{stem}_gantt.png", title)
        plot_throughput(data, out_dir / f"{stem}_throughput.png", title, args.bucket_sec)
        print(f"wrote {out_dir / f'{stem}_gantt.png'} and {stem}_throughput.png")


if __name__ == "__main__":
    main()
