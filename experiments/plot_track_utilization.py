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
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
