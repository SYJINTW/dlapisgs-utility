#!/usr/bin/env python3
"""Generic side-by-side comparison video (GT + N named configs) for one representative
streaming_sim trace -- a qualitative sanity check to go with the PSNR/SSIM/VMAF
comparison plots (plot_streaming_comparison.py). Reads the raw yuv420p sequences
FrameWriter already wrote during run_sweep (vmaf/gt.yuv + vmaf/bw_{bw}mbps/{method}/
distorted.yuv) -- no re-render needed. Uses ffmpeg xstack + drawtext, same tool the
pipeline already depends on. Supersedes make_track_schedule_video.py and
make_pruning_video.py (deleted 2026-07-16 -- identical script, only the hardcoded
CONFIGS list + xstack layout differed).

Usage:
  python experiments/make_streaming_comparison_video.py --trace user7 --bandwidth 600.0 \\
      --method ml \\
      --panel "no pruning=output/.../noprune/{trace}" \\
      --panel "50% per-tile prune=output/.../prune50/{trace}" \\
      --out output/.../video/user7_ml_600mbps.mp4

  # 5-config track-schedule video, wrapped 3-per-row like the original 6-panel figure:
  python experiments/make_streaming_comparison_video.py --trace user7 --bandwidth 600.0 \\
      --method ml --cols 3 \\
      --panel "single_track=output/.../n_tracks1/{trace}" \\
      --panel "round_robin_n2=output/.../round_robin_n2/{trace}" \\
      --panel "round_robin_n4=output/.../n_tracks4/{trace}" \\
      --panel "greedy_n2=output/.../greedy_n2/{trace}" \\
      --panel "greedy_n4=output/.../greedy_n4/{trace}" \\
      --out output/.../video/user7_ml_600mbps.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _parse_kv(items: list[str]) -> dict:
    out = {}
    for item in items:
        label, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"expected LABEL=VALUE, got: {item!r}")
        out[label] = value
    return out


def _grid_layout(n: int, cols: int) -> str:
    """xstack layout string for n equal-size panels in a row-major grid, `cols` wide.
    xstack's layout parser does NOT support "2*w0" multiplication (confirmed empirically,
    ffmpeg 5.1 -- silently collapses to overlapping positions with no error) -- explicit
    "w0+w1+..." sums instead."""
    positions = []
    for i in range(n):
        row, col = divmod(i, cols)
        x = "0" if col == 0 else "+".join(f"w{c}" for c in range(col))
        y = "0" if row == 0 else "+".join(f"h{cols * r}" for r in range(row))
        positions.append(f"{x}_{y}")
    return "|".join(positions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="e.g. user7")
    p.add_argument("--bandwidth", required=True, help="e.g. 600.0")
    p.add_argument("--method", required=True, help="e.g. ml")
    p.add_argument("--panel", required=True, action="append",
                    help="LABEL=OUTPUT_ROOT_TEMPLATE, repeatable, {trace} substituted. "
                         "GT is read from the first panel's dir (GT doesn't depend on config).")
    p.add_argument("--cols", type=int, default=None,
                    help="panels per row, including the GT panel (default: all in one row)")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=None,
                    help="playback fps. Default: real-time, i.e. 1/render-interval-sec read "
                         "from params.yaml. Override for a faster slideshow.")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    panel_templates = _parse_kv(args.panel)
    first_dir = REPO / next(iter(panel_templates.values())).format(trace=args.trace)
    params = json.loads((first_dir / "params.yaml").read_text())
    w, h = params["img_w"], params["img_h"]
    fps = args.fps if args.fps is not None else 1.0 / params["render_interval_sec"]

    panels = [("ground_truth", first_dir / "vmaf" / "gt.yuv")]
    for label, tmpl in panel_templates.items():
        d = REPO / tmpl.format(trace=args.trace)
        yuv = d / "vmaf" / f"bw_{args.bandwidth}mbps" / args.method / "distorted.yuv"
        assert yuv.exists(), f"missing: {yuv}"
        panels.append((label, yuv))

    cmd = ["ffmpeg", "-y"]
    for _, yuv in panels:
        cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
                "-r", str(fps), "-i", str(yuv)]

    filt = []
    labeled = []
    for i, (label, _) in enumerate(panels):
        filt.append(
            f"[{i}:v]scale=480:-2,drawtext=fontfile={FONT}:text='{label}':expansion=none:"
            f"fontcolor=white:fontsize=20:box=1:boxcolor=black@0.6:x=8:y=8[v{i}]")
        labeled.append(f"[v{i}]")
    n = len(panels)  # GT + len(panel_templates)
    cols = args.cols or n
    filt.append(f"{''.join(labeled)}xstack=inputs={n}:layout={_grid_layout(n, cols)}[out]")

    cmd += ["-filter_complex", ";".join(filt), "-map", "[out]",
            "-threads", str(args.threads), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(args.out)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
