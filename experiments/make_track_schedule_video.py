#!/usr/bin/env python3
"""6-panel side-by-side video (GT + 5 scheduling configs) for one representative trace, to
visually eyeball how round_robin/greedy/single_track differ over time -- not a metric, a
qualitative sanity check to go with the PSNR/SSIM/VMAF comparison plots.

Reads the raw yuv420p sequences FrameWriter already wrote during run_sweep (vmaf/gt.yuv +
vmaf/bw_{bw}mbps/{method}/distorted.yuv) -- no re-render needed. Uses ffmpeg xstack + drawtext,
same tool the pipeline already depends on (FrameWriter itself is an ffmpeg subprocess).

Usage:
  python experiments/make_track_schedule_video.py --trace user7 --bandwidth 600.0 --method ml \
      --out output/0716/streaming_sim_track_compare/video/user7_ml_600mbps.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# (label, output_root template) -- template gets .format(trace=trace)
CONFIGS = [
    ("single_track", "output/0715/streaming_sim_multitrack/n_tracks1/{trace}"),
    ("round_robin_n2", "output/0716/streaming_sim_track_compare/round_robin_n2/{trace}"),
    ("round_robin_n4", "output/0715/streaming_sim_multitrack/n_tracks4/{trace}"),
    ("greedy_n2", "output/0716/streaming_sim_track_compare/greedy_n2/{trace}"),
    ("greedy_n4", "output/0716/streaming_sim_track_compare/greedy_n4/{trace}"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="e.g. user7")
    p.add_argument("--bandwidth", required=True, help="e.g. 600.0")
    p.add_argument("--method", required=True, help="e.g. ml")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=None,
                    help="playback fps. Default: real-time, i.e. 1/render-interval-sec read "
                         "from params.yaml (e.g. 0.3s interval -> 3.33fps, so video length "
                         "matches simulated session length). Override for a faster slideshow.")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    gt_dir = Path(REPO / CONFIGS[0][1].format(trace=args.trace))
    params = json.loads((gt_dir / "params.yaml").read_text())
    w, h = params["img_w"], params["img_h"]
    fps = args.fps if args.fps is not None else 1.0 / params["render_interval_sec"]

    panels = [("ground_truth", gt_dir / "vmaf" / "gt.yuv")]
    for label, tmpl in CONFIGS:
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
            f"[{i}:v]scale=480:-2,drawtext=fontfile={FONT}:text='{label}':"
            f"fontcolor=white:fontsize=20:box=1:boxcolor=black@0.6:x=8:y=8[v{i}]")
        labeled.append(f"[v{i}]")
    n = len(panels)  # 6: GT + 5 configs
    # xstack's layout parser does NOT support "2*w0" multiplication (confirmed empirically,
    # ffmpeg 5.1 -- silently collapses to overlapping positions with no error, all panels
    # scaled to equal width so plain sums are exact) -- explicit "w0+w1" sums instead.
    filt.append(f"{''.join(labeled)}xstack=inputs={n}:layout="
                f"0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0[out]")

    cmd += ["-filter_complex", ";".join(filt), "-map", "[out]",
            "-threads", str(args.threads), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(args.out)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
