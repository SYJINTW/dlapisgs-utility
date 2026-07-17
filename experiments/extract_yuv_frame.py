#!/usr/bin/env python3
"""Extract one frame as PNG from a streaming_sim.py yuv420p rawvideo sequence (gt.yuv or
bw_*mbps/<method>/distorted.yuv), for visual root-causing of a metric dip/anomaly at a
specific time -- built 2026-07-16 after hand-rolling this exact ffmpeg call to root-cause
a PSNR/VMAF dip in the stateful streaming_sim.py rewrite (traced to a real camera-rotation
event in the trace, see .claude/PLAN.md). Reads img_w/img_h/render_interval_sec from the
run's params.yaml unless given explicitly. Pure stdlib + ffmpeg CLI -- no conda env needed.

Usage:
  python experiments/extract_yuv_frame.py --output-root <run_dir> \\
      --yuv vmaf/bw_300.0mbps/ml/distorted.yuv --t-sec 20.0 --out /tmp/frame.png
  python experiments/extract_yuv_frame.py --output-root <run_dir> \\
      --yuv vmaf/gt.yuv --frame-idx 40 --out /tmp/gt40.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--yuv", required=True,
                    help="Path relative to --output-root, e.g. vmaf/gt.yuv or "
                         "vmaf/bw_300.0mbps/ml/distorted.yuv")
    p.add_argument("--out", required=True)
    p.add_argument("--frame-idx", type=int)
    p.add_argument("--t-sec", type=float,
                    help="Converted to a frame index via params.yaml's render_interval_sec "
                         "if --frame-idx isn't given.")
    p.add_argument("--width", type=int, help="Override params.yaml's img_w")
    p.add_argument("--height", type=int, help="Override params.yaml's img_h")
    args = p.parse_args()

    root = Path(args.output_root)
    params = json.loads((root / "params.yaml").read_text())
    width = args.width or params["img_w"]
    height = args.height or params["img_h"]

    if args.frame_idx is not None:
        idx = args.frame_idx
    elif args.t_sec is not None:
        idx = round(args.t_sec / params["render_interval_sec"])
    else:
        raise SystemExit("need --frame-idx or --t-sec")

    yuv_path = root / args.yuv
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
        "-i", str(yuv_path), "-vf", f"select=eq(n\\,{idx})", "-vsync", "vfr", str(out_path),
        "-loglevel", "error",
    ], check=True)
    print(f"wrote {out_path} (frame {idx}, {width}x{height})")


if __name__ == "__main__":
    main()
