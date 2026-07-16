#!/usr/bin/env python3
"""Recompute VMAF under a different libvmaf model against already-written gt.yuv/
distorted.yuv sequences from a streaming_sim.py run (no re-simulation, no re-render).
Mirrors streaming_sim.py::run_vmaf's frame_idx-join logic exactly, but writes a new
summary CSV (default summary.csv is left untouched) so multiple model runs can coexist.

Usage:
  python experiments/recompute_vmaf.py \
      --glob "output/0715/streaming_sim_multitrack/n_tracks*/user*" \
      --model version=vmaf_4k_v0.6.1 --out-name summary_vmaf4k.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="glob matching run root dirs (each has params.yaml, metrics/summary.json, vmaf/)")
    p.add_argument("--model", required=True, help="libvmaf -m value, e.g. 'version=vmaf_4k_v0.6.1'")
    p.add_argument("--out-name", required=True, help="output CSV filename written into <run>/metrics/")
    p.add_argument("--json-name", required=True, help="per-(bw,method) vmaf json filename, avoids clobbering vmaf.json")
    args = p.parse_args()

    run_dirs = sorted(Path(d) for d in glob.glob(args.glob) if Path(d).is_dir())
    if not run_dirs:
        raise SystemExit(f"no dirs matched: {args.glob}")
    print(f"{len(run_dirs)} run dirs, model={args.model}")

    t_start = time.time()
    n_calls = 0
    for i, out_root in enumerate(run_dirs):
        params = json.loads((out_root / "params.yaml").read_text())
        img_w, img_h = params["img_w"], params["img_h"]
        summary_rows = json.loads((out_root / "metrics" / "summary.json").read_text())

        vmaf_dir = out_root / "vmaf"
        gt_yuv = vmaf_dir / "gt.yuv"

        vmaf_by_key = {}
        for bw in params["bandwidths_mbps"]:
            for method in params["methods"]:
                key_dir = vmaf_dir / f"bw_{bw}mbps" / method
                dist_yuv = key_dir / "distorted.yuv"
                out_json = key_dir / args.json_name

                subprocess.run(
                    ["vmaf", "-r", str(gt_yuv), "-d", str(dist_yuv),
                     "-w", str(img_w), "-h", str(img_h), "-p", "420", "-b", "8",
                     "--threads", "8", "--json", "-o", str(out_json),
                     "-m", args.model],
                    check=True)
                n_calls += 1
                scores = [f["metrics"]["vmaf"] for f in json.loads(out_json.read_text())["frames"]]
                for frame_idx, score in enumerate(scores):
                    vmaf_by_key[(frame_idx, method, bw)] = score

        for row in summary_rows:
            key = (row["frame_idx"], row["method"], row["bandwidth_mbps"])
            row["vmaf"] = vmaf_by_key.get(key)

        out_csv = out_root / "metrics" / args.out_name
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

        elapsed = time.time() - t_start
        print(f"[{i+1}/{len(run_dirs)}] {out_root} -> {out_csv} ({n_calls} vmaf calls, {elapsed:.0f}s elapsed)")


if __name__ == "__main__":
    main()
