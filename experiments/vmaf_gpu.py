#!/usr/bin/env python3
"""GPU VMAF (vmaf-torch) -- replaces streaming_sim.py's run_vmaf() CPU `vmaf` CLI pass.

Reads the same vmaf/*.yuv sequences FrameWriter already wrote during run_sweep and merges
scores into metrics/summary.csv, same as run_vmaf() -- but runs in the `gsquic` env (Python
3.9, vmaf-torch needs >=3.9) instead of `gaussian_splatting` (Python 3.7, where the main
pipeline runs), so this is always a separate pass, never imported into streaming_sim.py.

Model: vmaf-torch's bundled default is vmaf_v0.6.1, NOT the vmaf_4k_v0.6.1 profile every
prior run (incl. reused 07-15 single_track/round_robin_n4 baselines) used via the CPU `vmaf`
CLI's `-m version=vmaf_4k_v0.6.1`. Passing --model-json (default: the real Netflix model file
from the local libvmaf source checkout at /home/samk/builds/vmaf/model/) keeps every VMAF
number in this project on the same profile, CPU or GPU.

Usage:
  conda run -n gsquic python experiments/vmaf_gpu.py --output-root output/.../round_robin_n2/user1 --device cuda:1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from vmaf_torch import VMAF, yuv_to_tensor

DEFAULT_MODEL_JSON = "/home/samk/builds/vmaf/model/vmaf_4k_v0.6.1.json"


def _num_frames(yuv_path: Path, w: int, h: int) -> int:
    frame_bytes = w * h * 3 // 2  # yuv420p, 8-bit
    size = yuv_path.stat().st_size
    assert size % frame_bytes == 0, f"{yuv_path}: size {size} not a multiple of frame size {frame_bytes}"
    return size // frame_bytes


def compute_clip_vmaf(model: VMAF, gt_yuv: Path, dist_yuv: Path, w: int, h: int,
                       device: str, batch_size: int) -> list[float]:
    n = _num_frames(gt_yuv, w, h)
    assert n == _num_frames(dist_yuv, w, h), f"{gt_yuv} vs {dist_yuv}: frame count mismatch"
    scores: list[float] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ref = yuv_to_tensor(str(gt_yuv), w, h, n, channel="y")[start:end].to(device)
        dist = yuv_to_tensor(str(dist_yuv), w, h, n, channel="y")[start:end].to(device)
        with torch.no_grad():
            batch_scores = model.compute_vmaf_score(ref, dist)
        scores.extend(batch_scores.squeeze(-1).cpu().tolist())
    return scores


def run(output_root: Path, model_json: str, device: str, batch_size: int):
    summary_json_path = output_root / "metrics" / "summary.json"
    summary_rows = json.loads(summary_json_path.read_text())
    params = json.loads((output_root / "params.yaml").read_text())
    img_w, img_h = params["img_w"], params["img_h"]

    vmaf_dir = output_root / "vmaf"
    gt_yuv = vmaf_dir / "gt.yuv"

    model = VMAF(model_json_path=model_json, enable_motion=True).to(device).eval()

    vmaf_by_key = {}
    for bw in params["bandwidths_mbps"]:
        for method in params["methods"]:
            dist_yuv = vmaf_dir / f"bw_{bw}mbps" / method / "distorted.yuv"
            scores = compute_clip_vmaf(model, gt_yuv, dist_yuv, img_w, img_h, device, batch_size)
            for frame_idx, score in enumerate(scores):
                vmaf_by_key[(frame_idx, method, bw)] = score
            print(f"{output_root.name}: bw={bw} method={method}: {len(scores)} frames, "
                  f"mean vmaf={sum(scores) / len(scores):.2f}")

    for row in summary_rows:
        key = (row["frame_idx"], row["method"], row["bandwidth_mbps"])
        row["vmaf"] = vmaf_by_key.get(key)

    summary_csv = output_root / "metrics" / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    summary_json_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"VMAF (GPU, vmaf-torch, model={Path(model_json).name}) merged into {summary_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True, type=Path,
                    help="same --output-root a streaming_sim.py run used (must already contain "
                         "vmaf/gt.yuv + vmaf/bw_*mbps/*/distorted.yuv from run_sweep, and "
                         "metrics/summary.json)")
    p.add_argument("--model-json", default=DEFAULT_MODEL_JSON,
                    help="Netflix-format VMAF model json (default: vmaf_4k_v0.6.1, matching "
                         "every prior CPU-computed VMAF number in this project)")
    p.add_argument("--device", default="cuda:1", help="never cuda:0 -- see CLAUDE.md GPU rule")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    run(args.output_root, args.model_json, args.device, args.batch_size)


if __name__ == "__main__":
    main()
