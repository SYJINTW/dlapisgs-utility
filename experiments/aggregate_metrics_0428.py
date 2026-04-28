#!/usr/bin/env python3
"""Aggregate PSNR/SSIM for the 0428 trace-wide pipeline.

This script compares each rendered per-camera sequence against the shared
ground-truth render sequence and writes both a combined CSV and per-camera JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch
from PIL import Image
import torchvision.transforms.functional as tf

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT / "LapisGS-object-based-renderer"))
from utils.image_utils import psnr  # type: ignore  # noqa: E402
from utils.loss_utils import ssim  # type: ignore  # noqa: E402


def _load_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return tf.to_tensor(image).unsqueeze(0)


def _sorted_pngs(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() == ".png")


def _metrics_for_sequence(render_dir: Path, gt_dir: Path) -> dict[str, Any]:
    render_paths = _sorted_pngs(render_dir)
    gt_paths = _sorted_pngs(gt_dir)
    if not render_paths:
        raise FileNotFoundError(f"No rendered PNG files found in {render_dir}")
    if len(render_paths) != len(gt_paths):
        raise ValueError(f"Frame count mismatch: {render_dir} has {len(render_paths)} frames, GT has {len(gt_paths)}")

    psnr_values: list[float] = []
    ssim_values: list[float] = []

    for render_path, gt_path in zip(render_paths, gt_paths):
        if render_path.name != gt_path.name:
            raise ValueError(f"Frame name mismatch: {render_path.name} vs {gt_path.name}")

        render_tensor = _load_tensor(render_path)
        gt_tensor = _load_tensor(gt_path)
        psnr_values.append(float(psnr(render_tensor, gt_tensor).mean().item()))
        ssim_values.append(float(ssim(render_tensor, gt_tensor).item()))

    psnr_tensor = torch.tensor(psnr_values)
    ssim_tensor = torch.tensor(ssim_values)
    return {
        "frame_count": len(render_paths),
        "psnr_mean": float(psnr_tensor.mean().item()),
        "psnr_std": float(psnr_tensor.std(unbiased=False).item()),
        "ssim_mean": float(ssim_tensor.mean().item()),
        "ssim_std": float(ssim_tensor.std(unbiased=False).item()),
    }


def _discover_manifests(output_root: Path) -> list[Path]:
    return sorted(output_root.glob("**/selected.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 0428 rendering metrics")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Pipeline output root, defaults to dlapisgs-utility/output/0428")
    parser.add_argument("--metrics-dir", type=str, default=None,
                        help="Directory for aggregated metrics output")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root is not None else WORKSPACE_ROOT / "dlapisgs-utility" / "output" / "0428"
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir is not None else output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = output_root / "renders" / "ground_truth" / "renders"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth render directory not found: {gt_dir}")

    manifests = _discover_manifests(output_root)
    if not manifests:
        raise FileNotFoundError(f"No selection manifests found under {output_root}")

    rows: list[dict[str, Any]] = []
    per_camera: dict[str, Any] = {}

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_ply = Path(manifest["output_path"])
        rel_dir = selected_ply.parent.relative_to(output_root)
        render_dir = output_root / "renders" / rel_dir / "renders"
        metrics = _metrics_for_sequence(render_dir, gt_dir)

        row = {
            "scheme": manifest["scheme"],
            "camera_index": int(manifest["camera_index"]),
            "budget_mb": float(manifest["budget_mb"]),
            "budget_bytes": int(manifest["budget_bytes"]),
            "used_bytes": int(manifest["used_bytes"]),
            "selected_gaussians": int(manifest["selected_gaussians"]),
            "ply_bytes": int(selected_ply.stat().st_size),
            "render_dir": str(render_dir),
            "gt_dir": str(gt_dir),
            **metrics,
        }
        rows.append(row)

        per_camera_key = f"{manifest['scheme']}/camera_{int(manifest['camera_index']):03d}"
        per_camera[per_camera_key] = row

        per_json = metrics_dir / manifest["scheme"]
        per_json.mkdir(parents=True, exist_ok=True)
        (per_json / f"camera_{int(manifest['camera_index']):03d}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )

    rows.sort(key=lambda item: (item["scheme"], item["camera_index"]))

    csv_path = metrics_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = metrics_dir / "summary.json"
    json_path.write_text(json.dumps({"rows": rows, "per_camera": per_camera}, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} metric rows to {csv_path}")
    print(f"Wrote summary JSON to {json_path}")


if __name__ == "__main__":
    main()
