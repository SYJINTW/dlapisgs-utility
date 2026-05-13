#!/usr/bin/env python3
"""Render selected PLYs from a budget sweep and compute PSNR/SSIM.

Run in the gaussian_splatting conda environment.

Strategy:
  - Pre-render the ground-truth PLY at all camera poses (once).
  - For each selected.json manifest under --output-root, render its
    selected.ply at only the matching camera_index and compare to GT.
  - Write per-(budget, scheme, camera) JSON and a combined summary CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RENDERER_ROOT = WORKSPACE_ROOT / "LapisGS-object-based-renderer"
sys.path.insert(0, str(RENDERER_ROOT))

_DUMMY = WORKSPACE_ROOT / "exp-dataset" / "chair" / "predictions" / "color" / "test" / "r_0.png"
os.environ.setdefault("LAPISGS_DUMMY_IMAGE", str(_DUMMY))

from gaussian_renderer_lapisgs import GaussianModel, render  # type: ignore
from streaming_utils.camera_loader import load_camera_from_streaming_config  # type: ignore
from utils.image_utils import psnr  # type: ignore
from utils.loss_utils import ssim  # type: ignore


class _FakePipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


PIPELINE = _FakePipe()


def _load_trace(trace_path: Path) -> list[dict]:
    with open(trace_path) as f:
        data = json.load(f)
    angle_x = data["camera_angle_x"]
    for frame in data["frames"]:
        frame["camera_angle_x"] = angle_x
    return data["frames"]


def _render_ply(ply_path: Path, camera, sh_degree: int, white_bg: bool) -> torch.Tensor:
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path))
    gs_res = torch.ones(len(gaussians.get_xyz), device="cuda")
    bg = [1, 1, 1] if white_bg else [0, 0, 0]
    bg_color = torch.tensor(bg, dtype=torch.float32, device="cuda").view(3, 1, 1)
    bg_color = bg_color.expand(3, camera.image_height, camera.image_width)
    bg_depth = torch.zeros(1, camera.image_height, camera.image_width, device="cuda")
    with torch.no_grad():
        result = render(camera, gaussians, PIPELINE, bg_color, bg_depth, gs_res=gs_res)
    return result["render"].clamp(0.0, 1.0)


def _build_camera(frame: dict, width: int, height: int):
    return load_camera_from_streaming_config(frame, width=width, height=height)


def _compute_metrics(rendered: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    r, g = rendered.unsqueeze(0), gt.unsqueeze(0)
    return {
        "psnr": float(psnr(r, g).mean().item()),
        "ssim": float(ssim(r, g).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render + metrics for a budget sweep")
    parser.add_argument("--output-root", required=True,
                        help="Budget sweep output root (contains budget_*mb/ subdirs)")
    parser.add_argument("--gt-ply", required=True,
                        help="Ground-truth PLY path")
    parser.add_argument("--trace", required=True,
                        help="Camera trace JSON path")
    parser.add_argument("--width",     type=int, default=800)
    parser.add_argument("--height",    type=int, default=800)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--white-bg",  action="store_true")
    parser.add_argument("--render-dir", default=None,
                        help="Where to save GT and per-selection PNG renders (optional)")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    gt_ply      = Path(args.gt_ply)
    trace_path  = Path(args.trace)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    log_path = metrics_dir / "render_metrics.log"
    logger.add(str(log_path), level="INFO")
    logger.info("output_root={} gt_ply={} trace={}", output_root, gt_ply, trace_path)

    render_dir: Path | None = Path(args.render_dir) if args.render_dir else None
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    frames  = _load_trace(trace_path)
    cameras = [_build_camera(f, args.width, args.height) for f in frames]
    logger.info("Loaded {} cameras from {}", len(cameras), trace_path)

    # Pre-render ground truth
    logger.info("Rendering GT from {} ...", gt_ply)
    gt_gaussians = GaussianModel(args.sh_degree)
    gt_gaussians.load_ply(str(gt_ply))
    gt_gs_res = torch.ones(len(gt_gaussians.get_xyz), device="cuda")
    bg = [1, 1, 1] if args.white_bg else [0, 0, 0]
    gt_renders: list[torch.Tensor] = []
    with torch.no_grad():
        for idx, cam in enumerate(cameras):
            bg_color = torch.tensor(bg, dtype=torch.float32, device="cuda").view(3, 1, 1)
            bg_color = bg_color.expand(3, cam.image_height, cam.image_width)
            bg_depth = torch.zeros(1, cam.image_height, cam.image_width, device="cuda")
            result = render(cam, gt_gaussians, PIPELINE, bg_color, bg_depth, gs_res=gt_gs_res)
            frame_tensor = result["render"].clamp(0.0, 1.0)
            gt_renders.append(frame_tensor)
            if render_dir is not None:
                import torchvision
                gt_frame_path = render_dir / "ground_truth" / f"{idx:05d}.png"
                gt_frame_path.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(frame_tensor, str(gt_frame_path))
    logger.success("GT rendered: {} frames", len(gt_renders))
    del gt_gaussians

    # Iterate manifests
    manifests = sorted((output_root / "ply").glob("**/camera_*.json")
                       if (output_root / "ply").exists()
                       else output_root.glob("**/selected.json"))
    logger.info("Found {} manifests under {}", len(manifests), output_root)

    rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest   = json.loads(manifest_path.read_text())
        cam_idx    = int(manifest["camera_index"])
        scheme     = manifest["scheme"]
        budget_mb  = float(manifest["budget_mb"])
        selected_ply = Path(manifest["output_path"])

        if not selected_ply.exists():
            logger.warning("SKIP (PLY missing): {}", selected_ply)
            continue

        rendered = _render_ply(selected_ply, cameras[cam_idx], args.sh_degree, args.white_bg)
        m = _compute_metrics(rendered, gt_renders[cam_idx])

        if render_dir is not None:
            import torchvision
            rel = selected_ply.relative_to(output_root)
            out_png = render_dir / rel.with_suffix(".png")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(rendered, str(out_png))

        row: dict[str, Any] = {
            "budget_mb":          budget_mb,
            "scheme":             scheme,
            "camera_index":       cam_idx,
            "psnr":               m["psnr"],
            "ssim":               m["ssim"],
            "used_bytes":         int(manifest.get("used_bytes", 0)),
            "selected_gaussians": int(manifest.get("selected_gaussians", 0)),
            "ply_bytes":          int(selected_ply.stat().st_size),
        }
        rows.append(row)

        budget_tag = f"budget_{int(budget_mb)}mb"
        per_json_dir = metrics_dir / budget_tag / scheme
        per_json_dir.mkdir(parents=True, exist_ok=True)
        (per_json_dir / f"camera_{cam_idx:03d}.json").write_text(
            json.dumps({**manifest, **m}, indent=2)
        )
        logger.success("[{}/{}/camera_{:03d}]  PSNR={:.2f}  SSIM={:.4f}",
                       budget_tag, scheme, cam_idx, m["psnr"], m["ssim"])

    rows.sort(key=lambda r: (r["budget_mb"], r["scheme"], r["camera_index"]))

    csv_path = metrics_dir / "summary.csv"
    if rows:
        with csv_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote {} rows → {}", len(rows), csv_path)

    json_path = metrics_dir / "summary.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2))
    logger.info("Wrote summary → {}", json_path)


if __name__ == "__main__":
    main()
