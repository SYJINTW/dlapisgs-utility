#!/usr/bin/env python3
"""Render selected PLYs from the 0507 budget sweep and compute PSNR/SSIM.

Run in the gaussian_splatting conda environment via run_render_metrics_0507.sh.

Strategy:
  - Pre-render the ground-truth PLY at all 50 camera poses (once).
  - For each selected.json manifest, render its selected.ply at only the
    matching camera_index frame and compare to the cached GT tensor.
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

# ---------------------------------------------------------------------------
# Path setup – import LapisGS renderer utilities
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RENDERER_ROOT = WORKSPACE_ROOT / "LapisGS-object-based-renderer"
sys.path.insert(0, str(RENDERER_ROOT))

# LAPISGS_DUMMY_IMAGE must be set before importing camera_loader.
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
    r = rendered.unsqueeze(0)
    g = gt.unsqueeze(0)
    return {
        "psnr": float(psnr(r, g).mean().item()),
        "ssim": float(ssim(r, g).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render + metrics for 0507 budget sweep")
    parser.add_argument("--output-root", default=None,
                        help="Budget sweep output root (default: dlapisgs-utility/output/0507_budget_sweep)")
    parser.add_argument("--gt-ply", default=None,
                        help="Ground-truth PLY (default: exp-dataset/bicycle/point_cloud.ply)")
    parser.add_argument("--trace", default=None,
                        help="Camera trace JSON (default: Frustum-for-3DGS/sample_data/camera_trace/trace1.json)")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--white-bg", action="store_true")
    parser.add_argument("--render-dir", default=None,
                        help="Where to save GT and per-selection PNG renders (optional)")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else \
        WORKSPACE_ROOT / "dlapisgs-utility" / "output" / "0507_budget_sweep"
    gt_ply = Path(args.gt_ply) if args.gt_ply else \
        WORKSPACE_ROOT / "exp-dataset" / "bicycle" / "point_cloud.ply"
    trace_path = Path(args.trace) if args.trace else \
        WORKSPACE_ROOT / "Frustum-for-3DGS" / "sample_data" / "camera_trace" / "trace1.json"
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    render_dir: Path | None = Path(args.render_dir) if args.render_dir else None
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_trace(trace_path)
    cameras = [_build_camera(f, args.width, args.height) for f in frames]
    print(f"Loaded {len(cameras)} cameras from {trace_path}")

    # ------------------------------------------------------------------
    # Pre-render ground truth at all camera poses
    # ------------------------------------------------------------------
    print(f"Rendering GT from {gt_ply} ...")
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
    print(f"GT rendered: {len(gt_renders)} frames")

    del gt_gaussians  # free VRAM

    # ------------------------------------------------------------------
    # Iterate manifests and render each selection at its camera index
    # ------------------------------------------------------------------
    manifests = sorted(output_root.glob("**/selected.json"))
    print(f"Found {len(manifests)} manifests under {output_root}")

    rows: list[dict[str, Any]] = []

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        cam_idx: int = int(manifest["camera_index"])
        scheme: str = manifest["scheme"]
        budget_mb: float = float(manifest["budget_mb"])
        selected_ply = Path(manifest["output_path"])

        if not selected_ply.exists():
            print(f"  SKIP (PLY missing): {selected_ply}")
            continue

        cam = cameras[cam_idx]
        rendered = _render_ply(selected_ply, cam, args.sh_degree, args.white_bg)
        gt = gt_renders[cam_idx]

        m = _compute_metrics(rendered, gt)

        # Optionally save the rendered PNG
        if render_dir is not None:
            import torchvision
            rel = selected_ply.parent.relative_to(output_root)
            out_png = render_dir / rel.parent / f"{rel.name}.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(rendered, str(out_png))

        row: dict[str, Any] = {
            "budget_mb": budget_mb,
            "scheme": scheme,
            "camera_index": cam_idx,
            "psnr": m["psnr"],
            "ssim": m["ssim"],
            "used_bytes": int(manifest.get("used_bytes", 0)),
            "selected_gaussians": int(manifest.get("selected_gaussians", 0)),
            "ply_bytes": int(selected_ply.stat().st_size),
        }
        rows.append(row)

        # Per-camera JSON
        budget_tag = f"budget_{int(budget_mb)}mb"
        per_json_dir = metrics_dir / budget_tag / scheme
        per_json_dir.mkdir(parents=True, exist_ok=True)
        (per_json_dir / f"camera_{cam_idx:03d}.json").write_text(
            json.dumps({**manifest, **m}, indent=2)
        )

        print(f"  [{budget_tag}/{scheme}/camera_{cam_idx:03d}]  "
              f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    # ------------------------------------------------------------------
    # Summary CSV and JSON
    # ------------------------------------------------------------------
    rows.sort(key=lambda r: (r["budget_mb"], r["scheme"], r["camera_index"]))

    csv_path = metrics_dir / "summary.csv"
    if rows:
        with csv_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows → {csv_path}")

    json_path = metrics_dir / "summary.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"Wrote summary → {json_path}")


if __name__ == "__main__":
    main()
