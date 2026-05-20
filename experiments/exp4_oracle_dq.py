"""Exp 4 — LOO oracle ΔQ measurement.

For each (camera c, tile k), render the scene without tile k and record
MSE / PSNR / SSIM against the full-scene render. See
`.claude/exp4_oracle_dq.md` for the design.

Runs in the `gaussian_splatting` conda env (Python 3.7).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from loguru import logger


# --- timing / yaml helpers (copied from render_metrics.py for parity) -------

@contextmanager
def _timed(name: str, store: list, **labels):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        label_str = " ".join(f"{k}={v}" for k, v in labels.items())
        logger.info("{} stage={} t={:.3f}s", label_str, name, dt)
        row = {"stage": name, "t_sec": dt}
        row.update(labels)
        store.append(row)


def _yaml_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if obj is None or isinstance(obj, bool) or isinstance(obj, (int, float)):
        return obj
    return str(obj)


def _dump_params(output_root: Path, args: argparse.Namespace) -> None:
    payload = {
        "run": {
            "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cwd": os.getcwd(),
            "script": str(Path(__file__).resolve()),
        },
        "args": vars(args),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "params.yaml").write_text(
        yaml.safe_dump(_yaml_safe(payload), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


# --- renderer imports (paths copied from render_metrics.py) ------------------

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RENDERER_ROOT = WORKSPACE_ROOT / "LapisGS-object-based-renderer"
sys.path.insert(0, str(RENDERER_ROOT))

_DUMMY = WORKSPACE_ROOT / "exp-dataset" / "chair" / "predictions" / "color" / "test" / "r_0.png"
os.environ.setdefault("LAPISGS_DUMMY_IMAGE", str(_DUMMY))

from gaussian_renderer_lapisgs import GaussianModel, render  # type: ignore  # noqa: E402
from streaming_utils.camera_loader import load_camera_from_streaming_config  # type: ignore  # noqa: E402
from utils.loss_utils import ssim as _ssim  # type: ignore  # noqa: E402
import torchvision  # type: ignore  # noqa: E402

# Local helpers (same dir on sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_dq import (  # noqa: E402
    OracleDQResult,
    iter_tile_member_indices,
    load_tiling,
    n_gs_per_tile as _n_gs_per_tile,
    write_long_csv,
)


class _FakePipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


PIPELINE = _FakePipe()


# --- per-Gaussian tensor names (confirmed by D1 probe) ----------------------

_GS_TENSOR_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_opacity",
    "_scaling",
    "_rotation",
)


def _snapshot_gaussians(g) -> dict[str, torch.Tensor]:
    snaps: dict[str, torch.Tensor] = {}
    for name in _GS_TENSOR_NAMES:
        t = getattr(g, name)
        snaps[name] = t.detach().clone()
    return snaps


def _apply_subset(g, snaps: dict[str, torch.Tensor], keep_idx: torch.Tensor) -> None:
    """Restore from snapshot then slice by keep_idx, in place on the GaussianModel."""
    for name in _GS_TENSOR_NAMES:
        setattr(g, name, snaps[name].index_select(0, keep_idx))


def _restore_full(g, snaps: dict[str, torch.Tensor]) -> None:
    for name in _GS_TENSOR_NAMES:
        setattr(g, name, snaps[name])


# --- camera / trace ---------------------------------------------------------

def _load_trace(trace_path: Path) -> list[dict]:
    with open(trace_path) as f:
        data = json.load(f)
    angle_x = data["camera_angle_x"]
    for frame in data["frames"]:
        frame["camera_angle_x"] = angle_x
    return data["frames"]


def _build_camera(frame: dict, width: int, height: int):
    return load_camera_from_streaming_config(frame, width=width, height=height)


# --- rendering --------------------------------------------------------------

def _render_current(g, camera, white_bg: bool) -> torch.Tensor:
    """Render with whatever Gaussian tensors are currently assigned."""
    gs_res = torch.ones(len(g.get_xyz), device="cuda")
    bg = [1, 1, 1] if white_bg else [0, 0, 0]
    bg_color = torch.tensor(bg, dtype=torch.float32, device="cuda").view(3, 1, 1)
    bg_color = bg_color.expand(3, camera.image_height, camera.image_width)
    bg_depth = torch.zeros(1, camera.image_height, camera.image_width, device="cuda")
    with torch.no_grad():
        result = render(camera, g, PIPELINE, bg_color, bg_depth, gs_res=gs_res)
    return result["render"].clamp(0.0, 1.0)


def _mse_psnr(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    mse = F.mse_loss(a, b).item()
    psnr = -10.0 * float(np.log10(max(mse, 1e-12)))
    return float(mse), psnr


def _ssim_val(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(_ssim(a.unsqueeze(0), b.unsqueeze(0)).item())


# --- main -------------------------------------------------------------------

def _parse_int_list(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="LOO oracle ΔQ for tile importance (Exp 4)")
    parser.add_argument("--ply", required=True, help="Full-scene PLY")
    parser.add_argument("--trace", required=True, help="Camera trace JSON")
    parser.add_argument("--tiling-cache", required=True, help=".tiling_cache.npz")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-cameras", type=int, default=0,
                        help="Take first N cameras from trace. 0 = all.")
    parser.add_argument("--camera-indices", type=str, default=None,
                        help="Explicit comma-sep camera indices (overrides --num-cameras)")
    parser.add_argument("--tile-indices", type=str, default=None,
                        help="Explicit comma-sep tile indices (default = all non-empty)")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--white-bg", action="store_true")
    parser.add_argument("--save-pt", action="store_true",
                        help="Also dump R_full(c) as float16 .pt for reuse.")
    parser.add_argument("--save-ablation-png", action="store_true",
                        help="Save R_-k(c) PNG per (tile, camera). ~30 GB on bicycle.")
    parser.add_argument("--flush-every", type=int, default=10,
                        help="Flush partial NPZ every N tiles for crash recovery.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse existing NPZ; skip (c,k) entries already filled (non-NaN).")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    log_path = output_root / "exp4_oracle_dq.log"
    logger.add(str(log_path), level="INFO")
    logger.info("output_root={} ply={} trace={}", output_root, args.ply, args.trace)
    logger.info("tiling_cache={}", args.tiling_cache)

    _dump_params(output_root, args)
    timings: list = []

    # --- tiling -------------------------------------------------------------
    with _timed("tiling_load", timings):
        tiling = load_tiling(args.tiling_cache)
    n_tiles_all = tiling["n_tiles"]
    logger.info("Loaded tiling: N_tiles={} total_GS={}", n_tiles_all,
                int(tiling["flat_indices"].shape[0]))

    tile_subset = _parse_int_list(args.tile_indices)
    tile_indices: list[int] = (
        list(range(n_tiles_all)) if tile_subset is None else tile_subset
    )

    # --- cameras ------------------------------------------------------------
    with _timed("camera_load", timings):
        frames = _load_trace(Path(args.trace))
        cameras_all = [_build_camera(f, args.width, args.height) for f in frames]

    cam_subset = _parse_int_list(args.camera_indices)
    if cam_subset is not None:
        camera_indices = cam_subset
    elif args.num_cameras and args.num_cameras > 0:
        camera_indices = list(range(min(args.num_cameras, len(cameras_all))))
    else:
        camera_indices = list(range(len(cameras_all)))
    cameras = [cameras_all[c] for c in camera_indices]
    logger.info("Using {} cameras (from {} in trace)", len(cameras), len(cameras_all))

    n_cams = len(camera_indices)
    n_tiles = n_tiles_all  # matrices always sized to all tiles for stable indexing

    # --- result matrices (load existing if resuming) ------------------------
    npz_path = output_root / "oracle_dq.npz"
    if args.skip_existing and npz_path.exists():
        logger.info("Resuming from existing {}", npz_path)
        prev = OracleDQResult.load_npz(npz_path)
        mse_mat = prev.mse.copy()
        psnr_mat = prev.psnr.copy()
        ssim_mat = prev.ssim.copy()
        if mse_mat.shape != (n_cams, n_tiles):
            logger.warning("Existing NPZ shape {} != requested {}; reinitializing.",
                           mse_mat.shape, (n_cams, n_tiles))
            mse_mat = np.full((n_cams, n_tiles), np.nan, dtype=np.float32)
            psnr_mat = np.full((n_cams, n_tiles), np.nan, dtype=np.float32)
            ssim_mat = np.full((n_cams, n_tiles), np.nan, dtype=np.float32)
    else:
        mse_mat = np.full((n_cams, n_tiles), np.nan, dtype=np.float32)
        psnr_mat = np.full((n_cams, n_tiles), np.nan, dtype=np.float32)
        ssim_mat = np.full((n_cams, n_tiles), np.nan, dtype=np.float32)

    full_dir = output_root / "full_renders"
    full_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir = output_root / "ablation_renders"
    if args.save_ablation_png:
        ablation_dir.mkdir(parents=True, exist_ok=True)

    # --- load scene + snapshot ---------------------------------------------
    logger.info("Loading scene PLY...")
    with _timed("scene_load", timings):
        g = GaussianModel(args.sh_degree)
        g.load_ply(str(Path(args.ply)))
        snaps = _snapshot_gaussians(g)
    n_total = int(g.get_xyz.shape[0])
    logger.info("Scene loaded: N_total={} GS", n_total)

    # --- full-scene reference renders (kept on GPU at float32) --------------
    logger.info("Rendering R_full(c) for {} cameras...", n_cams)
    full_renders: list[torch.Tensor] = []
    with _timed("full_render_all", timings, n=n_cams):
        with torch.no_grad():
            for i, (c_global, cam) in enumerate(zip(camera_indices, cameras)):
                img = _render_current(g, cam, args.white_bg)
                full_renders.append(img)  # GPU float32
                png_path = full_dir / f"camera_{c_global:03d}.png"
                if not png_path.exists():
                    torchvision.utils.save_image(img, str(png_path))
                if args.save_pt:
                    pt_path = full_dir / f"camera_{c_global:03d}.pt"
                    if not pt_path.exists():
                        torch.save(img.half().cpu(), str(pt_path))
    logger.success("R_full rendered: {} frames", n_cams)

    # --- determinism self-test: render full scene twice, expect MSE ≈ 0 ----
    with _timed("determinism_check", timings):
        with torch.no_grad():
            img_check = _render_current(g, cameras[0], args.white_bg)
        det_mse = F.mse_loss(img_check, full_renders[0]).item()
        if det_mse > 1e-10:
            logger.error(
                "Rasterizer not deterministic: MSE(R_full, R_full)={:.3e}. "
                "LOO oracle would be floored at this value. Aborting.", det_mse)
            raise SystemExit(2)
        logger.info("Determinism OK: MSE(R_full, R_full)={:.3e}", det_mse)
        del img_check

    # --- LOO loop -----------------------------------------------------------
    all_idx = torch.arange(n_total, dtype=torch.long, device="cuda")

    def _flush_partial():
        result = OracleDQResult(
            min_corners=tiling["min_corners"],
            max_corners=tiling["max_corners"],
            index_offsets=tiling["index_offsets"],
            flat_indices=tiling["flat_indices"],
            camera_indices=np.asarray(camera_indices, dtype=np.int32),
            n_gs_per_tile=_n_gs_per_tile(tiling),
            mse=mse_mat,
            psnr=psnr_mat,
            ssim=ssim_mat,
            gen_meta={
                "scene_ply": str(Path(args.ply)),
                "trace": str(Path(args.trace)),
                "tiling_cache": str(Path(args.tiling_cache)),
                "width": args.width,
                "height": args.height,
                "sh_degree": args.sh_degree,
                "white_bg": bool(args.white_bg),
                "n_total_gs": n_total,
                "schema_note": (
                    "tile_keys absent (test_utility cache schema); "
                    "min_corners/max_corners serve as tile identity."
                ),
                "timings_so_far": timings.copy(),
            },
        )
        result.save_npz(npz_path)

    for ti, k in enumerate(tile_indices):
        member = tiling["flat_indices"][tiling["index_offsets"][k] : tiling["index_offsets"][k + 1]]
        n_member = int(member.shape[0])
        if n_member == 0:
            logger.warning("tile {} empty, skipping", k)
            continue

        # If resuming and all cameras already filled for this tile, skip rendering.
        if args.skip_existing and np.all(np.isfinite(mse_mat[:, k])):
            logger.info("tile {} already complete (skip-existing)", k)
            continue

        with _timed("tile_subset", timings, k=k, n_member=n_member):
            # boolean keep over the full index set
            keep_mask = torch.ones(n_total, dtype=torch.bool, device="cuda")
            member_t = torch.as_tensor(member, dtype=torch.long, device="cuda")
            keep_mask[member_t] = False
            keep_idx = all_idx[keep_mask]
            _apply_subset(g, snaps, keep_idx)

        ablation_tile_dir: Path | None = None
        if args.save_ablation_png:
            ablation_tile_dir = ablation_dir / f"tile_{k:04d}"
            ablation_tile_dir.mkdir(parents=True, exist_ok=True)

        with _timed("tile_render_all_cams", timings, k=k, n_cams=n_cams):
            with torch.no_grad():
                for ci, (c_global, cam) in enumerate(zip(camera_indices, cameras)):
                    if args.skip_existing and np.isfinite(mse_mat[ci, k]):
                        continue
                    img = _render_current(g, cam, args.white_bg)
                    ref = full_renders[ci]
                    mse_v, psnr_v = _mse_psnr(img, ref)
                    ssim_v = _ssim_val(img, ref)
                    mse_mat[ci, k] = mse_v
                    psnr_mat[ci, k] = psnr_v
                    ssim_mat[ci, k] = ssim_v
                    if ablation_tile_dir is not None:
                        torchvision.utils.save_image(
                            img,
                            str(ablation_tile_dir / f"camera_{c_global:03d}.png"),
                        )

        if (ti + 1) % max(args.flush_every, 1) == 0:
            with _timed("flush_partial", timings, tiles_done=ti + 1):
                _flush_partial()
            logger.info("[{}/{}] tile {} done; partial flushed",
                        ti + 1, len(tile_indices), k)

    # --- final flush --------------------------------------------------------
    with _timed("final_flush", timings):
        _flush_partial()

    with _timed("long_csv", timings):
        result = OracleDQResult.load_npz(npz_path)
        write_long_csv(result, output_root / "oracle_dq.csv")

    timings_path = output_root / "timings.json"
    timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    logger.success("Done. NPZ: {}", npz_path)


if __name__ == "__main__":
    main()
