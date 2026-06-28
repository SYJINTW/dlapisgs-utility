"""Per-camera Spearman ρ: marginal keys vs oracle value-per-byte.

Decisive comparison:
  ρ(baseline_key,  ΔMSE/byte)  — (v_c/d_c) / byte_k
  ρ(heuristic_key, ΔMSE/byte)  — (v_c/d_c · W_k) / byte_k

Supporting:
  ρ(W_k, ΔMSE)               — raw W_k vs oracle importance
  ρ(W_k_resid_after_N_k, ΔMSE/byte) — W_k after OLS-removing N_k component

Unit of observation: tile, at a given camera (visible tiles only).
Aggregate: median + IQR of per-camera ρ values.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n gaussian_splatting \\
        python experiments/0521/diag_ck_spearman.py \\
        --ply        <scene.ply> \\
        --tiling-cache <.tiling_cache.npz> \\
        --camera-trace <sparse_views_eval.json> \\
        --oracle     output/0615/exp5_oracle/eval/{scene}/oracle_dq.npz \\
        [--camera-indices 0 1 ... 149] [--img-w 1600] [--img-h 1600] \\
        [--out-dir output/diag_marginal_corr/{scene}]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))
sys.path.insert(0, str(WORKSPACE / "GGSP"))
sys.path.insert(0, str(WORKSPACE / "GS-Interface"))
sys.path.insert(0, str(WORKSPACE / "dlapisgs-utility"))

import visibility_AABB_pytorch  # noqa: E402
import utility_calculation as uc  # noqa: E402
import io_3dgs  # noqa: E402  # pyright: ignore[reportMissingImports]


def _screen_area_weights(gs, cam, device):
    opacity = gs.data["opacity"]["data"]
    scale_0 = gs.data["scale_0"]["data"]
    scale_1 = gs.data["scale_1"]["data"]
    scale_2 = gs.data["scale_2"]["data"]
    rot_0 = gs.data["rot_0"]["data"]
    rot_1 = gs.data["rot_1"]["data"]
    rot_2 = gs.data["rot_2"]["data"]
    rot_3 = gs.data["rot_3"]["data"]
    xyz = np.stack([gs.data["x"]["data"], gs.data["y"]["data"],
                     gs.data["z"]["data"]], axis=1)
    xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device)
    cam_center = cam.camera_center.to(device)
    return uc.compute_gaussian_weights_v2(
        "screen_area",
        opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
        xyz=xyz_t, cam_center=cam_center,
        rot_0=rot_0, rot_1=rot_1, rot_2=rot_2, rot_3=rot_3,
        world_view=cam.world_view_transform, proj=cam.projection_matrix,
        img_w=args_g.img_w, img_h=args_g.img_h,
        fov_x=getattr(cam, "FoVx", None), fov_y=getattr(cam, "FoVy", None),
    ).to(device), xyz_t


args_g = None  # set in main, used by _screen_area_weights


def _iqr(arr):
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def main():
    global args_g
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--tiling-cache", required=True)
    parser.add_argument("--camera-trace", required=True)
    parser.add_argument("--oracle", required=True,
                        help="oracle_dq.npz; key 'mse' shape (N_cams, N_tiles)")
    parser.add_argument("--camera-indices", nargs="+", type=int, default=None)
    parser.add_argument("--img-w", type=int, default=1600)
    parser.add_argument("--img-h", type=int, default=1600)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    args_g = args

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # --- Tiling ---
    tc = np.load(args.tiling_cache)
    min_corners = tc["min_corners"].astype(np.float32)
    max_corners = tc["max_corners"].astype(np.float32)
    index_offsets = tc["index_offsets"].astype(np.int64)
    flat_indices = tc["flat_indices"].astype(np.int64)
    num_tiles = len(index_offsets) - 1
    print(f"tiles={num_tiles}")

    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers_t = (min_corners_t + max_corners_t) / 2.0
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices  = torch.tensor(flat_indices,  dtype=torch.long, device=device)

    # --- Oracle ---
    oracle = np.load(args.oracle)
    n_gs = oracle["n_gs_per_tile"].astype(np.int64)   # (N_tiles,)
    mse_loo = oracle["mse"].astype(np.float64)         # (N_cams, N_tiles)
    assert len(n_gs) == num_tiles

    # --- PLY + bytes_per_gaussian ---
    print(f"loading PLY: {args.ply}")
    gs = io_3dgs.GaussianModelV2(args.ply)
    bpg = int(np.sum([np.dtype(v["val_dtype"]).itemsize for v in gs.data.values()]))
    byte_k = (n_gs * bpg).astype(np.float64)          # (N_tiles,)
    print(f"bytes_per_gaussian={bpg}")

    # --- Cameras ---
    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
        args.camera_trace, args.img_w, args.img_h
    )
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    if args.camera_indices is None:
        cam_indices = list(range(len(cameras)))
    else:
        cam_indices = sorted(set(args.camera_indices))
    print(f"cameras={len(cam_indices)}")

    # --- Per-camera ρ collectors ---
    rho_baseline  = []
    rho_heuristic = []
    rho_wk_raw    = []
    rho_wk_resid  = []

    for ci in cam_indices:
        cam = cameras[ci]
        vis = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device
        ).float().cpu().numpy()                               # (N,)
        dists = uc.calculate_distances(
            tile_centers_t, cam.camera_center.to(device)
        ).cpu().numpy()                                       # (N,)

        w_sc, _ = _screen_area_weights(gs, cam, device)
        Wk, _ = uc.compute_tile_weights_and_counts(
            tile_index_offsets, tile_flat_indices, w_sc,
            w_norm="none", c_norm="none", w_mode="sum",
        )
        Wk_np = Wk.cpu().numpy()                              # (N,)

        mask = (vis > 0) & (byte_k > 0) & (mse_loo[ci] > 0)
        if mask.sum() < 3:
            print(f"  cam {ci:03d} skip (visible={mask.sum()})")
            continue

        vd   = vis[mask] / (dists[mask] + 1e-3)
        bk   = byte_k[mask]
        Wk_m = Wk_np[mask]
        Nk_m = n_gs[mask].astype(np.float64)
        truth_per_byte = mse_loo[ci][mask] / bk

        rho_baseline.append( scipy.stats.spearmanr(vd / bk,          truth_per_byte)[0])
        rho_heuristic.append(scipy.stats.spearmanr(vd * Wk_m / bk,   truth_per_byte)[0])
        rho_wk_raw.append(   scipy.stats.spearmanr(Wk_m,              mse_loo[ci][mask])[0])

        if Nk_m.std() > 0 and Wk_m.std() > 0:
            A = np.column_stack([np.ones_like(Nk_m), Nk_m])
            coefs, *_ = np.linalg.lstsq(A, Wk_m, rcond=None)
            wk_resid = Wk_m - A @ coefs
            rho_wk_resid.append(scipy.stats.spearmanr(wk_resid, truth_per_byte)[0])

        print(f"  cam {ci:03d} done  vis={mask.sum()}")

    # --- Aggregate ---
    def _agg(rhos):
        arr = np.array(rhos)
        return {"median": float(np.median(arr)), "iqr": _iqr(arr),
                "n": len(arr), "rhos": arr.tolist()}

    signals = {
        "baseline_key":        _agg(rho_baseline),
        "heuristic_key":       _agg(rho_heuristic),
        "W_k_raw":             _agg(rho_wk_raw),
        "W_k_resid_after_N_k": _agg(rho_wk_resid),
    }

    header = f"\n{'signal':<24}  {'median_ρ':>9}  {'IQR_ρ':>7}  {'n':>5}"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, s in signals.items():
        print(f"  {name:<24}  {s['median']:+.4f}     {s['iqr']:.4f}  {s['n']:>5}")
    print(sep)

    result = {
        "n_cameras": len(cam_indices),
        "n_tiles": num_tiles,
        "bytes_per_gaussian": bpg,
        "signals": signals,
    }
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.oracle).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diag_marginal_corr.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
