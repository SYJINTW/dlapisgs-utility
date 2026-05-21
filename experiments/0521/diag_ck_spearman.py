"""Diagnostic: Spearman ρ(C_k, ε_k) for a new complexity descriptor.

ε_k = log(ΔMSE_k) − OLS(log #GS_k)  is the oracle residual after removing
the bulk-count tautology.  A descriptor is worth putting in U(k) only if
ρ(C_k, ε_k) > 0.3.

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n gsquic python experiments/0521/diag_ck_spearman.py \\
        --ply        <scene.ply> \\
        --tiling-cache <.tiling_cache.npz> \\
        --camera-trace <sparse_views_100.json> \\
        --oracle     output/0519/exp4_oracle_dq/ship/oracle_dq.npz \\
        --c-kind     eigenentropy \\
        [--camera-indices 0 1 2 ... 14]  [--weight-mode screen_area] \\
        [--img-w 800] [--img-h 800] [--out-dir /tmp/ck_diag]

Decision rule: ρ > 0.3 → proceed to full Exp 2-1 sweep.
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


def _compute_w_gi(gs, cam, args, device):
    opacity = gs.data["opacity"]["data"]
    scale_0 = gs.data["scale_0"]["data"]
    scale_1 = gs.data["scale_1"]["data"]
    scale_2 = gs.data["scale_2"]["data"]
    gs_xyz = np.stack([gs.data["x"]["data"], gs.data["y"]["data"],
                        gs.data["z"]["data"]], axis=1)
    gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)
    cam_center = cam.camera_center.to(device)

    if args.weight_mode == "screen_area":
        rot_0 = gs.data["rot_0"]["data"]
        rot_1 = gs.data["rot_1"]["data"]
        rot_2 = gs.data["rot_2"]["data"]
        rot_3 = gs.data["rot_3"]["data"]
        w_gi = uc.compute_gaussian_weights_v2(
            "screen_area",
            opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
            xyz=gs_xyz_t, cam_center=cam_center,
            rot_0=rot_0, rot_1=rot_1, rot_2=rot_2, rot_3=rot_3,
            world_view=cam.world_view_transform, proj=cam.projection_matrix,
            img_w=args.img_w, img_h=args.img_h,
            fov_x=getattr(cam, "FoVx", None), fov_y=getattr(cam, "FoVy", None),
        ).to(device)
    elif args.weight_mode == "volume":
        w_gi = uc.compute_gaussian_weights_v2(
            "volume",
            opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
        ).to(device)
    elif args.weight_mode == "volume_over_d2":
        w_gi = uc.compute_gaussian_weights_v2(
            "volume_over_d2",
            opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
            xyz=gs_xyz_t, cam_center=cam_center,
        ).to(device)
    else:
        raise ValueError(f"unsupported weight_mode '{args.weight_mode}'")

    return w_gi, gs_xyz_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--tiling-cache", required=True)
    parser.add_argument("--camera-trace", required=True)
    parser.add_argument("--oracle", required=True,
                        help="oracle_dq.npz from exp4_oracle_dq")
    parser.add_argument("--c-kind", default="eigenentropy",
                        choices=list(uc.COMPLEXITY_KINDS))
    parser.add_argument("--camera-indices", nargs="+", type=int,
                        default=list(range(15)),
                        help="Which cameras to average C_k over (default: 0..14)")
    parser.add_argument("--weight-mode", default="screen_area",
                        choices=["screen_area", "volume", "volume_over_d2"])
    parser.add_argument("--img-w", type=int, default=800)
    parser.add_argument("--img-h", type=int, default=800)
    parser.add_argument("--grid-shape", nargs=3, type=int, default=[8, 8, 8])
    parser.add_argument("--out-dir", default=None,
                        help="Where to write diag_result.json (default: next to oracle)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  c_kind={args.c_kind}  weight_mode={args.weight_mode}")

    # --- Load tiling cache ---
    tc = np.load(args.tiling_cache)
    min_corners = tc["min_corners"].astype(np.float32)
    max_corners = tc["max_corners"].astype(np.float32)
    index_offsets = tc["index_offsets"].astype(np.int64)
    flat_indices = tc["flat_indices"].astype(np.int64)
    num_tiles = len(index_offsets) - 1
    print(f"tiles={num_tiles}  total_gs_in_tiles={len(flat_indices)}")

    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices  = torch.tensor(flat_indices,  dtype=torch.long, device=device)

    # --- Load oracle labels ---
    oracle = np.load(args.oracle)
    n_gs   = oracle["n_gs_per_tile"].astype(np.int64)  # (N_tiles,)
    mse    = oracle["mse"].astype(np.float32)           # (N_cams, N_tiles)
    assert len(n_gs) == num_tiles, (
        f"tile count mismatch: oracle has {len(n_gs)}, tiling cache has {num_tiles}"
    )

    # Camera-averaged oracle ΔMSE
    mean_mse = mse.mean(axis=0)  # (N_tiles,)

    # --- Load PLY + cameras ---
    print(f"loading PLY: {args.ply}")
    gs = io_3dgs.GaussianModelV2(args.ply)

    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
        args.camera_trace, args.img_w, args.img_h
    )
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    cam_indices = sorted(set(args.camera_indices))
    if any(i >= len(cameras) for i in cam_indices):
        bad = [i for i in cam_indices if i >= len(cameras)]
        raise ValueError(f"camera indices out of range: {bad} (trace has {len(cameras)})")
    print(f"averaging C_k over {len(cam_indices)} cameras")

    # --- Compute C_k per camera, then average ---
    C_k_sum = torch.zeros(num_tiles, dtype=torch.float32, device=device)
    for ci in cam_indices:
        cam = cameras[ci]
        w_gi, gs_xyz_t = _compute_w_gi(gs, cam, args, device)
        C_k_cam = uc.compute_tile_complexity(
            args.c_kind, tile_index_offsets, tile_flat_indices, w_gi, gs_xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        )
        C_k_sum += C_k_cam
        print(f"  cam {ci:03d}: C_k range [{C_k_cam.min():.4f}, {C_k_cam.max():.4f}]"
              f"  nonzero={int((C_k_cam > 0).sum())}")

    C_k_mean = (C_k_sum / len(cam_indices)).cpu().numpy()  # (N_tiles,)

    # --- Compute OLS residual ε_k = log(mean_mse) - (β0 + β1 * log(n_gs)) ---
    # Filter: need n_gs >= COMPLEXITY_SPARSE_GUARD and mean_mse > 0
    valid = (n_gs >= uc.COMPLEXITY_SPARSE_GUARD) & (mean_mse > 0) & (C_k_mean > 0)
    n_valid = valid.sum()
    print(f"\nvalid tiles for ρ: {n_valid} / {num_tiles}")
    if n_valid < 10:
        print("ERROR: too few valid tiles for reliable Spearman ρ")
        sys.exit(1)

    log_mse = np.log(mean_mse[valid])
    log_ngs = np.log(n_gs[valid].astype(np.float64))
    C_k_valid = C_k_mean[valid]

    # OLS: log_mse ~ β0 + β1 * log_ngs
    A = np.column_stack([np.ones_like(log_ngs), log_ngs])
    coefs, _, _, _ = np.linalg.lstsq(A, log_mse, rcond=None)
    fitted = A @ coefs
    eps_k = log_mse - fitted  # residual after removing #GS bulk trend

    rho_raw,   pval_raw   = scipy.stats.spearmanr(C_k_valid, log_mse)
    rho_resid, pval_resid = scipy.stats.spearmanr(C_k_valid, eps_k)

    print(f"\n{'─'*50}")
    print(f"  c_kind        : {args.c_kind}")
    print(f"  weight_mode   : {args.weight_mode}")
    print(f"  cameras       : {cam_indices}")
    print(f"  n_valid tiles : {n_valid}")
    print(f"  OLS β0={coefs[0]:.3f} β1={coefs[1]:.3f}  (log_mse ~ β0 + β1*log_ngs)")
    print(f"  ρ(C_k, raw log_mse)   = {rho_raw:.4f}  p={pval_raw:.3e}")
    print(f"  ρ(C_k, residual ε_k)  = {rho_resid:.4f}  p={pval_resid:.3e}")
    print(f"{'─'*50}")
    if rho_resid > 0.3:
        print("  DECISION: ρ_resid > 0.3  →  PROCEED to full Exp 2-1 sweep")
    else:
        print(f"  DECISION: ρ_resid = {rho_resid:.4f} ≤ 0.3  →  pivot to voxel_entropy or abort")
    print(f"{'─'*50}\n")

    result = {
        "c_kind": args.c_kind,
        "weight_mode": args.weight_mode,
        "camera_indices": cam_indices,
        "n_valid_tiles": int(n_valid),
        "ols_beta0": float(coefs[0]),
        "ols_beta1": float(coefs[1]),
        "rho_raw": float(rho_raw),
        "pval_raw": float(pval_raw),
        "rho_resid": float(rho_resid),
        "pval_resid": float(pval_resid),
        "decision": "proceed" if rho_resid > 0.3 else "pivot",
    }

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.oracle).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"diag_{args.c_kind}_{args.weight_mode}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
