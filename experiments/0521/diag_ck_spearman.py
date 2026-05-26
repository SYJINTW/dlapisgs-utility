"""Comprehensive diagnostic: Spearman ρ(X, ε_k) for all tile-ranking candidates.

ε_k = log(ΔMSE_k) − OLS(log #GS_k)  strips the bulk-count tautology.
A signal worth putting in U(k) should have |ρ_resid| > 0.3.

Candidates evaluated:
  - N_k          : #GS count (view-independent)
  - v_k          : visibility fraction (camera-averaged)
  - 1/d_k        : inverse distance (camera-averaged)
  - W_k_screen   : Σ w_i (screen_area weights, camera-averaged)
  - W_k_vol_d2   : Σ w_i (volume_over_d2 weights, camera-averaged)
  - C_eigenentropy, C_omnivariance, C_voxel_entropy, C_spectral_energy  (screen_area weighted)
  - C_eigenentropy_pos, C_omnivariance_pos, C_voxel_entropy_pos, C_spectral_pos  (uniform weights)

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n gsquic python experiments/0521/diag_ck_spearman.py \\
        --ply        <scene.ply> \\
        --tiling-cache <.tiling_cache.npz> \\
        --camera-trace <sparse_views_100.json> \\
        --oracle     output/0519/exp4_oracle_dq/ship/oracle_dq.npz \\
        [--camera-indices 0 1 ... 14]  [--img-w 800] [--img-h 800] \\
        [--out-dir /tmp/ck_diag]
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


def _vol_d2_weights(gs, cam, device):
    opacity = gs.data["opacity"]["data"]
    scale_0 = gs.data["scale_0"]["data"]
    scale_1 = gs.data["scale_1"]["data"]
    scale_2 = gs.data["scale_2"]["data"]
    xyz = np.stack([gs.data["x"]["data"], gs.data["y"]["data"],
                     gs.data["z"]["data"]], axis=1)
    xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device)
    cam_center = cam.camera_center.to(device)
    return uc.compute_gaussian_weights_v2(
        "volume_over_d2",
        opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
        xyz=xyz_t, cam_center=cam_center,
    ).to(device), xyz_t


args_g = None  # set in main, used by _screen_area_weights


def main():
    global args_g
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--tiling-cache", required=True)
    parser.add_argument("--camera-trace", required=True)
    parser.add_argument("--oracle", required=True,
                        help="oracle_dq.npz from exp4_oracle_dq")
    parser.add_argument("--camera-indices", nargs="+", type=int,
                        default=None,
                        help="Cameras to average over (default: all cameras in trace)")
    parser.add_argument("--img-w", type=int, default=800)
    parser.add_argument("--img-h", type=int, default=800)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--oracle-mode", choices=["loo", "aoi", "combined"], default="loo",
                        help="loo: MSE(full\\k); aoi: MSE({k}); combined: rank-avg of both")
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

    # View-independent: #GS count (raw, not normalized)
    sizes = (index_offsets[1:] - index_offsets[:-1]).astype(np.float64)

    # --- Oracle ---
    oracle = np.load(args.oracle)
    n_gs    = oracle["n_gs_per_tile"].astype(np.int64)
    mse_loo = oracle["mse"].astype(np.float32)   # (N_cams, N_tiles) — higher = more important
    assert len(n_gs) == num_tiles

    if args.oracle_mode == "loo":
        mean_mse = mse_loo.mean(axis=0)
    elif args.oracle_mode in ("aoi", "combined"):
        if "mse_aoi" not in oracle:
            raise SystemExit("oracle NPZ has no mse_aoi — re-run exp4 with --compute-aoi")
        mse_aoi = oracle["mse_aoi"].astype(np.float32)  # lower = more important
        if args.oracle_mode == "aoi":
            mean_mse = mse_aoi.mean(axis=0)   # will be negated in log below
        else:
            from scipy.stats import rankdata
            loo_rank = rankdata(mse_loo.mean(axis=0))       # higher loo MSE = higher rank
            aoi_rank = rankdata(-mse_aoi.mean(axis=0))      # lower aoi MSE = higher rank
            mean_mse = (loo_rank + aoi_rank) / 2.0          # combined importance score

    # --- PLY + cameras ---
    print(f"loading PLY: {args.ply}")
    gs = io_3dgs.GaussianModelV2(args.ply)
    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
        args.camera_trace, args.img_w, args.img_h
    )
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    if args.camera_indices is None:
        cam_indices = list(range(len(cameras)))
    else:
        cam_indices = sorted(set(args.camera_indices))
    print(f"averaging over {len(cam_indices)} cameras")

    # Accumulators for camera-averaged quantities
    vis_sum      = np.zeros(num_tiles)
    invd_sum     = np.zeros(num_tiles)
    Wk_sc_sum    = np.zeros(num_tiles)   # W_k screen_area
    Wk_vd_sum    = np.zeros(num_tiles)   # W_k volume_over_d2
    Ck_ee_sum    = np.zeros(num_tiles)   # eigenentropy (screen_area weighted)
    Ck_ov_sum    = np.zeros(num_tiles)   # omnivariance (screen_area weighted)
    Ck_ve_sum    = np.zeros(num_tiles)   # voxel_entropy (screen_area weighted)
    Ck_se_sum    = np.zeros(num_tiles)   # spectral_energy (screen_area weighted)
    # Positional-only (uniform weights): C_k measures shape/layout, not opacity/scale
    Ck_ee_pos_sum = np.zeros(num_tiles)
    Ck_ov_pos_sum = np.zeros(num_tiles)
    Ck_ve_pos_sum = np.zeros(num_tiles)
    Ck_se_pos_sum = np.zeros(num_tiles)

    for ci in cam_indices:
        cam = cameras[ci]
        vis = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device
        ).float().cpu().numpy()                               # (N,) 0/1
        dists = uc.calculate_distances(
            tile_centers_t, cam.camera_center.to(device)
        ).cpu().numpy()                                       # (N,)

        w_sc, xyz_t = _screen_area_weights(gs, cam, device)
        w_vd, _     = _vol_d2_weights(gs, cam, device)
        w_unif      = torch.ones_like(w_sc)

        Wk_sc, _ = uc.compute_tile_weights_and_counts(
            tile_index_offsets, tile_flat_indices, w_sc, w_norm="none", c_norm="none"
        )
        Wk_vd, _ = uc.compute_tile_weights_and_counts(
            tile_index_offsets, tile_flat_indices, w_vd, w_norm="none", c_norm="none"
        )
        Ck_ee = uc.compute_tile_complexity(
            "eigenentropy", tile_index_offsets, tile_flat_indices, w_sc, xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        )
        Ck_ov = uc.compute_tile_complexity(
            "omnivariance", tile_index_offsets, tile_flat_indices, w_sc, xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        )
        Ck_ve = uc.compute_tile_complexity(
            "voxel_entropy", tile_index_offsets, tile_flat_indices, w_sc, xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        )
        Ck_se = uc.compute_tile_complexity(
            "spectral_energy", tile_index_offsets, tile_flat_indices, w_sc, xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        )
        Ck_ee_pos = uc.compute_tile_complexity(
            "eigenentropy", tile_index_offsets, tile_flat_indices, w_unif, xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        )
        Ck_ov_pos = uc.compute_tile_complexity(
            "omnivariance", tile_index_offsets, tile_flat_indices, w_unif, xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        )
        Ck_ve_pos = uc.compute_tile_complexity(
            "voxel_entropy", tile_index_offsets, tile_flat_indices, w_unif, xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        )
        Ck_se_pos = uc.compute_tile_complexity(
            "spectral_energy", tile_index_offsets, tile_flat_indices, w_unif, xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        )

        vis_sum       += vis
        invd_sum      += 1.0 / (dists + 1e-3)
        Wk_sc_sum     += Wk_sc.cpu().numpy()
        Wk_vd_sum     += Wk_vd.cpu().numpy()
        Ck_ee_sum     += Ck_ee.cpu().numpy()
        Ck_ov_sum     += Ck_ov.cpu().numpy()
        Ck_ve_sum     += Ck_ve.cpu().numpy()
        Ck_se_sum     += Ck_se.cpu().numpy()
        Ck_ee_pos_sum += Ck_ee_pos.cpu().numpy()
        Ck_ov_pos_sum += Ck_ov_pos.cpu().numpy()
        Ck_ve_pos_sum += Ck_ve_pos.cpu().numpy()
        Ck_se_pos_sum += Ck_se_pos.cpu().numpy()
        print(f"  cam {ci:03d} done")

    n_cam = len(cam_indices)
    Wk_sc_avg = Wk_sc_sum / n_cam
    Wk_vd_avg = Wk_vd_sum / n_cam
    w_bar_k_sc = np.where(sizes > 0, Wk_sc_avg / sizes, 0.0)
    w_bar_k_vd = np.where(sizes > 0, Wk_vd_avg / sizes, 0.0)
    candidates = {
        "N_k":                 sizes,
        "v_k":                 vis_sum    / n_cam,
        "1/d_k":               invd_sum   / n_cam,
        "W_k_screen":          Wk_sc_avg,
        "W_k_vol_d2":          Wk_vd_avg,
        "wbar_k_screen":       w_bar_k_sc,
        "wbar_k_vol_d2":       w_bar_k_vd,
        "C_eigenentropy":      Ck_ee_sum  / n_cam,
        "C_omnivariance":      Ck_ov_sum  / n_cam,
        "C_voxel_entropy":     Ck_ve_sum  / n_cam,
        "C_spectral_energy":   Ck_se_sum  / n_cam,
        "C_eigenentropy_pos":  Ck_ee_pos_sum / n_cam,
        "C_omnivariance_pos":  Ck_ov_pos_sum / n_cam,
        "C_voxel_entropy_pos": Ck_ve_pos_sum / n_cam,
        "C_spectral_pos":      Ck_se_pos_sum / n_cam,
    }

    # --- OLS residual ---
    valid = (n_gs >= uc.COMPLEXITY_SPARSE_GUARD) & (mean_mse > 0)
    n_valid = valid.sum()
    print(f"\nvalid tiles: {n_valid} / {num_tiles}")

    log_mse = np.log(mean_mse[valid].astype(np.float64))
    log_ngs = np.log(n_gs[valid].astype(np.float64))
    A = np.column_stack([np.ones_like(log_ngs), log_ngs])
    coefs, _, _, _ = np.linalg.lstsq(A, log_mse, rcond=None)
    fitted = A @ coefs
    eps_k = log_mse - fitted
    print(f"OLS β0={coefs[0]:.3f} β1={coefs[1]:.3f}  R²={1 - np.var(eps_k)/np.var(log_mse):.3f}\n")

    # --- Compute and rank correlations ---
    rows = []
    for name, X in candidates.items():
        x = X[valid]
        if np.all(x == x[0]):   # constant → ρ undefined
            rows.append((name, float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        rho_raw,   p_raw   = scipy.stats.spearmanr(x, log_mse)
        rho_resid, p_resid = scipy.stats.spearmanr(x, eps_k)
        rows.append((name, rho_raw, p_raw, rho_resid, p_resid))

    rows.sort(key=lambda r: -abs(r[3]) if not np.isnan(r[3]) else -1)

    header = f"{'candidate':<22}  {'ρ_raw':>7}  {'p_raw':>9}  {'ρ_resid':>8}  {'p_resid':>9}  gate"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, r_raw, p_raw, r_res, p_res in rows:
        gate = "PASS" if abs(r_res) > 0.3 else "fail"
        if np.isnan(r_res):
            print(f"  {name:<22}  {'const':>7}  {'':>9}  {'':>8}  {'':>9}  {gate}")
        else:
            print(f"  {name:<22}  {r_raw:+.4f}  {p_raw:9.3e}  {r_res:+.8f}  {p_res:9.3e}  {gate}")
    print(sep)

    result = {
        "ols_beta0": float(coefs[0]), "ols_beta1": float(coefs[1]),
        "n_valid_tiles": int(n_valid),
        "camera_indices": cam_indices,
        "candidates": [
            {"name": name, "rho_raw": float(r_raw), "pval_raw": float(p_raw),
             "rho_resid": float(r_res), "pval_resid": float(p_res)}
            for name, r_raw, p_raw, r_res, p_res in rows
        ],
    }
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.oracle).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diag_all_candidates.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
