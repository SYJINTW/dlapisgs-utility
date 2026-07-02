"""Per-camera top-k precision + marginal-key Spearman rho, 4 schemes.

Open item 6 (PLAN.md): does a scheme's aggregate marginal-key rank
correlation predict its greedy tile_partial selection quality? Real scenes
have as few as 3-60 visible tiles/cam -- at that scale, PSNR is decided by
whether the top 1-3 tiles by true value get selected first, not by the
median rank correlation over the whole visible set. This measures both,
per camera, for the same 4 schemes used in exp2's canonical comparison:
vd_lod (baseline), vd_lod_w (heuristic, ours), v_lod_w (heuristic, no
distance), ml (lgbm, ours).

All 4 schemes are scored via selection_core.compute_raw_scores -- the same
function test_utility_inmem.py calls -- so no scheme math is reimplemented
here. Marginal key = raw_score / (n_gs_per_tile * bytes_per_gaussian),
matching selection_core.sort_tiles's marginal branch exactly.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n gaussian_splatting python \\
        experiments/0702/diag_topk_precision.py \\
        --scene chair \\
        --ply exp-dataset/chair/point_cloud.ply \\
        --oracle-npz output/oracle/8/eval/chair/oracle_dq.npz \\
        --camera-trace exp-dataset/chair/sparse_views_eval.json \\
        --ml-model-dir output/ml_models/8/chair/AC \\
        --out-dir output/diag_topk/chair \\
        [--camera-indices 0 1 2 3 4] [--img-w 1600] [--img-h 1600] \\
        [--topk 1 3 5]
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
import selection_core as sc  # noqa: E402
from ml import features as ml_features  # noqa: E402
from ml import predict as ml_predict  # noqa: E402

SCHEMES = ["vd_lod", "vd_lod_w", "v_lod_w", "ml"]


def _iqr(arr):
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--oracle-npz", required=True,
                     help="oracle_dq.npz with tiling (min/max_corners, index_offsets, "
                          "flat_indices) + n_gs_per_tile + mse + camera_indices bundled.")
    ap.add_argument("--camera-trace", required=True)
    ap.add_argument("--ml-model-dir", required=True, help="dir with lgbm.pkl + feature_names.json")
    ap.add_argument("--camera-indices", nargs="+", type=int, default=None)
    ap.add_argument("--img-w", type=int, default=1600)
    ap.add_argument("--img-h", type=int, default=1600)
    ap.add_argument("--topk", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.scene}] device={device}")

    # --- Oracle npz: tiling + labels bundled together ---
    oracle = np.load(args.oracle_npz, allow_pickle=False)
    min_corners = oracle["min_corners"].astype(np.float32)
    max_corners = oracle["max_corners"].astype(np.float32)
    index_offsets = oracle["index_offsets"].astype(np.int64)
    flat_indices = oracle["flat_indices"].astype(np.int64)
    n_gs_per_tile = oracle["n_gs_per_tile"].astype(np.int64)
    num_tiles = len(index_offsets) - 1
    cam_ids = oracle["camera_indices"].astype(np.int32)
    mse_loo = oracle["mse"].astype(np.float64)          # (N_cams, N_tiles)
    cam_idx_to_row = {int(c): i for i, c in enumerate(cam_ids)}
    print(f"[{args.scene}] tiles={num_tiles} oracle_cams={len(cam_ids)}")

    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers_t = (min_corners_t + max_corners_t) / 2.0
    tile_centers_np = tile_centers_t.cpu().numpy()
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)

    # --- PLY + bytes_per_gaussian ---
    print(f"[{args.scene}] loading PLY: {args.ply}")
    gs = io_3dgs.GaussianModelV2(args.ply)
    bpg = sc.bytes_per_gaussian(gs)
    byte_k = (n_gs_per_tile * bpg).astype(np.float64)
    print(f"[{args.scene}] bytes_per_gaussian={bpg}")
    gs_xyz = np.stack([gs.data["x"]["data"], gs.data["y"]["data"], gs.data["z"]["data"]], axis=1)
    gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)

    # --- ML model + static features (loaded once, reused every camera) ---
    model_dir = Path(args.ml_model_dir)
    feature_names = json.loads((model_dir / "feature_names.json").read_text())
    include_b = any(n in set(ml_features.GROUP_B_NAMES) for n in feature_names)
    if include_b:
        raise NotImplementedError("Group B (EWA) features not wired in this diagnostic; "
                                   "AC ablation (no Group B) expected.")
    static_feats = ml_features.load_feature_cache(model_dir / "feature_cache.npz", index_offsets)
    ml_model = ml_predict.load_model(str(model_dir), "lgbm", expected_n_gs=len(gs.data["x"]["data"]))

    # --- Cameras ---
    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
        args.camera_trace, args.img_w, args.img_h)
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    if args.camera_indices is None:
        cam_indices = sorted(cam_idx_to_row.keys())
    else:
        cam_indices = sorted(set(args.camera_indices) & set(cam_idx_to_row.keys()))
    print(f"[{args.scene}] cameras={len(cam_indices)}")

    per_scheme = {s: {"rho": [], "top1_hit": [], "rank_true_best": []} for s in SCHEMES}
    for k in args.topk:
        for s in SCHEMES:
            per_scheme[s][f"top{k}_prec"] = []

    csv_rows = []  # scene,scheme,camera,n_visible,rho,top1_hit,rank_true_best,top{k}_prec...

    import math as _math
    for ci in cam_indices:
        cam = cameras[ci]
        row = cam_idx_to_row[ci]

        visibility_t = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device)
        distances_t = uc.calculate_distances(tile_centers_t, cam.camera_center.to(device))
        vis_np = visibility_t.float().cpu().numpy()
        dist_np = distances_t.cpu().numpy()

        w_gi = sc.compute_camera_weights(
            cam, gs.data["opacity"]["data"], gs.data["scale_0"]["data"],
            gs.data["scale_1"]["data"], gs.data["scale_2"]["data"],
            gs.data["rot_0"]["data"], gs.data["rot_1"]["data"],
            gs.data["rot_2"]["data"], gs.data["rot_3"]["data"], gs_xyz_t, device,
            "screen_area", args.img_w, args.img_h)
        W_k, N_k = uc.compute_tile_weights_and_counts(
            tile_index_offsets, tile_flat_indices, w_gi,
            w_norm="none", c_norm="none")

        cam_w2v = cam.world_view_transform.cpu().numpy()
        cam_center_np = cam.camera_center.cpu().numpy()
        cam_fwd = cam_w2v[:3, 2]
        group_a = ml_features.build_group_a(
            cam_center_np, cam_fwd,
            float(getattr(cam, "FoVx", _math.pi / 2)), float(getattr(cam, "FoVy", _math.pi / 2)),
            tile_centers_np, n_gs_per_tile.astype(np.float32), dist_np, vis_np)
        ml_kwargs = dict(model_dir=str(model_dir), model_type="lgbm",
                          static_features=static_feats, group_a=group_a, group_b=None,
                          feature_names=feature_names, model=ml_model)

        mask = (vis_np > 0) & (byte_k > 0) & (mse_loo[row] > 0)
        n_vis = int(mask.sum())
        if n_vis < 3:
            print(f"  [{args.scene}] cam {ci:03d} skip (visible={n_vis})")
            continue

        truth_per_byte = mse_loo[row][mask] / byte_k[mask]
        order_true = np.argsort(-truth_per_byte)  # local indices within masked array

        row_out = {"scene": args.scene, "camera": ci, "n_visible": n_vis}
        for scheme in SCHEMES:
            raw = sc.compute_raw_scores(
                scheme, oracle_data=None, camera_index=ci, n_tiles=num_tiles,
                visibility=visibility_t, distances=distances_t, num_lod=1,
                W_k=W_k, C_k=N_k,
                ml_predict_kwargs=ml_kwargs if scheme == "ml" else None,
            )
            raw_np = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
            marginal = raw_np[mask] / byte_k[mask]

            rho = float(scipy.stats.spearmanr(marginal, truth_per_byte).correlation)
            order_scheme = np.argsort(-marginal)

            top1_hit = bool(order_scheme[0] == order_true[0])
            rank_true_best = int(np.where(order_scheme == order_true[0])[0][0]) + 1  # 1-indexed

            per_scheme[scheme]["rho"].append(rho)
            per_scheme[scheme]["top1_hit"].append(float(top1_hit))
            per_scheme[scheme]["rank_true_best"].append(rank_true_best)

            row_out[f"{scheme}_rho"] = rho
            row_out[f"{scheme}_top1_hit"] = int(top1_hit)
            row_out[f"{scheme}_rank_true_best"] = rank_true_best

            for k in args.topk:
                kk = min(k, n_vis)
                prec = len(set(order_scheme[:kk]) & set(order_true[:kk])) / kk
                per_scheme[scheme][f"top{k}_prec"].append(prec)
                row_out[f"{scheme}_top{k}_prec"] = prec

        csv_rows.append(row_out)
        print(f"  [{args.scene}] cam {ci:03d} done vis={n_vis}")

    # --- Aggregate ---
    def _agg(vals):
        arr = np.array(vals, dtype=np.float64)
        return {"median": float(np.median(arr)), "mean": float(np.mean(arr)),
                "iqr": _iqr(arr), "n": len(arr)}

    summary = {"scene": args.scene, "n_cameras": len(csv_rows), "n_tiles": num_tiles,
               "topk": args.topk, "schemes": {}}
    header = f"\n{'scheme':<10}  {'median_rho':>10}  {'top1_hit%':>9}  {'rank_true_best_med':>18}"
    for k in args.topk:
        header += f"  {'top'+str(k)+'_prec':>10}"
    print(header)
    print("-" * len(header))
    for scheme in SCHEMES:
        d = per_scheme[scheme]
        s = {"rho": _agg(d["rho"]), "top1_hit": _agg(d["top1_hit"]),
             "rank_true_best": _agg(d["rank_true_best"])}
        line = (f"{scheme:<10}  {s['rho']['median']:>+10.4f}  "
                f"{100*s['top1_hit']['mean']:>8.1f}%  {s['rank_true_best']['median']:>18.1f}")
        for k in args.topk:
            s[f"top{k}_prec"] = _agg(d[f"top{k}_prec"])
            line += f"  {s[f'top{k}_prec']['median']:>10.4f}"
        print(line)
        summary["schemes"][scheme] = s

    (out_dir / "diag_topk.json").write_text(json.dumps(summary, indent=2))

    import csv as _csv
    csv_path = out_dir / "diag_topk_percam.csv"
    if csv_rows:
        with csv_path.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"\n[{args.scene}] saved {out_dir / 'diag_topk.json'} + {csv_path}")


if __name__ == "__main__":
    main()
