"""Selected-Gaussian-set Jaccard overlap vs oracle, at real exp2 budget levels.

Follow-up to diag_topk_precision.py: that measured marginal-key rank quality
(Spearman rho, top-k precision) and found ml_lgbm's real-scene ranking wasn't
collapsed -- yet exp2's actual selection PSNR on real scenes for `ml` is
catastrophic. Ranking metrics don't directly say what got SELECTED. This
measures that directly: for each (scene, camera, budget), the actual
selected Gaussian index set under `tile_partial` + greedy_key=marginal
(the real pipeline path, via selection_core.build_greedy_order /
select_at_budget -- not reimplemented), compared to oracle_loo's selected
set at the same budget via Jaccard = |A n B| / |A u B|.

W_k always uses w_mode=sum -- w_mode=mean was removed from the codebase
2026-07-02 (utility_calculation.py W_MODES) after discovering
experiments/0630/run_exp2.sh had hardcoded --w-mode mean, silently
overriding the sum canon fixed 2026-06-28; exp2's published numbers were
run under that stale mean setting and need a rerun under the fixed code
before being cited further (PLAN.md). gs_order: vd_lod uses "ply" (no
per-GS weight, matching run_exp2.sh's `run_one vd_lod ply`),
heuristic/ml/oracle use "weight" (matching run_exp2.sh's `run_one main
weight ...`).

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n gaussian_splatting python \\
        experiments/0702/diag_jaccard_overlap.py \\
        --scene chair \\
        --ply exp-dataset/chair/point_cloud.ply \\
        --oracle-npz output/oracle/8/eval/chair/oracle_dq.npz \\
        --camera-trace exp-dataset/chair/sparse_views_eval.json \\
        --ml-model-dir output/ml_models/8/chair/AC \\
        --out-dir output/diag_jaccard/chair \\
        [--camera-indices 0 1 2] [--budget-pct 10 25 40 55 70 85 99 100]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
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
GS_ORDER = {"vd_lod": "ply", "vd_lod_w": "weight", "v_lod_w": "weight",
            "ml": "weight", "oracle_loo": "weight"}


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = len(a) + len(b) - inter
    return float(inter) / float(union) if union > 0 else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--oracle-npz", required=True)
    ap.add_argument("--camera-trace", required=True)
    ap.add_argument("--ml-model-dir", required=True)
    ap.add_argument("--camera-indices", nargs="+", type=int, default=None)
    ap.add_argument("--img-w", type=int, default=1600)
    ap.add_argument("--img-h", type=int, default=1600)
    ap.add_argument("--budget-pct", nargs="+", type=float,
                     default=[10, 25, 40, 55, 70, 85, 99, 100])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.scene}] device={device}")

    oracle = np.load(args.oracle_npz, allow_pickle=False)
    min_corners = oracle["min_corners"].astype(np.float32)
    max_corners = oracle["max_corners"].astype(np.float32)
    index_offsets = oracle["index_offsets"].astype(np.int64)
    flat_indices = oracle["flat_indices"].astype(np.int64)
    n_gs_per_tile = oracle["n_gs_per_tile"].astype(np.int64)
    num_tiles = len(index_offsets) - 1
    cam_ids = oracle["camera_indices"].astype(np.int32)
    mse_loo = oracle["mse"].astype(np.float64)
    cam_idx_to_row = {int(c): i for i, c in enumerate(cam_ids)}
    oracle_data = {"mse_loo": mse_loo, "cam_idx_to_row": cam_idx_to_row}
    total_n_gs = int(index_offsets[-1])
    print(f"[{args.scene}] tiles={num_tiles} total_gs={total_n_gs} oracle_cams={len(cam_ids)}")

    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers_t = (min_corners_t + max_corners_t) / 2.0
    tile_centers_np = tile_centers_t.cpu().numpy()
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)

    print(f"[{args.scene}] loading PLY: {args.ply}")
    gs = io_3dgs.GaussianModelV2(args.ply)
    bpg = sc.bytes_per_gaussian(gs)
    budget_bytes_list = [round(p / 100.0 * total_n_gs * bpg) for p in args.budget_pct]
    print(f"[{args.scene}] bpg={bpg} budget_pct={args.budget_pct} -> budget_bytes={budget_bytes_list}")
    gs_xyz = np.stack([gs.data["x"]["data"], gs.data["y"]["data"], gs.data["z"]["data"]], axis=1)
    gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)

    model_dir = Path(args.ml_model_dir)
    feature_names = json.loads((model_dir / "feature_names.json").read_text())
    static_feats = ml_features.load_feature_cache(model_dir / "feature_cache.npz", index_offsets)
    ml_model = ml_predict.load_model(str(model_dir), "lgbm", expected_n_gs=len(gs.data["x"]["data"]))

    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
        args.camera_trace, args.img_w, args.img_h)
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    if args.camera_indices is None:
        cam_indices = sorted(cam_idx_to_row.keys())
    else:
        cam_indices = sorted(set(args.camera_indices) & set(cam_idx_to_row.keys()))
    print(f"[{args.scene}] cameras={len(cam_indices)}")

    rows = []
    per_scheme_budget = {(s, p): [] for s in SCHEMES for p in args.budget_pct}

    for ci in cam_indices:
        cam = cameras[ci]
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
            w_norm="sum", c_norm="sum")

        cam_w2v = cam.world_view_transform.cpu().numpy()
        cam_center_np = cam.camera_center.cpu().numpy()
        group_a = ml_features.build_group_a(
            cam_center_np, cam_w2v[:3, 2],
            float(getattr(cam, "FoVx", math.pi / 2)), float(getattr(cam, "FoVy", math.pi / 2)),
            tile_centers_np, n_gs_per_tile.astype(np.float32), dist_np, vis_np)
        ml_kwargs = dict(model_dir=str(model_dir), model_type="lgbm",
                          static_features=static_feats, group_a=group_a, group_b=None,
                          feature_names=feature_names, model=ml_model)

        # oracle_loo reference selection, once per (camera, budget)
        oracle_raw = sc.compute_raw_scores(
            "oracle_loo", oracle_data=oracle_data, camera_index=ci, n_tiles=num_tiles,
            visibility=visibility_t, distances=distances_t, num_lod=1, W_k=W_k, C_k=N_k)
        if oracle_raw is None:
            print(f"  [{args.scene}] cam {ci:03d} skip (no oracle row)")
            continue
        oracle_pairs = sc.sort_tiles(oracle_raw, n_gs_per_tile, bpg, greedy_key="marginal")

        scheme_raw = {}
        for scheme in SCHEMES:
            raw = sc.compute_raw_scores(
                scheme, oracle_data=None, camera_index=ci, n_tiles=num_tiles,
                visibility=visibility_t, distances=distances_t, num_lod=1,
                W_k=W_k, C_k=N_k,
                ml_predict_kwargs=ml_kwargs if scheme == "ml" else None)
            scheme_raw[scheme] = sc.sort_tiles(raw, n_gs_per_tile, bpg, greedy_key="marginal")

        # The sort inside build_greedy_order doesn't depend on budget, only the final
        # truncation does -- compute the FULL scene ordering once per (camera, scheme)
        # at max_budget_bytes=full scene size, then slice per budget. Calling this once
        # per budget level (8x) instead of once total was an 8x redundant GPU sort.
        full_scene_bytes = total_n_gs * bpg
        oracle_ordered_full = sc.build_greedy_order(
            "tile_partial", "oracle_loo", oracle_pairs, visibility_t,
            tile_index_offsets, tile_flat_indices, w_gi, bpg, full_scene_bytes,
            gs_order=GS_ORDER["oracle_loo"])[0]
        scheme_ordered_full = {}
        for scheme in SCHEMES:
            scheme_ordered_full[scheme] = sc.build_greedy_order(
                "tile_partial", scheme, scheme_raw[scheme], visibility_t,
                tile_index_offsets, tile_flat_indices, w_gi, bpg, full_scene_bytes,
                gs_order=GS_ORDER[scheme])[0]

        for pct, bbytes in zip(args.budget_pct, budget_bytes_list):
            count = bbytes // bpg
            oracle_sel = np.sort(oracle_ordered_full[:count])

            row = {"scene": args.scene, "camera": ci, "budget_pct": pct, "n_selected": len(oracle_sel)}
            for scheme in SCHEMES:
                sel = np.sort(scheme_ordered_full[scheme][:count])
                jac = _jaccard(sel, oracle_sel)
                row[f"{scheme}_jaccard"] = jac
                per_scheme_budget[(scheme, pct)].append(jac)

            rows.append(row)
        print(f"  [{args.scene}] cam {ci:03d} done")

    summary = {"scene": args.scene, "n_cameras": len(cam_indices), "budget_pct": args.budget_pct,
               "schemes": {}}
    header = f"\n{'scheme':<10}" + "".join(f"{p:>8.0f}%" for p in args.budget_pct)
    print(header)
    print("-" * len(header))
    for scheme in SCHEMES:
        meds = [float(np.median(per_scheme_budget[(scheme, p)])) if per_scheme_budget[(scheme, p)] else float("nan")
                for p in args.budget_pct]
        summary["schemes"][scheme] = dict(zip(args.budget_pct, meds))
        print(f"{scheme:<10}" + "".join(f"{m:>9.3f}" for m in meds))

    (out_dir / "diag_jaccard.json").write_text(json.dumps(summary, indent=2))
    import csv as _csv
    csv_path = out_dir / "diag_jaccard_percam.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n[{args.scene}] saved {out_dir / 'diag_jaccard.json'} + {csv_path}")


if __name__ == "__main__":
    main()
