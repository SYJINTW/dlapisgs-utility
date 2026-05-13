#!/usr/bin/env python3
"""Interactive headless debugger for the 3DGS tile-utility pipeline.

Runs as a Plotly Dash app. Launch on the server, port-forward, open in a
local browser:

    conda run -n gsquic python experiments/debug_viewer_app.py \
        --ply /path/to/point_cloud.ply \
        --camera-trace /path/to/trace.json \
        --grid-shape 4 4 4 \
        --port 8050

    # then on your laptop:
    ssh -L 8050:localhost:8050 <server>
    # open http://localhost:8050
    
    example:
    
conda run -n gsquic python experiments/debug_viewer_app.py \
        --ply /mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply \
        --camera-trace /mnt/data1/samk/gs-quic/cs5262_tile_quic/Frustum-for-3DGS/sample_data/camera_trace/trace1.json\
        --grid-shape 4 4 4 \
        --port 8050 --debug

Widgets:
  - Camera-index slider               -> change viewport
  - Weight-mode dropdown              -> {legacy, volume, volume_over_d2, screen_area}
  - W_k and C_k normalization dropdowns
  - Subsample fraction slider         -> how many per-Gaussian dots to draw
  - Overlay checklist                 -> turn panels on/off

Panels (Plotly subplot grid):
  (0,0) Tile utility (current scheme)
  (0,1) Per-tile #GS heatmap (raw C_k)
  (0,2) Per-tile W_k heatmap (under chosen normalization)
  (1,0) Per-Gaussian density (subsampled, colored by w_gi)
  (1,1) Camera/scene status text + tile AABB top-down view
  (1,2) (reserved for future overlays)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT / "Frustum-for-3DGS"))
sys.path.insert(0, str(WORKSPACE_ROOT / "GGSP"))
sys.path.insert(0, str(WORKSPACE_ROOT / "GS-Interface"))
sys.path.insert(0, str(WORKSPACE_ROOT / "dlapisgs-utility"))

import visibility_AABB_pytorch  # type: ignore  # noqa: E402
import tiling as ggsp_tiling  # type: ignore  # noqa: E402
import io_3dgs  # type: ignore  # noqa: E402  # pyright: ignore[reportMissingImports]
import utility_calculation as uc  # noqa: E402

import dash  # noqa: E402
from dash import dcc, html, Input, Output, State, ctx, dash_table  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level state. Populated once at startup; reused by Dash callbacks.
# ---------------------------------------------------------------------------
STATE: dict = {}


def _project_points(points_world: np.ndarray, world_view: np.ndarray, proj: np.ndarray):
    """Project Nx3 world points -> NDC xy + in_front flag.

    Uses the same 3DGS convention as experiments/debug_viewport.py:
        clip = [xyz, 1] @ world_view @ proj
    """
    N = len(points_world)
    ones = np.ones((N, 1), dtype=np.float32)
    p4 = np.concatenate([points_world.astype(np.float32), ones], axis=1)
    cam = p4 @ world_view
    clip = cam @ proj
    w = clip[:, 3:4]
    in_front = w[:, 0] > 0
    w_safe = np.where(np.abs(w) < 1e-6, 1e-6, w)
    ndc = clip[:, :2] / w_safe
    return ndc, in_front


def _normalize(v: np.ndarray):
    n = np.linalg.norm(v)
    return v / n if n >= 1e-8 else None


def _estimate_camera_frame(camera_center, tile_centers, visibility):
    """Estimate camera forward/right/up from mean of visible tile centers."""
    visible_mask = np.asarray(visibility).astype(bool)
    if visible_mask.size == 0 or not np.any(visible_mask):
        return None, None, None
    vis_centers = np.asarray(tile_centers)[visible_mask]
    forward = _normalize(np.mean(vis_centers, axis=0) - camera_center)
    if forward is None:
        return None, None, None
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(forward, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _normalize(np.cross(forward, world_up))
    if right is None:
        return None, None, None
    up = _normalize(np.cross(right, forward))
    return forward, right, up


def _get_aabb_edges(mn, mx):
    """12 edges of an AABB as list of (p1, p2) vertex pairs."""
    v = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    return [(v[i], v[j]) for i, j in
            [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]]


def _frustum_lines(cam_center, tile_centers, vis_np, min_corners, max_corners):
    """Build frustum line segments (None-separated x/y/z lists) for Scatter3d."""
    forward, right, up = _estimate_camera_frame(cam_center, tile_centers, vis_np)
    if forward is None:
        return None, None, None
    scene_diag = float(np.linalg.norm(np.max(max_corners, axis=0) - np.min(min_corners, axis=0)))
    if scene_diag < 1e-8:
        return None, None, None
    near_d = max(0.02 * scene_diag, 1e-3)
    far_d = max(0.22 * scene_diag, near_d * 3.0)
    vis_centers = tile_centers[vis_np]
    tan_h, tan_v = 0.35, 0.28
    if vis_centers.shape[0] > 0:
        rel = vis_centers - cam_center
        proj_f = rel @ forward
        mask = proj_f > 1e-6
        rel, proj_f = rel[mask], proj_f[mask]
        if rel.shape[0] > 0:
            tan_h = float(np.clip(np.percentile(np.abs((rel @ right) / proj_f), 90), 0.08, 1.2))
            tan_v = float(np.clip(np.percentile(np.abs((rel @ up) / proj_f), 90), 0.08, 1.0))
    near_c, far_c = cam_center + forward * near_d, cam_center + forward * far_d
    near_pts = np.array([
        near_c + right * near_d * tan_h + up * near_d * tan_v,
        near_c - right * near_d * tan_h + up * near_d * tan_v,
        near_c - right * near_d * tan_h - up * near_d * tan_v,
        near_c + right * near_d * tan_h - up * near_d * tan_v,
    ])
    far_pts = np.array([
        far_c + right * far_d * tan_h + up * far_d * tan_v,
        far_c - right * far_d * tan_h + up * far_d * tan_v,
        far_c - right * far_d * tan_h - up * far_d * tan_v,
        far_c + right * far_d * tan_h - up * far_d * tan_v,
    ])
    xs, ys, zs = [], [], []
    for i in range(4):  # camera to far corners
        xs += [cam_center[0], far_pts[i, 0], None]
        ys += [cam_center[1], far_pts[i, 1], None]
        zs += [cam_center[2], far_pts[i, 2], None]
    for ring in (near_pts, far_pts):  # near and far rectangles
        for i in range(4):
            j = (i + 1) % 4
            xs += [ring[i, 0], ring[j, 0], None]
            ys += [ring[i, 1], ring[j, 1], None]
            zs += [ring[i, 2], ring[j, 2], None]
    return xs, ys, zs


def _load_state(ply_path: Path, camera_trace: Path, grid_shape: tuple[int, int, int],
                img_w: int, img_h: int, device: str) -> None:
    """Populate STATE with everything that doesn't change between camera/widget updates."""
    logger.info("Loading PLY: {}", ply_path)
    gs = io_3dgs.GaussianModelV2(str(ply_path))

    logger.info("Tiling with grid_shape={}", grid_shape)
    tile_aabbs, tile_indices, scene_min, scene_max = ggsp_tiling.tiling_uniform_layered_gs(
        [gs], grid_shape=tuple(grid_shape)
    )
    sorted_tile_keys = sorted(tile_aabbs.keys())
    min_corners = np.asarray([tile_aabbs[k]["min_corner"] for k in sorted_tile_keys], dtype=np.float32)
    max_corners = np.asarray([tile_aabbs[k]["max_corner"] for k in sorted_tile_keys], dtype=np.float32)
    flat_indices_list = [tile_indices[k][0].astype(np.int64) for k in sorted_tile_keys]
    index_offsets = np.zeros(len(sorted_tile_keys) + 1, dtype=np.int64)
    for i, arr in enumerate(flat_indices_list):
        index_offsets[i + 1] = index_offsets[i] + len(arr)
    flat_indices = (np.concatenate(flat_indices_list)
                    if flat_indices_list else np.empty((0,), dtype=np.int64))

    logger.info("Loading camera trace: {}", camera_trace)
    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(str(camera_trace), img_w, img_h)
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)

    # GS attribute tensors on device.
    opacity = torch.as_tensor(gs.data["opacity"]["data"], dtype=torch.float32, device=device)
    scale_0 = torch.as_tensor(gs.data["scale_0"]["data"], dtype=torch.float32, device=device)
    scale_1 = torch.as_tensor(gs.data["scale_1"]["data"], dtype=torch.float32, device=device)
    scale_2 = torch.as_tensor(gs.data["scale_2"]["data"], dtype=torch.float32, device=device)
    rot_0 = torch.as_tensor(gs.data["rot_0"]["data"], dtype=torch.float32, device=device) if "rot_0" in gs.data else None
    rot_1 = torch.as_tensor(gs.data["rot_1"]["data"], dtype=torch.float32, device=device) if "rot_1" in gs.data else None
    rot_2 = torch.as_tensor(gs.data["rot_2"]["data"], dtype=torch.float32, device=device) if "rot_2" in gs.data else None
    rot_3 = torch.as_tensor(gs.data["rot_3"]["data"], dtype=torch.float32, device=device) if "rot_3" in gs.data else None
    gs_xyz = np.stack([gs.data["x"]["data"], gs.data["y"]["data"], gs.data["z"]["data"]], axis=1)
    gs_xyz_t = torch.as_tensor(gs_xyz, dtype=torch.float32, device=device)

    min_corners_t = torch.as_tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.as_tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers = (min_corners_t + max_corners_t) / 2.0
    tile_index_offsets_t = torch.as_tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices_t = torch.as_tensor(flat_indices, dtype=torch.long, device=device)

    STATE.update(
        device=device,
        img_w=img_w, img_h=img_h,
        cameras=cameras,
        opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
        rot_0=rot_0, rot_1=rot_1, rot_2=rot_2, rot_3=rot_3,
        gs_xyz=gs_xyz, gs_xyz_t=gs_xyz_t,
        min_corners=min_corners, max_corners=max_corners,
        min_corners_t=min_corners_t, max_corners_t=max_corners_t,
        tile_centers=tile_centers,
        tile_index_offsets=tile_index_offsets_t,
        tile_flat_indices=tile_flat_indices_t,
    )
    logger.success("State ready: {} cameras, {} tiles, {} Gaussians",
                   len(cameras), len(sorted_tile_keys), len(gs_xyz))


def _compute_per_camera(cam, weight_mode: str, w_norm: str, c_norm: str, gamma: float = 1.0):
    """Compute everything that changes per (camera, weight_mode, normalization)."""
    s = STATE
    device = s["device"]
    cam_center = cam.camera_center.to(device)

    distances = uc.calculate_distances(s["tile_centers"], cam_center)
    visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
        s["min_corners_t"], s["max_corners_t"], cam, device=device
    )

    if weight_mode == "det_gamma_over_d2":
        w_gi = uc.compute_gaussian_weights(
            s["opacity"], s["scale_0"], s["scale_1"], s["scale_2"],
            gamma=gamma, xyz=s["gs_xyz_t"], cam_center=cam_center,
        ).to(device)
    else:
        kw = dict(
            opacity=s["opacity"], scale_0=s["scale_0"], scale_1=s["scale_1"], scale_2=s["scale_2"],
            xyz=s["gs_xyz_t"], cam_center=cam_center,
        )
        if weight_mode == "screen_area":
            if any(r is None for r in (s["rot_0"], s["rot_1"], s["rot_2"], s["rot_3"])):
                raise RuntimeError("screen_area requires rot_0..rot_3 in the PLY")
            kw.update(
                rot_0=s["rot_0"], rot_1=s["rot_1"], rot_2=s["rot_2"], rot_3=s["rot_3"],
                world_view=cam.world_view_transform, proj=cam.projection_matrix,
                img_w=s["img_w"], img_h=s["img_h"],
                fov_x=getattr(cam, "FoVx", None), fov_y=getattr(cam, "FoVy", None),
            )
        w_gi = uc.compute_gaussian_weights_v2(weight_mode, **kw).to(device)

    W_k, C_k = uc.compute_tile_weights_and_counts(
        s["tile_index_offsets"], s["tile_flat_indices"], w_gi,
        w_norm=w_norm, c_norm=c_norm,
    )
    sizes = (s["tile_index_offsets"][1:] - s["tile_index_offsets"][:-1]).to(torch.float32)
    base = (torch.where(visibility, 1.0, uc.INVISIBLE_PRIORITY_EPS)
            / (distances + uc.DISTANCE_EPS))
    u = (base * W_k * C_k).detach().cpu().numpy()
    return dict(
        distances=distances, visibility=visibility, w_gi=w_gi,
        W_k=W_k, C_k=C_k, C_k_raw=sizes, u=u,
    )


def _build_figure(cam, cam_idx, fields, weight_mode, w_norm, c_norm,
                  subsample_frac: float, overlays: list[str], scheme: str = "vd_lod_w_c") -> go.Figure:
    s = STATE
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "scene"}],
        ],
        subplot_titles=[
            f"Tile utility U (scheme={scheme})",
            "Per-tile #GS (raw count)",
            f"Per-tile W_k (w_norm={w_norm})",
            f"Per-GS weight density (subsampled, {weight_mode})",
            "Tiles in NDC (red=visible, gray=culled)",
            "3D scene — tile AABBs, camera, frustum",
        ],
        horizontal_spacing=0.14, vertical_spacing=0.22,
    )

    wv = cam.world_view_transform.detach().cpu().numpy()
    pj = cam.projection_matrix.detach().cpu().numpy()
    cam_center_np = cam.camera_center.cpu().numpy().ravel().astype(np.float64)

    tile_centers = s["tile_centers"].detach().cpu().numpy()
    ndc_tiles, tiles_in_front = _project_points(tile_centers, wv, pj)
    vis_np = fields["visibility"].detach().cpu().numpy()

    # Project camera center into the same projected space for the reference star
    ndc_cam, _ = _project_points(cam_center_np.reshape(1, 3), wv, pj)
    cam_px, cam_py = float(ndc_cam[0, 0]), float(ndc_cam[0, 1])
    cam_proj_valid = np.isfinite(cam_px) and np.isfinite(cam_py)

    u = fields["u"]

    # Colorbars: use coloraxis references so positions are layout-level, not trace-level.
    # With horizontal_spacing=0.14, 3 cols: col domains ≈ [0,0.24], [0.38,0.62], [0.76,1.0]
    # Row domains (vertical_spacing=0.22): row1 ≈ [0.61,1.0], row2 ≈ [0.0,0.39]
    fig.update_layout(
        coloraxis1=dict(colorscale="Viridis",
                        colorbar=dict(title="U",    x=0.26, y=0.80, len=0.38, thickness=12)),
        coloraxis2=dict(colorscale="Cividis",
                        colorbar=dict(title="#GS",  x=0.64, y=0.80, len=0.38, thickness=12)),
        coloraxis3=dict(colorscale="Magma",
                        colorbar=dict(title="W_k",  x=1.01, y=0.80, len=0.38, thickness=12)),
        coloraxis4=dict(colorscale="Plasma",
                        colorbar=dict(title="w_gi", x=0.26, y=0.20, len=0.38, thickness=12)),
    )

    # ---- panel (0,0): tile utility ----
    if "utility" in overlays:
        u_disp = np.log1p(u) / (np.log1p(u.max()) + 1e-12)  # log1p to spread skewed distribution
        fig.add_trace(
            go.Scatter(
                x=ndc_tiles[:, 0], y=ndc_tiles[:, 1], mode="markers",
                marker=dict(size=18, color=u_disp, coloraxis="coloraxis1"),
                text=[f"tile {i}: U={ui:.3g}" for i, ui in enumerate(u)],
                hoverinfo="text", showlegend=False,
            ), row=1, col=1,
        )

    # ---- panel (0,1): per-tile #GS heatmap ----
    if "ck" in overlays:
        c_raw = fields["C_k_raw"].detach().cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x=ndc_tiles[:, 0], y=ndc_tiles[:, 1], mode="markers",
                marker=dict(size=18, color=c_raw, coloraxis="coloraxis2"),
                text=[f"tile {i}: {int(c)} GS" for i, c in enumerate(c_raw)],
                hoverinfo="text", showlegend=False,
            ), row=1, col=2,
        )

    # ---- panel (0,2): per-tile W_k ----
    if "wk" in overlays:
        w_np = fields["W_k"].detach().cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x=ndc_tiles[:, 0], y=ndc_tiles[:, 1], mode="markers",
                marker=dict(size=18, color=w_np, coloraxis="coloraxis3"),
                text=[f"tile {i}: W_k={wi:.3g}" for i, wi in enumerate(w_np)],
                hoverinfo="text", showlegend=False,
            ), row=1, col=3,
        )

    # ---- panel (1,0): per-Gaussian density (subsample) ----
    if "gs_density" in overlays:
        n_gs = len(s["gs_xyz"])
        n_take = max(100, int(n_gs * subsample_frac))
        rng = np.random.default_rng(0)
        idx = rng.choice(n_gs, size=min(n_take, n_gs), replace=False)
        sub_xyz = s["gs_xyz"][idx]
        ndc_gs, gs_in_front = _project_points(sub_xyz, wv, pj)
        w_gi_np = fields["w_gi"][torch.as_tensor(idx, device=s["device"])].detach().cpu().numpy()
        mask = gs_in_front
        w_gi_sub = w_gi_np[mask]
        w_gi_disp = np.log1p(w_gi_sub) / (np.log1p(w_gi_sub.max()) + 1e-12)  # log1p to spread skewed distribution
        fig.add_trace(
            go.Scatter(
                x=ndc_gs[mask, 0], y=ndc_gs[mask, 1], mode="markers",
                marker=dict(size=2, color=w_gi_disp, coloraxis="coloraxis4", opacity=0.5),
                hoverinfo="skip", showlegend=False,
            ), row=2, col=1,
        )

    # ---- panel (1,1): tiles in NDC colored by visibility ----
    if "tiles_vis" in overlays:
        colors = np.where(vis_np, 1.0, 0.0)
        fig.add_trace(
            go.Scatter(
                x=ndc_tiles[:, 0], y=ndc_tiles[:, 1], mode="markers",
                marker=dict(size=18, color=colors,
                            colorscale=[[0, "lightgray"], [1, "crimson"]], showscale=False),
                text=[f"tile {i}: vis={bool(v)}" for i, v in enumerate(vis_np)],
                hoverinfo="text", showlegend=False,
            ), row=2, col=2,
        )

    # Camera star marker in all 5 projected 2D panels
    if cam_proj_valid:
        for _r, _c in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]:
            fig.add_trace(go.Scatter(
                x=[cam_px], y=[cam_py], mode="markers",
                marker=dict(symbol="star", size=16, color="gold",
                            line=dict(color="darkorange", width=1.5)),
                hovertext="camera", hoverinfo="text", showlegend=False,
            ), row=_r, col=_c)

    # ---- panel (2,3): 3D scene — tile AABBs, camera, frustum ----
    vis_x, vis_y, vis_z = [], [], []
    cul_x, cul_y, cul_z = [], [], []
    for i, (mn, mx) in enumerate(zip(s["min_corners"], s["max_corners"])):
        tx, ty, tz = (vis_x, vis_y, vis_z) if vis_np[i] else (cul_x, cul_y, cul_z)
        for p1, p2 in _get_aabb_edges(mn, mx):
            tx += [p1[0], p2[0], None]
            ty += [p1[1], p2[1], None]
            tz += [p1[2], p2[2], None]
    if vis_x:
        fig.add_trace(go.Scatter3d(
            x=vis_x, y=vis_y, z=vis_z, mode="lines",
            line=dict(color="limegreen", width=2), hoverinfo="skip", showlegend=False,
        ), row=2, col=3)
    if cul_x:
        fig.add_trace(go.Scatter3d(
            x=cul_x, y=cul_y, z=cul_z, mode="lines",
            line=dict(color="crimson", width=1), opacity=0.35, hoverinfo="skip", showlegend=False,
        ), row=2, col=3)
    fig.add_trace(go.Scatter3d(
        x=[cam_center_np[0]], y=[cam_center_np[1]], z=[cam_center_np[2]],
        mode="markers", marker=dict(size=8, color="gold", symbol="diamond"),
        hoverinfo="skip", showlegend=False,
    ), row=2, col=3)
    fx, fy, fz = _frustum_lines(cam_center_np, tile_centers, vis_np, s["min_corners"], s["max_corners"])
    if fx is not None:
        fig.add_trace(go.Scatter3d(
            x=fx, y=fy, z=fz, mode="lines",
            line=dict(color="deepskyblue", width=2), hoverinfo="skip", showlegend=False,
        ), row=2, col=3)
        forward, _, _ = _estimate_camera_frame(cam_center_np, tile_centers, vis_np)
        if forward is not None:
            scene_diag = float(np.linalg.norm(np.max(s["max_corners"], axis=0) - np.min(s["min_corners"], axis=0)))
            tip = cam_center_np + forward * 0.12 * scene_diag
            fig.add_trace(go.Scatter3d(
                x=[cam_center_np[0], tip[0]], y=[cam_center_np[1], tip[1]], z=[cam_center_np[2], tip[2]],
                mode="lines+markers", line=dict(color="navy", width=4),
                marker=dict(size=[0, 5], color="navy"), hoverinfo="skip", showlegend=False,
            ), row=2, col=3)

    for r, c in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]:
        fig.update_xaxes(showgrid=False, zeroline=False, row=r, col=c)
        fig.update_yaxes(showgrid=False, zeroline=False, row=r, col=c)
    fig.update_layout(
        height=1000, margin=dict(l=30, r=30, t=30, b=0),
        scene=dict(aspectmode="data"),
    )
    return fig


def _build_table(fields: dict) -> tuple[list[dict], list[dict]]:
    """Build per-tile stats table sorted by utility descending."""
    s = STATE
    sizes = (s["tile_index_offsets"][1:] - s["tile_index_offsets"][:-1]).cpu().numpy()
    vis_np = fields["visibility"].cpu().numpy()
    dist_np = fields["distances"].cpu().numpy()
    w_np = fields["W_k"].cpu().numpy()
    c_np = fields["C_k"].cpu().numpy()
    u_np = fields["u"]
    order = np.argsort(u_np)[::-1]
    rows = [
        {
            "rank": int(rank + 1),
            "tile": int(i),
            "visible": "yes" if vis_np[i] else "no",
            "dist": f"{dist_np[i]:.2f}",
            "gs": int(sizes[i]),
            "W_k": f"{w_np[i]:.4f}",
            "C_k": f"{c_np[i]:.4f}",
            "U": f"{u_np[i]:.4g}",
        }
        for rank, i in enumerate(order)
    ]
    columns = [
        {"name": "Rank", "id": "rank"},
        {"name": "Tile", "id": "tile"},
        {"name": "Visible", "id": "visible"},
        {"name": "Dist", "id": "dist"},
        {"name": "#GS", "id": "gs"},
        {"name": "W_k", "id": "W_k"},
        {"name": "C_k", "id": "C_k"},
        {"name": "U", "id": "U"},
    ]
    return rows, columns


def _fmt_count(n: int) -> str:
    """Format a large integer compactly: 5998962 → '6.0M', 2e9 → '2.0G'."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


_WEIGHT_FORMULA = {
    "det_gamma_over_d2": "sigmoid(o)·det(Σ)^γ / d²",
    "volume_over_d2":    "sigmoid(o)·vol(Σ) / d²",
    "volume":            "sigmoid(o)·vol(Σ)",
    "screen_area":       "screen_area(Σ)",
}


def _build_dist_figure(fields: dict, weight_mode: str = "") -> go.Figure:
    """Histogram of U_k (per tile) and w(g_i) (all Gaussians — already computed)."""
    u_np = fields["u"]
    w_gi_all = fields["w_gi"].cpu().numpy()

    # log10-transform — both span many orders of magnitude
    w_gi_log = np.log10(w_gi_all + 1e-30)
    u_log = np.log10(u_np + 1e-30)

    formula = _WEIGHT_FORMULA.get(weight_mode, weight_mode)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"U_k — log₁₀ histogram ({len(u_np)} tiles)",
            f"w(g_i) = {formula} — log₁₀ histogram ({_fmt_count(len(w_gi_all))} GS)",
        ],
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Histogram(
        x=u_log, nbinsx=40, name="U_k",
        marker_color="steelblue", opacity=0.85,
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=w_gi_log, nbinsx=80, name="w(g_i)",
        marker_color="darkorange", opacity=0.85,
    ), row=1, col=2)

    fig.update_xaxes(title_text="log₁₀(U_k)", row=1, col=1)
    fig.update_xaxes(title_text="log₁₀(w(g_i))", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_yaxes(title_text="count", row=1, col=2)
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=50, b=40), showlegend=False)
    return fig



def _build_app() -> dash.Dash:
    app = dash.Dash(__name__)
    n_cams = len(STATE["cameras"])
    slider_marks = {i: str(i) for i in range(0, n_cams, max(1, n_cams // 10))}

    _MODAL_HIDDEN = {"display": "none"}
    _MODAL_SHOWN = {
        "display": "block", "position": "fixed", "top": 0, "left": 0,
        "width": "100%", "height": "100%", "backgroundColor": "rgba(0,0,0,0.55)",
        "zIndex": 1000, "overflowY": "auto",
    }
    _INFO_TEXT = """
LAUNCH
  conda run -n gsquic python experiments/debug_viewer_app.py \\
      --ply /path/to/point_cloud.ply \\
      --camera-trace /path/to/trace.json \\
      --grid-shape 4 4 4 \\
      --port 8050 --debug

  VSCode Simple Browser → http://127.0.0.1:8050
  --debug enables hot-reload (edits reflect on save, no restart).

PANELS (2×3 grid)
  (1,1) Tile utility U  — log₁₀-normalized for display, raw value in hover
  (1,2) Per-tile #GS    — raw Gaussian count per tile
  (1,3) Per-tile W_k    — aggregate Gaussian importance weight per tile
  (2,1) Per-GS density  — subsampled GS dots colored by w(g_i); fraction set by slider
  (2,2) Tile visibility — red = visible to this camera, gray = frustum-culled
  (2,3) 3D scene        — tile AABBs (green=visible, red=culled), gold=camera,
                          sky-blue=estimated frustum, navy arrow=view direction

  Gold star ★ in panels 1–5 marks the projected camera position.

UTILITY FORMULA
  U(k) = log(β·(ℓ+1)) · (v_k / d_k) · W_k · C_k

  v_k  — visibility (0/1 from frustum test)
  d_k  — camera→tile center distance
  W_k  — sum of w(g_i) over tile k, then normalized by w_norm
  C_k  — Gaussian count in tile k, normalized by c_norm
  ℓ    — LOD level (fixed at 1 in norm sweep; log factor is constant)

PER-GAUSSIAN WEIGHT w(g_i)  — set via weight_mode dropdown
  volume_over_d2    sigmoid(o) · exp(sx+sy+sz) / d²   ← default / recommended
  det_gamma_over_d2 sigmoid(o) · exp(2γ(sx+sy+sz)) / d²
  volume            sigmoid(o) · exp(sx+sy+sz)
  screen_area       projected screen-space area

NORMALIZATION (w_norm / c_norm)
  none   — identity (unbounded)
  max    — divide by max → [0, 1]
  minmax — (x−min)/(max−min) → [0, 1]
  log1p  — log1p(x)/log1p(max) → [0, 1]
  sum    — divide by sum → probability mass (default)

DISTRIBUTION HISTOGRAMS (below table)
  U_k histogram      — one entry per tile (~40 tiles); shows how utility is spread
  w(g_i) histogram   — all Gaussians; already computed when U_k is computed, no extra cost
  Both axes are log₁₀ to reveal structure across orders of magnitude.

TABLE
  Rows sorted by U descending. Visible tiles highlighted yellow.
  Scrollable — no pagination.
"""

    app.layout = html.Div([
        html.Div([
            html.H3("3DGS tile-utility debug viewer",
                    style={"display": "inline-block", "margin": 0, "marginRight": "10px"}),
            html.Button("ℹ how to use", id="info-btn", n_clicks=0, style={
                "fontSize": "13px", "padding": "3px 10px", "cursor": "pointer",
                "borderRadius": "4px", "border": "1px solid #aaa",
                "backgroundColor": "#f0f0f0", "verticalAlign": "middle",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Div(id="info-modal", style=_MODAL_HIDDEN, children=[
            html.Div([
                html.Div([
                    html.H4("How to use — 3DGS tile-utility debug viewer",
                            style={"marginTop": 0, "marginBottom": "12px"}),
                    html.Pre(_INFO_TEXT, style={
                        "fontFamily": "monospace", "fontSize": "12px",
                        "whiteSpace": "pre-wrap", "lineHeight": "1.6",
                        "backgroundColor": "#f8f8f8", "padding": "12px",
                        "borderRadius": "4px", "overflowY": "auto", "maxHeight": "60vh",
                    }),
                    html.Button("Close", id="info-close-btn", n_clicks=0, style={
                        "marginTop": "12px", "padding": "6px 20px", "cursor": "pointer",
                        "borderRadius": "4px", "border": "1px solid #aaa",
                        "backgroundColor": "#e0e0e0", "fontSize": "13px",
                    }),
                ], style={
                    "backgroundColor": "white", "borderRadius": "8px",
                    "padding": "24px", "maxWidth": "760px", "margin": "60px auto",
                    "boxShadow": "0 4px 24px rgba(0,0,0,0.25)",
                }),
            ]),
        ]),
        html.Div([
            html.Label("Camera index"),
            dcc.Slider(id="camera-slider", min=0, max=n_cams - 1, step=1, value=0,
                       marks=slider_marks, tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("weight_mode"),
                dcc.Dropdown(id="weight-mode", value="det_gamma_over_d2",
                             options=[{"label": m, "value": m} for m in uc.WEIGHT_MODES],
                             clearable=False, style={"width": "220px"}),
            ], style={"display": "inline-block", "marginRight": "16px"}),
            html.Div([
                html.Label("w_norm"),
                dcc.Dropdown(id="w-norm", value="sum",
                             options=[{"label": m, "value": m} for m in uc.NORM_MODES],
                             clearable=False, style={"width": "120px"}),
            ], style={"display": "inline-block", "marginRight": "16px"}),
            html.Div([
                html.Label("c_norm"),
                dcc.Dropdown(id="c-norm", value="sum",
                             options=[{"label": m, "value": m} for m in uc.NORM_MODES],
                             clearable=False, style={"width": "120px"}),
            ], style={"display": "inline-block", "marginRight": "16px"}),
            html.Div([
                html.Label("GS subsample fraction (panel 4 scatter only)"),
                dcc.Slider(id="subsample", min=0.001, max=0.05, step=0.001, value=0.01,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"display": "inline-block", "width": "260px", "verticalAlign": "top"}),
        ], style={"marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Overlays"),
                dcc.Checklist(
                    id="overlays",
                    options=[
                        {"label": " tile utility",           "value": "utility"},
                        {"label": " per-tile #GS",           "value": "ck"},
                        {"label": " per-tile W_k",           "value": "wk"},
                        {"label": " per-Gaussian density",   "value": "gs_density"},
                        {"label": " tile visibility",        "value": "tiles_vis"},
                    ],
                    value=["utility", "ck", "wk", "gs_density", "tiles_vis"],
                    inline=True,
                ),
            ], style={"display": "inline-block", "verticalAlign": "top", "marginRight": "24px"}),
            html.Div(id="status-text", style={
                "display": "inline-block", "verticalAlign": "top",
                "fontFamily": "monospace", "fontSize": "12px",
                "backgroundColor": "#f5f5f5", "padding": "6px 12px",
                "borderRadius": "4px", "lineHeight": "1.7",
            }),
        ], style={"marginBottom": "8px"}),
        dcc.Graph(id="main-figure"),
        html.H4("Per-tile stats — sorted by utility ↓", style={"marginTop": "16px"}),
        dcc.Graph(id="dist-figure"),
        dash_table.DataTable(
            id="tile-table",
            style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "480px"},
            style_cell={"textAlign": "right", "fontFamily": "monospace", "fontSize": "12px",
                        "padding": "4px 10px"},
            style_header={"fontWeight": "bold", "textAlign": "center", "position": "sticky", "top": 0,
                          "backgroundColor": "white", "zIndex": 1},
            style_data_conditional=[
                {"if": {"filter_query": '{visible} = "yes"'},
                 "backgroundColor": "#fef9ec"},
            ],
            page_action="none",
        ),
    ])

    @app.callback(
        Output("info-modal", "style"),
        Input("info-btn", "n_clicks"),
        Input("info-close-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_info_modal(_open, _close):
        return _MODAL_SHOWN if ctx.triggered_id == "info-btn" else _MODAL_HIDDEN

    @app.callback(
        Output("main-figure", "figure"),
        Output("tile-table", "data"),
        Output("tile-table", "columns"),
        Output("status-text", "children"),
        Output("dist-figure", "figure"),
        Input("camera-slider", "value"),
        Input("weight-mode", "value"),
        Input("w-norm", "value"),
        Input("c-norm", "value"),
        Input("subsample", "value"),
        Input("overlays", "value"),
    )
    def update(cam_idx, weight_mode, w_norm, c_norm, subsample, overlays):
        cam = STATE["cameras"][int(cam_idx)]
        fields = _compute_per_camera(cam, weight_mode, w_norm, c_norm)
        fig = _build_figure(cam, int(cam_idx), fields, weight_mode, w_norm, c_norm,
                            float(subsample), overlays or [])
        table_data, table_cols = _build_table(fields)
        u = fields["u"]
        vis_np = fields["visibility"].cpu().numpy()
        n_vis = int(vis_np.sum())
        w_gi_all = fields["w_gi"].cpu().numpy()
        status_children = [
            html.B(f"Cam {cam_idx}"), f"  {n_vis}/{len(vis_np)} tiles visible", html.Br(),
            f"weight: {weight_mode}  w_norm: {w_norm}  c_norm: {c_norm}", html.Br(),
            f"U      [{u.min():.3g}, {u.max():.3g}]", html.Br(),
            f"W_k    [{float(fields['W_k'].min()):.3g}, {float(fields['W_k'].max()):.3g}]", html.Br(),
            f"C_k    [{float(fields['C_k'].min()):.3g}, {float(fields['C_k'].max()):.3g}]", html.Br(),
            f"w(g_i) [{float(w_gi_all.min()):.3g}, {float(w_gi_all.max()):.3g}]", html.Br(),
            f"#GS {_fmt_count(len(STATE['gs_xyz']))}",
        ]
        dist_fig = _build_dist_figure(fields, weight_mode)
        return fig, table_data, table_cols, status_children, dist_fig

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive headless 3DGS debug viewer")
    parser.add_argument("--ply", required=True, type=str)
    parser.add_argument("--camera-trace", required=True, type=str)
    parser.add_argument("--grid-shape", nargs=3, type=int, default=[4, 4, 4])
    parser.add_argument("--img-w", type=int, default=800)
    parser.add_argument("--img-h", type=int, default=800)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_state(Path(args.ply), Path(args.camera_trace), tuple(args.grid_shape),
                args.img_w, args.img_h, device)
    app = _build_app()
    logger.info("Serving on http://{host}:{port} — forward: ssh -L {port}:localhost:{port} <server>",
                host=args.host, port=args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
