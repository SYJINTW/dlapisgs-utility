"""Numpy-only helpers for the LOO oracle ΔQ artifact (Exp 4).

Importable from either conda env — no torch dependency. See
`.claude/exp4_oracle_dq.md` §D4 for the on-disk schema this module
reads and writes.

Oracle delta Q(tile)

"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


# --- tiling loader -----------------------------------------------------------

def load_tiling(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Read a `.tiling_cache.npz` written by `test_utility.py`.

    Returns keys we need downstream:
      - flat_indices  (total_GS,) int64
      - index_offsets (N_tiles+1,) int64
      - min_corners   (N_tiles, 3) float32  (passed through for traceability)
      - max_corners   (N_tiles, 3) float32
      - n_tiles       int

    Note: cache schema produced by test_utility.py omits the `tile_keys`
    column that `GGSP/tiling.py` would normally save. Min/max corners
    serve the same traceability role.
    """
    data = np.load(str(npz_path))
    out = {
        "flat_indices": data["flat_indices"].astype(np.int64, copy=False),
        "index_offsets": data["index_offsets"].astype(np.int64, copy=False),
        "min_corners": data["min_corners"].astype(np.float32, copy=False),
        "max_corners": data["max_corners"].astype(np.float32, copy=False),
    }
    out["n_tiles"] = int(out["index_offsets"].shape[0] - 1)
    return out


def iter_tile_member_indices(
    tiling: dict[str, np.ndarray],
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (tile_k, member_indices) for every non-empty tile."""
    off = tiling["index_offsets"]
    flat = tiling["flat_indices"]
    for k in range(tiling["n_tiles"]):
        yield k, flat[off[k] : off[k + 1]]


def n_gs_per_tile(tiling: dict[str, np.ndarray]) -> np.ndarray:
    off = tiling["index_offsets"]
    return (off[1:] - off[:-1]).astype(np.int64, copy=False)


# --- result schema -----------------------------------------------------------

@dataclass
class OracleDQResult:
    """LOO + optional AOI oracle artifact per scene. See exp4_oracle_dq.md §D4."""
    min_corners: np.ndarray         # (N_tiles, 3) float32
    max_corners: np.ndarray         # (N_tiles, 3) float32
    index_offsets: np.ndarray       # (N_tiles+1,) int64
    flat_indices: np.ndarray        # (total_GS,) int64
    camera_indices: np.ndarray      # (N_cams,) int32
    n_gs_per_tile: np.ndarray       # (N_tiles,) int64
    mse: np.ndarray                 # (N_cams, N_tiles) float32  — LOO: MSE(full\k) vs full
    psnr: np.ndarray                # (N_cams, N_tiles) float32
    ssim: np.ndarray                # (N_cams, N_tiles) float32
    gen_meta: dict[str, Any] = field(default_factory=dict)
    mse_aoi: np.ndarray | None = None   # (N_cams, N_tiles) float32 — AOI: MSE({k}) vs full

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            min_corners=self.min_corners,
            max_corners=self.max_corners,
            index_offsets=self.index_offsets,
            flat_indices=self.flat_indices,
            camera_indices=self.camera_indices,
            n_gs_per_tile=self.n_gs_per_tile,
            mse=self.mse,
            psnr=self.psnr,
            ssim=self.ssim,
            gen_meta=np.asarray(json.dumps(self.gen_meta)),
        )
        if self.mse_aoi is not None:
            arrays["mse_aoi"] = self.mse_aoi
        np.savez(str(path), **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> "OracleDQResult":
        d = np.load(str(path), allow_pickle=False)
        meta_str = str(d["gen_meta"].item()) if d["gen_meta"].shape == () else "{}"
        try:
            meta = json.loads(meta_str)
        except (ValueError, TypeError):
            meta = {}
        return cls(
            min_corners=d["min_corners"],
            max_corners=d["max_corners"],
            index_offsets=d["index_offsets"],
            flat_indices=d["flat_indices"],
            camera_indices=d["camera_indices"],
            n_gs_per_tile=d["n_gs_per_tile"],
            mse=d["mse"],
            psnr=d["psnr"],
            ssim=d["ssim"],
            gen_meta=meta,
            mse_aoi=d["mse_aoi"] if "mse_aoi" in d else None,
        )


def write_long_csv(result: OracleDQResult, path: str | Path) -> None:
    """Write long-form (camera, tile, n_gs_in_tile, mse, psnr, ssim) CSV.

    Pandas-free so this works in the gaussian_splatting env.
    """
    import csv

    n_cams, n_tiles = result.mse.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["camera", "tile", "n_gs_in_tile", "mse", "psnr", "ssim"])
        for ci in range(n_cams):
            cam = int(result.camera_indices[ci])
            for k in range(n_tiles):
                w.writerow([
                    cam, k, int(result.n_gs_per_tile[k]),
                    float(result.mse[ci, k]),
                    float(result.psnr[ci, k]),
                    float(result.ssim[ci, k]),
                ])


def to_long_df(result: OracleDQResult):
    """Optional pandas DataFrame (requires pandas in the calling env)."""
    import pandas as pd

    n_cams, n_tiles = result.mse.shape
    cam_col = np.repeat(result.camera_indices, n_tiles)
    tile_col = np.tile(np.arange(n_tiles, dtype=np.int32), n_cams)
    n_gs_col = np.tile(result.n_gs_per_tile.astype(np.int64), n_cams)
    return pd.DataFrame({
        "camera": cam_col,
        "tile": tile_col,
        "n_gs_in_tile": n_gs_col,
        "mse": result.mse.reshape(-1),
        "psnr": result.psnr.reshape(-1),
        "ssim": result.ssim.reshape(-1),
    })


# --- analysis stub (follow-up; expanded once U_k dumps exist) ----------------

def spearman_vs_u(
    result: OracleDQResult,
    u_per_cam_tile: np.ndarray,
    metric: str = "mse",
    visible_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Spearman ρ between candidate U_k and oracle ΔQ_k.

    Parameters
    ----------
    u_per_cam_tile : (N_cams, N_tiles) float — formula's U_k per camera.
    visible_mask   : (N_cams, N_tiles) bool, optional — True where tile is
                     visible from camera. Per-camera ρ uses only visible
                     tiles; ranks on invisibles are degenerate ties.
    """
    from scipy.stats import spearmanr

    target = getattr(result, metric)
    if u_per_cam_tile.shape != target.shape:
        raise ValueError(f"shape mismatch: U {u_per_cam_tile.shape} vs ΔQ {target.shape}")

    per_cam_rho: list[float] = []
    for c in range(target.shape[0]):
        u_row = u_per_cam_tile[c]
        d_row = target[c]
        if visible_mask is not None:
            mask = visible_mask[c]
            if mask.sum() < 3:
                continue
            u_row = u_row[mask]
            d_row = d_row[mask]
        rho, _ = spearmanr(u_row, d_row)
        if np.isfinite(rho):
            per_cam_rho.append(float(rho))

    u_agg = u_per_cam_tile.mean(axis=0)
    d_agg = target.mean(axis=0)
    rho_agg, _ = spearmanr(u_agg, d_agg)

    return {
        "metric": metric,
        "rho_per_cam_mean": float(np.mean(per_cam_rho)) if per_cam_rho else float("nan"),
        "rho_per_cam_n": len(per_cam_rho),
        "rho_agg": float(rho_agg) if np.isfinite(rho_agg) else float("nan"),
    }
