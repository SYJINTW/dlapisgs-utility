#!/usr/bin/env python3
"""Aggregate selection-only metrics from 0507 budget sweep manifests.

This script does not depend on rendering outputs. It summarizes per-camera
selection results into per-(scheme,budget) means/stds for plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _discover_manifests(output_root: Path) -> list[Path]:
    return sorted(output_root.glob("**/selected.json"))


def _safe_pstdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate selection sweep metrics")
    parser.add_argument("--output-root", required=True, type=str,
                        help="Root directory of budget sweep outputs")
    parser.add_argument("--metrics-dir", required=True, type=str,
                        help="Directory to write aggregate CSV files")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifests = _discover_manifests(output_root)
    if not manifests:
        raise FileNotFoundError(f"No selected.json manifests found under {output_root}")

    per_camera_rows: list[dict[str, Any]] = []

    for manifest_path in manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        budget_bytes = int(data["budget_bytes"])
        used_bytes = int(data["used_bytes"])
        row = {
            "scheme": str(data["scheme"]),
            "budget_mb": float(data["budget_mb"]),
            "camera_index": int(data["camera_index"]),
            "budget_bytes": budget_bytes,
            "used_bytes": used_bytes,
            "used_ratio": (used_bytes / budget_bytes) if budget_bytes > 0 else 0.0,
            "selected_gaussians": int(data["selected_gaussians"]),
            "bytes_per_gaussian": int(data["bytes_per_gaussian"]),
            "manifest_path": str(manifest_path),
        }
        per_camera_rows.append(row)

    per_camera_rows.sort(key=lambda r: (r["scheme"], r["budget_mb"], r["camera_index"]))

    per_cam_csv = metrics_dir / "per_camera_selection_metrics.csv"
    with per_cam_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_camera_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_camera_rows)

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in per_camera_rows:
        grouped[(row["scheme"], row["budget_mb"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (scheme, budget_mb), rows in grouped.items():
        selected_vals = [float(r["selected_gaussians"]) for r in rows]
        ratio_vals = [float(r["used_ratio"]) for r in rows]
        used_vals = [float(r["used_bytes"]) for r in rows]

        summary_rows.append({
            "scheme": scheme,
            "budget_mb": budget_mb,
            "num_views": len(rows),
            "selected_gaussians_mean": float(mean(selected_vals)),
            "selected_gaussians_std": _safe_pstdev(selected_vals),
            "used_ratio_mean": float(mean(ratio_vals)),
            "used_ratio_std": _safe_pstdev(ratio_vals),
            "used_bytes_mean": float(mean(used_vals)),
            "used_bytes_std": _safe_pstdev(used_vals),
        })

    summary_rows.sort(key=lambda r: (r["scheme"], r["budget_mb"]))

    summary_csv = metrics_dir / "summary_by_budget_scheme.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote per-camera metrics: {per_cam_csv}")
    print(f"Wrote summary metrics: {summary_csv}")


if __name__ == "__main__":
    main()
