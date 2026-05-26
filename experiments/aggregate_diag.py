#!/usr/bin/env python3
"""Aggregate diag_ck_spearman JSONs into cross-scene × oracle-mode tables.

Reads `<diag_root>/<scene>_<mode>/diag_all_candidates.json` for
mode ∈ {loo, aoi, combined} and writes one markdown table per mode:
rows = candidate signal, columns = scene, cell = ρ_resid (after stripping
the log #GS tautology).

Bold ✱ marks |ρ_resid| ≥ 0.3 (PASS threshold per PLAN.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODES = ("loo", "aoi", "combined")
PASS_THRESHOLD = 0.30


def _load_scene_mode(diag_root: Path, scene: str, mode: str) -> dict[str, float] | None:
    p = diag_root / f"{scene}_{mode}" / "diag_all_candidates.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {c["name"]: c["rho_resid"] for c in d["candidates"]}


def _print_mode_table(mode: str, scenes: list[str], data: dict[str, dict[str, float]]) -> None:
    candidates: list[str] = []
    seen = set()
    for s in scenes:
        for k in data.get(s, {}):
            if k not in seen:
                candidates.append(k)
                seen.add(k)
    if not candidates:
        print(f"\n## oracle = {mode} — no data\n")
        return

    # Sort candidates by mean |ρ_resid| across available scenes (best on top)
    def _score(c):
        vals = [abs(data[s][c]) for s in scenes if s in data and c in data[s]]
        return sum(vals) / max(len(vals), 1)
    candidates.sort(key=_score, reverse=True)

    headers = ["candidate"] + scenes + ["mean|ρ|", "PASS"]
    rows = []
    for c in candidates:
        cells = [c]
        vals = []
        for s in scenes:
            v = data.get(s, {}).get(c)
            if v is None:
                cells.append("—")
            else:
                vals.append(v)
                mark = "✱" if abs(v) >= PASS_THRESHOLD else " "
                cells.append(f"{v:+.2f}{mark}")
        mean_abs = sum(abs(v) for v in vals) / max(len(vals), 1) if vals else 0.0
        cells.append(f"{mean_abs:.2f}")
        n_pass = sum(1 for v in vals if abs(v) >= PASS_THRESHOLD)
        cells.append(f"{n_pass}/{len(vals)}")
        rows.append(cells)

    table = [headers] + rows
    widths = [max(len(c) for c in col) for col in zip(*table)]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    print(f"\n## oracle = {mode}   (✱ = |ρ_resid| ≥ {PASS_THRESHOLD})\n")
    print(line(headers))
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for r in rows:
        print(line(r))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--diag-root", default="output/0522/diag")
    p.add_argument(
        "--scenes",
        nargs="+",
        default=["chair", "drums", "ficus", "hotdog", "mic", "materials", "ship", "bicycle"],
    )
    p.add_argument("--modes", nargs="+", default=list(MODES))
    args = p.parse_args()

    diag_root = Path(args.diag_root)
    available_scenes_by_mode = {m: [] for m in args.modes}
    data_by_mode = {m: {} for m in args.modes}
    for mode in args.modes:
        for s in args.scenes:
            rec = _load_scene_mode(diag_root, s, mode)
            if rec is not None:
                data_by_mode[mode][s] = rec
                available_scenes_by_mode[mode].append(s)

    for mode in args.modes:
        _print_mode_table(mode, available_scenes_by_mode[mode], data_by_mode[mode])

    print(
        "\n**Read:** rows are candidate signals, columns are scenes. "
        "ρ_resid = Spearman ρ vs log(ΔMSE) after subtracting OLS on log(#GS) "
        "(i.e. the bulk-count tautology stripped). "
        "PASS gate = |ρ_resid| ≥ 0.3 per PLAN.md."
    )


if __name__ == "__main__":
    main()
