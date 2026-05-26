"""Project folder discovery with file-presence scoring."""
from __future__ import annotations

from pathlib import Path


def product_score(path: Path) -> int:
    score = 0
    if (path / "results_tight.nc").exists():
        score += 8
    if (path / "results_wide.nc").exists():
        score += 5
    if (path / "sbas_results_masked.npz").exists():
        score += 4
    if (path / "parameters.json").exists():
        score += 3
    if (path / "manifest.json").exists():
        score += 2
    if (path / "geotiffs").is_dir():
        score += 2
    return score


def find_product_dir(root_dir: Path) -> Path:
    """Return the best-scoring product folder at or below root_dir."""
    candidates: list[Path] = [root_dir]

    outputs_dir = root_dir / "outputs"
    if outputs_dir.exists():
        candidates.extend(p for p in outputs_dir.iterdir() if p.is_dir())

    # Also search for any results_tight.nc deeper in the tree
    for nc in root_dir.rglob("results_tight.nc"):
        candidates.append(nc.parent)
    for nc in root_dir.rglob("results_wide.nc"):
        candidates.append(nc.parent)

    scored = sorted(
        ((product_score(p), p) for p in dict.fromkeys(candidates) if p.exists()),
        key=lambda item: (item[0], -len(item[1].parts)),
        reverse=True,
    )
    if scored and scored[0][0] > 0:
        return scored[0][1].resolve()

    return root_dir.resolve()
