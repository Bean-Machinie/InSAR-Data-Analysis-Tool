"""Server-side PNG overlay generation with LRU cache."""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image

from ..project.state import get_project


def _png_bytes(rgba: np.ndarray) -> bytes:
    image = Image.fromarray(np.flipud(rgba), mode="RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@lru_cache(maxsize=128)
def overlay_png_bytes(key: str, date_index: int) -> bytes:
    """Render a single overlay variable to PNG bytes (lat-axis flipped for Leaflet)."""
    proj = get_project()

    is_temporal = key in proj.temporal_values
    if is_temporal:
        n = proj.temporal_values[key].shape[0]
        idx = max(0, min(n - 1, date_index))
        values = proj.temporal_values[key][idx]
    elif key in proj.static_values:
        values = proj.static_values[key]
    else:
        raise KeyError(f"Variable '{key}' not available.")

    from ..project.registry import get_spec
    spec = get_spec(key)

    if spec and spec.category == "metadata":
        # Binary mask: two-colour render
        mask_vals = np.where(np.isfinite(values), values > 0.5, False)
        rgba = np.zeros(values.shape + (4,), dtype="uint8")
        rgba[proj.spatial_mask & ~mask_vals] = [154, 167, 173, 185]
        rgba[proj.spatial_mask & mask_vals] = [33, 166, 122, 225]
        return _png_bytes(rgba)

    visible = proj.spatial_mask & np.isfinite(values)
    lo, hi = proj.layer_ranges.get(key, (0.0, 1.0))
    cmap_name = spec.default_colormap if spec else "viridis"
    if cmap_name == "mask":
        cmap_name = "binary"

    symmetric = spec.symmetric if spec else False
    if symmetric and lo < 0 < hi:
        norm: Normalize = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    else:
        norm = Normalize(vmin=lo, vmax=hi)

    cmap = colormaps.get_cmap(cmap_name)
    normalized = np.zeros(values.shape, dtype="float64")
    normalized[visible] = norm(values[visible])
    colors = cmap(normalized)
    alpha = np.where(visible, 1.0, 0.0)

    rgba = np.clip(colors * 255, 0, 255).astype("uint8")
    rgba[..., 3] = np.clip(alpha * 255, 0, 255).astype("uint8")
    return _png_bytes(rgba)


def invalidate_cache() -> None:
    overlay_png_bytes.cache_clear()
