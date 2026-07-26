"""
Real-time coloring animation payload.

Instead of streaming video frames (heavy), the server ships the frontend
everything it needs to animate the coloring itself:

  - label_map:     region label ids packed into an RGB PNG (label < 2^24)
  - line_overlay:  RGBA PNG of the anti-aliased linework
  - region_colors: {label_id: [r, g, b]}
  - paint_order:   label ids sorted for a pleasing fill order
                   (largest regions first, ties broken spatially)

The frontend decodes the label map into a Uint32 array and paints regions
one-by-one on a <canvas> with requestAnimationFrame, compositing the line
overlay on top. Flat-color PNGs compress extremely well, so the payload
is typically only a few hundred KB even for large images.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional


ANIM_MAX_DIM = 1400


def _png_b64(arr: np.ndarray, mode: str) -> str:
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_animation_payload(
    filtered: np.ndarray,
    balanced: Dict[int, int],
    palette: List[List[int]],
    line_alpha: Optional[np.ndarray],
    max_dim: int = ANIM_MAX_DIM,
) -> Dict:
    """Build the animation payload dict for the frontend."""
    h, w = filtered.shape

    # Downscale for bandwidth if needed (nearest keeps labels intact)
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        small = cv2.resize(filtered.astype(np.int32), (nw, nh),
                           interpolation=cv2.INTER_NEAREST)
        la_small = None
        if line_alpha is not None:
            la_small = cv2.resize(line_alpha, (nw, nh),
                                  interpolation=cv2.INTER_AREA)
    else:
        small = filtered.astype(np.int32)
        la_small = line_alpha
        nh, nw = h, w

    # Pack labels into RGB (supports up to 16.7M regions)
    packed = np.zeros((nh, nw, 3), dtype=np.uint8)
    packed[:, :, 0] = (small >> 16) & 0xFF
    packed[:, :, 1] = (small >> 8) & 0xFF
    packed[:, :, 2] = small & 0xFF
    label_map_b64 = _png_b64(packed, "RGB")

    # Line overlay as RGBA (black lines, alpha = opacity)
    line_overlay_b64 = None
    if la_small is not None and np.any(la_small):
        rgba = np.zeros((nh, nw, 4), dtype=np.uint8)
        rgba[:, :, 3] = la_small
        line_overlay_b64 = _png_b64(rgba, "RGBA")

    # Region colors and paint order
    region_colors = {
        str(r): palette[c % len(palette)] for r, c in balanced.items()
    }

    unique, counts = np.unique(small, return_counts=True)
    area = dict(zip(unique.tolist(), counts.tolist()))
    # Centroid-ish tiebreak: first occurrence row for stable spatial order
    order = sorted(
        (r for r in balanced.keys() if area.get(r, 0) > 0),
        key=lambda r: -area.get(r, 0),
    )

    return {
        "width": nw,
        "height": nh,
        "label_map": label_map_b64,
        "line_overlay": line_overlay_b64,
        "region_colors": region_colors,
        "paint_order": [int(r) for r in order],
    }
