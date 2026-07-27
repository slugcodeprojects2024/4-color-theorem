"""
Photo to Line Art Converter (v2)

Converts photographs into clean coloring-book style line art.

Approach:
  1. Upscale small images so edge operators have enough pixels to work with.
  2. Flatten texture with edge-preserving smoothing (repeated bilateral
     filtering on a downscaled copy - the classic "cartoon" trick).
  3. Extract lines two ways and combine:
       a. Adaptive threshold  -> organic, sketch-like strokes
       b. Median-based Canny  -> structural object boundaries
  4. Close small gaps so regions are enclosed (colorable), remove specks.
  5. Optional thickness adjustment, then anti-aliased final render.
"""

import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _smooth_flatten(rgb: np.ndarray) -> np.ndarray:
    """Edge-preserving texture flattening (bilateral pyramid)."""
    h, w = rgb.shape[:2]
    # Work at reduced size for speed, then restore
    scale = 1.0
    small = rgb
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    for _ in range(2):
        small = cv2.bilateralFilter(small, 9, 55, 9)
    if scale != 1.0:
        small = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return small


def _remove_small_line_specks(lines: np.ndarray, min_area: int) -> np.ndarray:
    """Remove tiny isolated line fragments (operates on line pixels == 255)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    if n <= 1:
        return lines
    keep = np.zeros(n, dtype=np.uint8)
    keep[stats[:, cv2.CC_STAT_AREA] >= min_area] = 255
    keep[0] = 0
    return keep[labels]


def convert_photo_to_lineart(
    image: np.ndarray,
    line_thickness: str = "medium",
    detail_level: str = "detailed",
    contrast: float = 1.0,
    segmentation_edges: np.ndarray = None,
) -> np.ndarray:
    """
    Convert a photo to coloring-book line art.

    Returns an RGB image: black anti-aliased lines on a white background.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    h0, w0 = image.shape[:2]

    # --- 1. Ensure a workable resolution -------------------------------
    work = image
    upscaled = False
    if max(h0, w0) < 700:
        f = 700.0 / max(h0, w0)
        work = cv2.resize(image, (int(w0 * f), int(h0 * f)),
                          interpolation=cv2.INTER_CUBIC)
        upscaled = True
    h, w = work.shape[:2]

    # --- 2. Flatten texture --------------------------------------------
    flat = _smooth_flatten(work)
    gray = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY)

    # Mild local contrast so shadowed areas still yield edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    contrast = float(np.clip(contrast, 0.5, 2.0))
    if contrast != 1.0:
        gray = cv2.convertScaleAbs(gray, alpha=contrast,
                                   beta=128 * (1 - contrast))

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # --- 3a. Adaptive-threshold strokes --------------------------------
    # Block size scales with image size; C controls sensitivity.
    if detail_level == "detailed":
        block = _odd(max(11, min(h, w) // 40))
        c_val = 6
    else:
        block = _odd(max(21, min(h, w) // 18))
        c_val = 13
        blurred = cv2.medianBlur(blurred, 5)
    sketch = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, c_val,
    )

    # --- 3b. Structural Canny edges ------------------------------------
    med = float(np.median(blurred))
    lo = int(max(10, 0.55 * med))
    hi = int(min(255, 1.35 * med))
    canny = cv2.Canny(blurred, lo, hi, L2gradient=True)
    # Slight dilation so canny lines survive the AND-style merge visually
    canny = cv2.dilate(canny, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)

    lines = cv2.bitwise_or(sketch, canny)

    # Object boundaries from color segmentation (closes contours between
    # objects that have no luminance edge - crucial so separate objects
    # become separate colorable regions instead of one giant patch)
    if segmentation_edges is not None:
        seg = segmentation_edges
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        seg = cv2.dilate(seg, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
        lines = cv2.bitwise_or(lines, seg)

    # --- 4. Clean up ----------------------------------------------------
    # Close 1-2 px gaps so regions are enclosed for coloring
    lines = cv2.morphologyEx(
        lines, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    # Remove speckles (scaled with resolution)
    speck = max(24, (h * w) // 18000)
    if detail_level != "detailed":
        speck *= 3
    lines = _remove_small_line_specks(lines, speck)

    # --- 5. Thickness ---------------------------------------------------
    if line_thickness == "thick":
        lines = cv2.dilate(lines, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    elif line_thickness == "thin":
        lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(
                                     cv2.MORPH_ELLIPSE, (2, 2)))

    # --- 6. Anti-aliased render ----------------------------------------
    # Soft edges: blur the binary mask slightly, invert to white bg
    soft = cv2.GaussianBlur(lines, (3, 3), 0.6).astype(np.float32) / 255.0
    out = (255.0 * (1.0 - soft)).astype(np.uint8)

    if upscaled:
        out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_AREA)

    dark_pct = float((out < 128).mean())
    logger.info(f"Line art: {w0}x{h0}, dark={dark_pct:.1%}, "
                f"thickness={line_thickness}, detail={detail_level}")

    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


# Backwards-compatible class wrapper
class PhotoToLineArt:
    def convert(self, image, line_thickness="medium",
                detail_level="detailed", contrast=1.0):
        return convert_photo_to_lineart(image, line_thickness,
                                        detail_level, contrast)
