"""
Stained Glass Effect v2 - fully vectorized (no per-pixel Python loops).

Region-aware: when a line_alpha / outline mask is available from the
coloring pipeline, lead lines follow the artwork exactly instead of
re-detecting edges from the flattened result (which used to black out
densely detailed images).
"""

import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _lead_lines(image: np.ndarray,
                line_alpha: Optional[np.ndarray],
                intensity: float) -> np.ndarray:
    """Return a float32 HxW alpha map (0..1) for the lead came."""
    h, w = image.shape[:2]
    if line_alpha is not None:
        alpha = line_alpha.astype(np.float32) / 255.0
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
        alpha = cv2.GaussianBlur(edges, (3, 3), 0.8).astype(np.float32) / 255.0
    return np.clip(alpha * (0.6 + 0.4 * intensity), 0.0, 1.0)


def _glass_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Multi-octave smooth noise in [0,1], vectorized."""
    tex = np.zeros((h, w), dtype=np.float32)
    for scale, weight in [(96, 0.5), (48, 0.28), (24, 0.14), (12, 0.08)]:
        gh, gw = max(2, h // scale), max(2, w // scale)
        noise = rng.standard_normal((gh, gw)).astype(np.float32)
        tex += cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC) * weight
    tex -= tex.min()
    tex /= (tex.max() + 1e-8)
    return cv2.GaussianBlur(tex, (0, 0), 2.0)


def apply_stained_glass(image: np.ndarray,
                        labeled_regions: Optional[np.ndarray] = None,
                        intensity: float = 0.8,
                        line_alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply a stained glass look. All operations are vectorized; runs in
    well under a second even at 3000px (the old version took minutes
    because of nested Python pixel loops).
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]
    rng = np.random.default_rng(42)
    img = image.astype(np.float32)

    # 1. Gentle glass texture modulation (multiplicative, hue-preserving)
    tex = _glass_texture(h, w, rng)
    mod = 1.0 + (tex[:, :, None] - 0.5) * 0.25 * intensity
    img *= mod

    # 2. Directional light: soft diagonal gradient (vectorized meshgrid)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w),
                         indexing="ij")
    light = 1.0 + (1.0 - np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2)) \
        * 0.18 * intensity
    img *= light[:, :, None]

    # 3. A few soft specular blooms (screen blend)
    bloom = np.zeros((h, w), dtype=np.float32)
    for _ in range(3):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = int(min(h, w) * (0.10 + 0.10 * rng.random()))
        dist2 = (np.arange(w)[None, :] - cx) ** 2 + \
                (np.arange(h)[:, None] - cy) ** 2
        bloom += np.exp(-dist2 / (r * r + 1e-8)) * 40.0 * intensity
    img = 255.0 - (255.0 - img) * (1.0 - np.clip(bloom, 0, 120)[:, :, None] / 255.0)

    # 4. Saturation boost (rich glass colors)
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.30 * intensity), 0, 255)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                       cv2.COLOR_HSV2RGB).astype(np.float32)

    # 5. Lead came: composite ONCE with a dark metallic color.
    lead_a = _lead_lines(image, line_alpha, intensity)[:, :, None]
    lead_color = np.array([24, 24, 30], dtype=np.float32)[None, None, :]
    img = img * (1.0 - lead_a) + lead_color * lead_a

    # 6. Mild vignette (vectorized radial falloff)
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((np.arange(w)[None, :] - cx) ** 2 +
                   (np.arange(h)[:, None] - cy) ** 2)
    vig = 1.0 - (dist / dist.max()) * 0.18 * intensity
    img *= vig[:, :, None]

    return np.clip(img, 0, 255).astype(np.uint8)
