"""
In-memory cache for recolor data.

After the heavy processing pipeline runs once, we cache the labelled
region map, the graph-coloring assignment, and the outline mask.
Subsequent palette swaps only need to re-render — no edge detection,
no region finding, no graph coloring.

TTL and max-entry limits keep memory bounded.
"""

import threading
import time
import uuid
import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RecolorCache:
    def __init__(self, max_entries: int = 8, ttl_seconds: int = 1800):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.max_entries = max_entries
        self.ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, data: dict) -> str:
        """Store recolor data, return a session_id."""
        session_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._evict()
            self._store[session_id] = {
                "data": data,
                "created_at": time.time(),
            }
        logger.info(
            f"Cached recolor data: session={session_id}, "
            f"entries={len(self._store)}"
        )
        return session_id

    def get(self, session_id: str) -> Optional[dict]:
        """Retrieve cached recolor data, or None if expired / missing."""
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            if time.time() - entry["created_at"] > self.ttl:
                del self._store[session_id]
                return None
            return entry["data"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self):
        now = time.time()
        expired = [k for k, v in self._store.items()
                   if now - v["created_at"] > self.ttl]
        for k in expired:
            del self._store[k]
        while len(self._store) >= self.max_entries:
            oldest = min(self._store, key=lambda k: self._store[k]["created_at"])
            del self._store[oldest]
            logger.info(f"Evicted oldest recolor cache entry")


# Module-level singleton
recolor_cache = RecolorCache()


def render_from_cache(
    filtered: np.ndarray,
    balanced: Dict[int, int],
    outline_mask,
    palette: list,
    line_alpha=None,
) -> np.ndarray:
    """
    Re-render a coloured image from cached intermediate data.

    This is the fast path: no edge detection, no region finding, no
    graph colouring — just a lookup-table recolour + outline overlay.
    Runs in <50 ms for most images.
    """
    max_label = int(filtered.max()) + 1
    color_lut = np.full((max_label, 3), 255, dtype=np.uint8)
    for region_id, color_idx in balanced.items():
        color_lut[region_id] = palette[color_idx % len(palette)]
    result = color_lut[filtered]
    if line_alpha is not None:
        from core.photo_processor import composite_lines
        result = composite_lines(result, line_alpha)
    elif outline_mask is not None:
        result[outline_mask] = [0, 0, 0]
    return result


def render_from_cache_entry(cached: dict, palette: list) -> np.ndarray:
    """Render honoring per-session flags (e.g. luminance-sorted palette)."""
    if cached.get("palette_luminance_sort"):
        from core.photo_processor import sort_palette_by_luminance
        palette = sort_palette_by_luminance(palette)
    return render_from_cache(
        cached["filtered"], cached["balanced"], cached["outline_mask"],
        palette, line_alpha=cached.get("line_alpha"),
    )