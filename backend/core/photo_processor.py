"""
Unified Image Processing Pipelines

Three entry points:

1. process_photo()       - For photographs (K-means LAB clustering)
2. process_coloring_book() - For line art / coloring book images
3. is_coloring_book()    - Auto-detect image type

Both pipelines fill every pixel with colour, use area-balanced
graph colouring, and draw thin outlines only at region boundaries.
"""

import cv2
import numpy as np
from scipy import ndimage
from collections import Counter
import networkx as nx
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Auto-detection
# ======================================================================

def is_coloring_book(image_rgb: np.ndarray) -> bool:
    """
    Detect whether an image is a coloring book (black lines on white/light
    background) vs a photograph.

    Coloring books have a strongly bimodal histogram: mostly light pixels
    (>180) with a small percentage of dark line pixels, and very few
    mid-tone pixels.  Photos have a much more even distribution.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()

    light = hist_norm[180:].sum()
    mid = hist_norm[80:180].sum()

    is_cb = light > 0.50 and mid < 0.30
    logger.info(
        f"Auto-detect: light={light:.2f} mid={mid:.2f} "
        f"-> {'coloring_book' if is_cb else 'photo'}"
    )
    return is_cb


# ======================================================================
# Shared helpers
# ======================================================================

def _fill_unlabeled(labeled: np.ndarray) -> np.ndarray:
    unlabeled = labeled == 0
    if not np.any(unlabeled):
        return labeled
    _, idx = ndimage.distance_transform_edt(unlabeled, return_indices=True)
    out = labeled.copy()
    out[unlabeled] = labeled[idx[0][unlabeled], idx[1][unlabeled]]
    return out


def _filter_regions(labeled, min_area):
    # Build a lookup table: old_label -> new_label (0 = removed)
    unique_labels, counts = np.unique(labeled, return_counts=True)
    max_label = int(unique_labels.max()) + 1
    lut = np.zeros(max_label, dtype=np.int32)
    nl = 1
    for label, count in zip(unique_labels, counts):
        if label == 0:
            continue
        if count >= min_area:
            lut[label] = nl
            nl += 1
    # Apply lookup table (vectorized, no per-region loop)
    filtered = lut[labeled]
    filtered = _fill_unlabeled(filtered)
    valid_labels = sorted(set(filtered.flatten()) - {0})
    return filtered, valid_labels


def _build_adjacency(filtered, valid_labels, radius=4):
    """Vectorized adjacency detection using NumPy array operations."""
    from collections import defaultdict

    # Horizontal: compare pixels `radius` apart
    left = filtered[:, :-radius].ravel()
    right = filtered[:, radius:].ravel()
    hmask = (left != right) & (left > 0) & (right > 0)

    # Vertical: compare pixels `radius` apart
    top = filtered[:-radius, :].ravel()
    bottom = filtered[radius:, :].ravel()
    vmask = (top != bottom) & (top > 0) & (bottom > 0)

    # Combine all adjacent pairs
    p1 = np.concatenate([left[hmask], top[vmask]])
    p2 = np.concatenate([right[hmask], bottom[vmask]])

    if len(p1) == 0:
        return {r: set() for r in valid_labels}

    # Deduplicate: pack sorted pairs into int64
    mins = np.minimum(p1, p2).astype(np.int64)
    maxs = np.maximum(p1, p2).astype(np.int64)
    packed = (mins << 32) | maxs
    unique_packed = np.unique(packed)

    # Unpack into adjacency dict
    u1 = (unique_packed >> 32).astype(int)
    u2 = (unique_packed & 0xFFFFFFFF).astype(int)

    adj = defaultdict(set)
    for s1, s2 in zip(u1.tolist(), u2.tolist()):
        adj[s1].add(s2)
        adj[s2].add(s1)

    # Ensure all valid labels have an entry
    for r in valid_labels:
        if r not in adj:
            adj[r] = set()

    return dict(adj)


def _graph_color(valid_labels, adjacency, max_colors):
    G = nx.Graph()
    for r in valid_labels:
        G.add_node(r)
    for r, nbrs in adjacency.items():
        for n in nbrs:
            G.add_edge(r, n)
    coloring = nx.greedy_color(G, strategy="DSATUR")
    for r in coloring:
        coloring[r] = coloring[r] % max_colors
    return coloring, G


def _area_balance(coloring, region_areas, adjacency, max_colors,
                  target=0.08, max_iter=500):
    balanced = dict(coloring)
    total = sum(region_areas.values())
    if total == 0:
        return balanced
    for _ in range(max_iter):
        ca = [0] * max_colors
        for r, c in balanced.items():
            ca[c] += region_areas.get(r, 0)
        maxc = max(range(max_colors), key=lambda c: ca[c])
        minc = min(range(max_colors), key=lambda c: ca[c])
        if maxc == minc or (ca[maxc] - ca[minc]) / total < target:
            break
        cands = sorted(
            [(r, region_areas[r]) for r, c in balanced.items() if c == maxc],
            key=lambda x: x[1], reverse=True,
        )
        swapped = False
        for r, _ in cands:
            nc = {balanced[n] for n in adjacency.get(r, set()) if n in balanced}
            if minc not in nc:
                balanced[r] = minc
                swapped = True
                break
        if not swapped:
            break
    return balanced


def _compute_areas(filtered, valid_labels):
    """Vectorized region area calculation."""
    unique, counts = np.unique(filtered, return_counts=True)
    area_map = dict(zip(unique.tolist(), counts.tolist()))
    return {r: area_map.get(r, 0) for r in valid_labels}


def _make_stats(valid_labels, balanced, G):
    return {
        "regions": len(valid_labels),
        "colors_used": len(set(balanced.values())),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }


# ======================================================================
# Pipeline 1: Photos
# ======================================================================

def process_photo(
    image_rgb: np.ndarray,
    palette: List[List[int]],
    n_clusters: int = 8,
    min_region_area: int = 200,
    max_colors: int = 4,
    outline_thickness: str = "thin",
) -> Tuple[np.ndarray, Dict]:
    """Process a photograph via K-means clustering."""
    h, w = image_rgb.shape[:2]
    logger.info(f"Photo pipeline: {w}x{h}")

    # CLAHE
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    # K-means cluster boundaries (subsampled for speed)
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    max_kmeans_samples = 100_000
    if len(pixels) > max_kmeans_samples:
        sample_idx = np.random.choice(len(pixels), max_kmeans_samples, replace=False)
        _, _, centers = cv2.kmeans(
            pixels[sample_idx], n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        # Assign all pixels to nearest center in chunks
        labels_km = np.zeros(len(pixels), dtype=np.int32)
        chunk_size = 500_000
        for i in range(0, len(pixels), chunk_size):
            chunk = pixels[i : i + chunk_size]
            dists = np.linalg.norm(
                chunk[:, None, :] - centers[None, :, :], axis=2
            )
            labels_km[i : i + chunk_size] = np.argmin(dists, axis=1)
    else:
        _, labels_km, _ = cv2.kmeans(
            pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        labels_km = labels_km.ravel()

    clustered = labels_km.reshape(h, w).astype(np.uint8)
    kc = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cluster_edges = (
        cv2.morphologyEx(clustered, cv2.MORPH_GRADIENT, kc) > 0
    ).astype(np.uint8) * 255

    # Multi-scale Canny
    med = np.median(enhanced_gray)
    lo, hi = int(max(0, 0.67 * med)), int(min(255, 1.33 * med))
    edges_multi = np.zeros((h, w), dtype=np.uint8)
    for ks in [3, 5, 7]:
        bl = cv2.GaussianBlur(enhanced_gray, (ks, ks), 0)
        edges_multi = cv2.bitwise_or(edges_multi, cv2.Canny(bl, lo, hi))

    # Combine & close
    combined = (
        edges_multi.astype(np.float32) * 0.4
        + cluster_edges.astype(np.float32) * 0.6
    )
    _, cb = cv2.threshold(combined, 40, 255, cv2.THRESH_BINARY)
    cb = cb.astype(np.uint8)
    cb = cv2.morphologyEx(
        cb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    thickened = cv2.dilate(
        cb, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1
    )

    # Regions
    _, labeled = cv2.connectedComponents(cv2.bitwise_not(thickened), connectivity=4)
    labeled = _fill_unlabeled(labeled)
    filtered, valid_labels = _filter_regions(labeled, min_region_area)
    logger.info(f"Photo regions: {len(valid_labels)}")

    adjacency = _build_adjacency(filtered, valid_labels, radius=3)
    coloring, G = _graph_color(valid_labels, adjacency, max_colors)
    region_areas = _compute_areas(filtered, valid_labels)
    balanced = _area_balance(coloring, region_areas, adjacency, max_colors)

    # Render using lookup table (vectorized, no per-region loop)
    max_label = int(filtered.max()) + 1
    color_lut = np.full((max_label, 3), 255, dtype=np.uint8)
    for r, c in balanced.items():
        color_lut[r] = palette[c % len(palette)]
    result = color_lut[filtered]
    result[thickened > 0] = [0, 0, 0]

    return result, _make_stats(valid_labels, balanced, G)


# ======================================================================
# Pipeline 2: Coloring book / line art
# ======================================================================

def _auto_threshold_coloring_book(gray: np.ndarray, target_pct: float = 0.10) -> int:
    """
    Find the brightness threshold where ~target_pct of pixels are darker.
    This adapts to different line weights and background shades.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    cumsum = np.cumsum(hist / hist.sum())
    threshold = 128
    for t in range(256):
        if cumsum[t] >= target_pct:
            threshold = t
            break
    return max(100, min(threshold, 200))


def process_coloring_book(
    image_rgb: np.ndarray,
    palette: List[List[int]],
    min_region_area: int = 50,
    max_colors: int = 4,
    outline_thickness: str = "thin",
) -> Tuple[np.ndarray, Dict]:
    """
    Process a coloring book / line art image.

    Uses adaptive thresholding tuned to capture ~10% of pixels as edges,
    preserving fine detail in intricate designs while still finding clean
    region boundaries in simple images.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    logger.info(f"Coloring book pipeline: {w}x{h}")

    # Adaptive threshold: capture ~10% of pixels as lines
    threshold = _auto_threshold_coloring_book(gray, target_pct=0.10)
    _, lines = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Canny for additional edge detail
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
    edges_canny = cv2.Canny(blurred, 30, 80)

    # Combine and lightly close gaps
    combined = cv2.bitwise_or(lines, edges_canny)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    logger.info(
        f"Coloring book edges: {np.sum(combined > 0) / combined.size * 100:.1f}% "
        f"(threshold={threshold})"
    )

    # Find regions
    _, labeled = cv2.connectedComponents(cv2.bitwise_not(combined), connectivity=4)
    labeled = _fill_unlabeled(labeled)

    # Adaptive min-area: keep smaller regions for detailed images
    raw_count = len(np.unique(labeled)) - 1
    if raw_count > 500:
        min_area = max(20, min_region_area // 2)
    else:
        min_area = min_region_area

    filtered, valid_labels = _filter_regions(labeled, min_area)
    logger.info(f"Coloring book regions: {len(valid_labels)} (min_area={min_area})")

    adjacency = _build_adjacency(filtered, valid_labels, radius=4)
    coloring, G = _graph_color(valid_labels, adjacency, max_colors)
    region_areas = _compute_areas(filtered, valid_labels)
    balanced = _area_balance(coloring, region_areas, adjacency, max_colors)

    # Render using lookup table (vectorized)
    max_label = int(filtered.max()) + 1
    color_lut = np.full((max_label, 3), 255, dtype=np.uint8)
    for r, c in balanced.items():
        color_lut[r] = palette[c % len(palette)]
    result = color_lut[filtered]

    # Thin outline at boundaries, using original dark pixels
    if outline_thickness != "none":
        boundary = np.zeros((h, w), dtype=bool)
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            shifted = np.roll(np.roll(filtered, dy, axis=0), dx, axis=1)
            boundary |= (filtered != shifted)
        original_dark = gray < threshold
        outline = boundary & original_dark
        if outline_thickness == "medium":
            kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            outline = cv2.dilate(
                outline.astype(np.uint8), kern, iterations=1
            ).astype(bool)
        result[outline] = [0, 0, 0]

    return result, _make_stats(valid_labels, balanced, G)