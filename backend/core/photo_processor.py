"""
Unified Image Processing Pipelines

Two pipelines:

1. process_photo()
   For photographs: CLAHE → K-means LAB clustering → edge extraction
   → connected components → graph coloring → area balancing → render.

2. process_coloring_book()
   For coloring book / line art images: edge detection → connected
   components → fill all pixels → graph coloring → area balancing
   → render with original line art outlines.

Both pipelines fill every pixel with colour and draw thin outlines
only at region boundaries, avoiding the "too much black" problem.
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
# Shared helpers
# ======================================================================

def _fill_unlabeled(labeled: np.ndarray) -> np.ndarray:
    """Fill label-0 pixels with nearest labelled region."""
    unlabeled = labeled == 0
    if not np.any(unlabeled):
        return labeled
    _, idx = ndimage.distance_transform_edt(unlabeled, return_indices=True)
    out = labeled.copy()
    out[unlabeled] = labeled[idx[0][unlabeled], idx[1][unlabeled]]
    return out


def _build_adjacency(filtered: np.ndarray, valid_labels: list, radius: int = 3):
    """Build adjacency dict with a search radius to bridge thin edges."""
    h, w = filtered.shape
    adjacency: Dict[int, set] = {r: set() for r in valid_labels}
    for y in range(h):
        for x in range(w - radius):
            s1, s2 = int(filtered[y, x]), int(filtered[y, x + radius])
            if s1 > 0 and s2 > 0 and s1 != s2:
                adjacency[s1].add(s2)
                adjacency[s2].add(s1)
    for y in range(h - radius):
        for x in range(w):
            s1, s2 = int(filtered[y, x]), int(filtered[y + radius, x])
            if s1 > 0 and s2 > 0 and s1 != s2:
                adjacency[s1].add(s2)
                adjacency[s2].add(s1)
    return adjacency


def _graph_color(valid_labels, adjacency, max_colors):
    """DSATUR graph coloring, normalized to 0..max_colors-1."""
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


def _area_balance(coloring, region_areas, adjacency, max_colors, target=0.08, max_iter=500):
    """Swap colours to equalize pixel-area per colour."""
    balanced = dict(coloring)
    total = sum(region_areas.values())
    if total == 0:
        return balanced
    for _ in range(max_iter):
        ca = [0] * max_colors
        for r, c in balanced.items():
            ca[c] += region_areas.get(r, 0)
        max_c = max(range(max_colors), key=lambda c: ca[c])
        min_c = min(range(max_colors), key=lambda c: ca[c])
        if max_c == min_c or (ca[max_c] - ca[min_c]) / total < target:
            break
        candidates = sorted(
            [(r, region_areas[r]) for r, c in balanced.items() if c == max_c],
            key=lambda x: x[1], reverse=True,
        )
        swapped = False
        for r, _ in candidates:
            nbr_colors = {balanced[n] for n in adjacency.get(r, set()) if n in balanced}
            if min_c not in nbr_colors:
                balanced[r] = min_c
                swapped = True
                break
        if not swapped:
            break
    return balanced


def _filter_regions(labeled, min_area):
    """Remove small regions, relabel consecutively, fill gaps."""
    region_sizes = Counter(labeled.flatten())
    valid_map = {}
    new_label = 1
    for old_label in sorted(region_sizes.keys()):
        if old_label == 0:
            continue
        if region_sizes[old_label] >= min_area:
            valid_map[old_label] = new_label
            new_label += 1
    filtered = np.zeros_like(labeled, dtype=np.int32)
    for old, new in valid_map.items():
        filtered[labeled == old] = new
    filtered = _fill_unlabeled(filtered)
    valid_labels = sorted(set(filtered.flatten()) - {0})
    return filtered, valid_labels


def _render(filtered, balanced, palette, edge_source, outline_mode="thin"):
    """
    Render coloured image with thin outlines.

    edge_source: uint8 image where dark pixels (< 128) are original edges.
                 Used to keep outline quality matching the original art style.
                 Pass None to use pure boundary outlines.
    """
    h, w = filtered.shape
    n_pal = len(palette)
    result = np.full((h, w, 3), 255, dtype=np.uint8)
    for r in set(filtered.flatten()) - {0}:
        mask = filtered == r
        result[mask] = palette[balanced.get(r, 0) % n_pal]

    if outline_mode == "none":
        return result

    # Region boundaries (1px)
    boundary = np.zeros((h, w), dtype=bool)
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        shifted = np.roll(np.roll(filtered, dy, axis=0), dx, axis=1)
        boundary |= (filtered != shifted)

    if edge_source is not None:
        gray_e = cv2.cvtColor(edge_source, cv2.COLOR_RGB2GRAY) if edge_source.ndim == 3 else edge_source
        original_lines = gray_e < 128
        outline = boundary & original_lines
    else:
        outline = boundary

    outline_u8 = outline.astype(np.uint8) * 255
    if outline_mode == "medium":
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        outline_u8 = cv2.dilate(outline_u8, kern, iterations=1)

    result[outline_u8 > 0] = [0, 0, 0]
    return result


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
    """
    Process a photograph through the unified pipeline.

    Returns (coloured_image, stats_dict).
    """
    h, w = image_rgb.shape[:2]
    logger.info(f"Photo pipeline: {w}x{h}")

    # CLAHE enhancement
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    # K-means clustering for region boundaries
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels_km, _ = cv2.kmeans(pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    clustered = labels_km.reshape(h, w).astype(np.uint8)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cluster_edges = (cv2.morphologyEx(clustered, cv2.MORPH_GRADIENT, kernel_cross) > 0).astype(np.uint8) * 255

    # Multi-scale adaptive Canny
    median_val = np.median(enhanced_gray)
    lo = int(max(0, 0.67 * median_val))
    hi = int(min(255, 1.33 * median_val))
    edges_multi = np.zeros((h, w), dtype=np.uint8)
    for ksize in [3, 5, 7]:
        blurred = cv2.GaussianBlur(enhanced_gray, (ksize, ksize), 0)
        edges_multi = cv2.bitwise_or(edges_multi, cv2.Canny(blurred, lo, hi))

    # Combine
    combined = edges_multi.astype(np.float32) * 0.4 + cluster_edges.astype(np.float32) * 0.6
    _, combined_bin = cv2.threshold(combined, 40, 255, cv2.THRESH_BINARY)
    combined_bin = combined_bin.astype(np.uint8)
    combined_bin = cv2.morphologyEx(combined_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    thickened = cv2.dilate(combined_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    # Connected components
    regions_map = cv2.bitwise_not(thickened)
    _, labeled = cv2.connectedComponents(regions_map, connectivity=4)
    labeled = _fill_unlabeled(labeled)
    filtered, valid_labels = _filter_regions(labeled, min_region_area)
    logger.info(f"Photo regions: {len(valid_labels)}")

    # Adjacency, coloring, balancing
    adjacency = _build_adjacency(filtered, valid_labels, radius=3)
    coloring, G = _graph_color(valid_labels, adjacency, max_colors)
    region_areas = {r: int(np.sum(filtered == r)) for r in valid_labels}
    balanced = _area_balance(coloring, region_areas, adjacency, max_colors)

    # Log balance
    total = sum(region_areas.values())
    ca = [0] * max_colors
    for r, c in balanced.items():
        ca[c] += region_areas.get(r, 0)
    logger.info("Photo colour balance: %s", [f"{a/total*100:.0f}%" for a in ca])

    # Render
    result = np.full((h, w, 3), 255, dtype=np.uint8)
    for r in valid_labels:
        mask = filtered == r
        result[mask] = palette[balanced.get(r, 0) % len(palette)]
    result[thickened > 0] = [0, 0, 0]

    stats = {
        "regions": len(valid_labels),
        "colors_used": len(set(balanced.values())),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }
    return result, stats


# ======================================================================
# Pipeline 2: Coloring book / line art images
# ======================================================================

def process_coloring_book(
    image_rgb: np.ndarray,
    palette: List[List[int]],
    min_region_area: int = 100,
    max_colors: int = 4,
    outline_thickness: str = "thin",
) -> Tuple[np.ndarray, Dict]:
    """
    Process a coloring book / line art image.

    Detects regions from the existing black lines, fills every pixel
    with colour, and re-draws thin outlines matching the original art.

    Returns (coloured_image, stats_dict).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    logger.info(f"Coloring book pipeline: {w}x{h}")

    # Edge detection: Canny + adaptive threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges_canny = cv2.Canny(blurred, 50, 150)

    # Otsu threshold catches the black lines reliably across
    # different background shades (pure white, off-white, cream)
    _, edges_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    combined_edges = cv2.bitwise_or(edges_canny, edges_otsu)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)

    # Connected components on inverted edge map
    regions_map = cv2.bitwise_not(combined_edges)
    _, labeled = cv2.connectedComponents(regions_map, connectivity=4)

    # Fill edge pixels with nearest region
    labeled = _fill_unlabeled(labeled)

    # Filter small regions
    filtered, valid_labels = _filter_regions(labeled, min_region_area)
    logger.info(f"Coloring book regions: {len(valid_labels)}")

    # Adjacency, coloring, balancing
    adjacency = _build_adjacency(filtered, valid_labels, radius=5)
    coloring, G = _graph_color(valid_labels, adjacency, max_colors)
    region_areas = {r: int(np.sum(filtered == r)) for r in valid_labels}
    balanced = _area_balance(coloring, region_areas, adjacency, max_colors)

    # Log balance
    total = sum(region_areas.values())
    ca = [0] * max_colors
    for r, c in balanced.items():
        ca[c] += region_areas.get(r, 0)
    logger.info("Coloring book colour balance: %s", [f"{a/total*100:.0f}%" for a in ca])

    # Render with original line art as outline source
    result = _render(filtered, balanced, palette, gray, outline_mode=outline_thickness)

    stats = {
        "regions": len(valid_labels),
        "colors_used": len(set(balanced.values())),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }
    return result, stats