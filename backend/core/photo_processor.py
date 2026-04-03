"""
Unified Photo Processing Pipeline

Converts photos directly to 4-color theorem output WITHOUT an
intermediate line-art step.  This preserves the semantic region
boundaries (sky, ground, trees, water) that get lost when the
image is first converted to line art and then re-segmented.

Flow:
  Photo → CLAHE enhance → K-means color clustering → edge extraction
  → connected-component regions → adjacency graph → DSATUR colouring
  → area-balanced recolouring → render with thin outlines
"""

import cv2
import numpy as np
from scipy import ndimage
from collections import Counter
import networkx as nx
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def process_photo(
    image_rgb: np.ndarray,
    palette: List[List[int]],
    n_clusters: int = 8,
    min_region_area: int = 200,
    max_colors: int = 4,
    outline_thickness: str = "thin",
) -> Tuple[np.ndarray, Dict]:
    """
    Process a photo through the unified 4-colour pipeline.

    Args:
        image_rgb:  Input image as RGB uint8 numpy array.
        palette:    List of [R, G, B] colour lists (len >= max_colors).
        n_clusters: Number of K-means colour clusters (8-12 is good).
        min_region_area: Ignore regions smaller than this many pixels.
        max_colors: 4 or 5.
        outline_thickness: 'none', 'thin', or 'medium'.

    Returns:
        (colored_image, stats_dict)
    """
    h, w = image_rgb.shape[:2]
    logger.info(f"Photo pipeline: {w}x{h}, {n_clusters} clusters, {max_colors} colours")

    # ------------------------------------------------------------------
    # 1. CLAHE contrast enhancement
    # ------------------------------------------------------------------
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    # ------------------------------------------------------------------
    # 2. K-means colour clustering → boundary edges
    # ------------------------------------------------------------------
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels_km, _centers = cv2.kmeans(
        pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    clustered = labels_km.reshape(h, w).astype(np.uint8)

    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cluster_edges = (
        cv2.morphologyEx(clustered, cv2.MORPH_GRADIENT, kernel_cross) > 0
    ).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # 3. Multi-scale adaptive Canny edges
    # ------------------------------------------------------------------
    median_val = np.median(enhanced_gray)
    lower = int(max(0, 0.67 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges_multi = np.zeros((h, w), dtype=np.uint8)
    for ksize in [3, 5, 7]:
        blurred = cv2.GaussianBlur(enhanced_gray, (ksize, ksize), 0)
        edges_multi = cv2.bitwise_or(
            edges_multi, cv2.Canny(blurred, lower, upper)
        )

    # ------------------------------------------------------------------
    # 4. Combine edge sources
    # ------------------------------------------------------------------
    combined = (
        edges_multi.astype(np.float32) * 0.4
        + cluster_edges.astype(np.float32) * 0.6
    )
    _, combined_bin = cv2.threshold(combined, 40, 255, cv2.THRESH_BINARY)
    combined_bin = combined_bin.astype(np.uint8)

    # Close small gaps so regions are properly enclosed
    combined_bin = cv2.morphologyEx(
        combined_bin,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    thickened = cv2.dilate(
        combined_bin,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    # ------------------------------------------------------------------
    # 5. Connected-component region detection
    # ------------------------------------------------------------------
    regions_map = cv2.bitwise_not(thickened)
    num_labels, labeled = cv2.connectedComponents(regions_map, connectivity=4)

    # Fill edge pixels with nearest region (no black gaps)
    unlabeled = labeled == 0
    if np.any(unlabeled):
        _, idx = ndimage.distance_transform_edt(unlabeled, return_indices=True)
        labeled[unlabeled] = labeled[idx[0][unlabeled], idx[1][unlabeled]]

    # Filter small regions and relabel consecutively
    region_sizes = Counter(labeled.flatten())
    valid_map = {}
    new_label = 1
    for old_label in sorted(region_sizes.keys()):
        if old_label == 0:
            continue
        if region_sizes[old_label] >= min_region_area:
            valid_map[old_label] = new_label
            new_label += 1

    filtered = np.zeros_like(labeled, dtype=np.int32)
    for old, new in valid_map.items():
        filtered[labeled == old] = new

    # Fill gaps left by removed small regions
    unlabeled2 = filtered == 0
    if np.any(unlabeled2):
        _, idx2 = ndimage.distance_transform_edt(unlabeled2, return_indices=True)
        filtered[unlabeled2] = filtered[idx2[0][unlabeled2], idx2[1][unlabeled2]]

    valid_labels = sorted(set(filtered.flatten()) - {0})
    logger.info(f"Regions after filtering: {len(valid_labels)}")

    # ------------------------------------------------------------------
    # 6. Adjacency detection (search-radius to bridge thin edges)
    # ------------------------------------------------------------------
    adjacency: Dict[int, set] = {r: set() for r in valid_labels}
    radius = 3

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

    total_edges = sum(len(v) for v in adjacency.values()) // 2
    logger.info(f"Adjacency edges: {total_edges}")

    # ------------------------------------------------------------------
    # 7. Graph colouring (DSATUR)
    # ------------------------------------------------------------------
    G = nx.Graph()
    for r in valid_labels:
        G.add_node(r)
    for r, nbrs in adjacency.items():
        for n in nbrs:
            G.add_edge(r, n)

    coloring = nx.greedy_color(G, strategy="DSATUR")
    for r in coloring:
        coloring[r] = coloring[r] % max_colors

    # ------------------------------------------------------------------
    # 8. Area-balanced recolouring
    # ------------------------------------------------------------------
    region_areas = {r: int(np.sum(filtered == r)) for r in valid_labels}
    total_area = sum(region_areas.values())

    balanced = dict(coloring)
    for _ in range(500):
        ca = [0] * max_colors
        for r, c in balanced.items():
            ca[c] += region_areas.get(r, 0)
        max_c = max(range(max_colors), key=lambda c: ca[c])
        min_c = min(range(max_colors), key=lambda c: ca[c])
        if max_c == min_c or (ca[max_c] - ca[min_c]) / total_area < 0.08:
            break
        candidates = sorted(
            [(r, region_areas[r]) for r, c in balanced.items() if c == max_c],
            key=lambda x: x[1],
            reverse=True,
        )
        swapped = False
        for r, _ in candidates:
            nbr_colors = {
                balanced[n] for n in adjacency.get(r, set()) if n in balanced
            }
            if min_c not in nbr_colors:
                balanced[r] = min_c
                swapped = True
                break
        if not swapped:
            break

    ca = [0] * max_colors
    for r, c in balanced.items():
        ca[c] += region_areas.get(r, 0)
    logger.info(
        "Colour balance: %s",
        [f"{a / total_area * 100:.1f}%" for a in ca],
    )

    # ------------------------------------------------------------------
    # 9. Render
    # ------------------------------------------------------------------
    n_pal = len(palette)
    result = np.full((h, w, 3), 255, dtype=np.uint8)
    for r in valid_labels:
        mask = filtered == r
        result[mask] = palette[balanced.get(r, 0) % n_pal]

    # Draw edge outlines
    if outline_thickness != "none":
        result[thickened > 0] = [0, 0, 0]

    colors_used = len(set(balanced.values()))
    stats = {
        "regions": len(valid_labels),
        "colors_used": colors_used,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }

    return result, stats