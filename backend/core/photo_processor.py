"""
Unified Image Processing Pipelines

Three entry points:

1. process_photo()       - For photographs (K-means LAB clustering)
2. process_coloring_book() - For line art / coloring book images
3. is_coloring_book()    - Auto-detect image type

Both pipelines fill every pixel with colour, use area-balanced
graph colouring, and draw thin outlines only at region boundaries.

Each pipeline returns (colored_image, stats, recolor_data) where
recolor_data contains the intermediate arrays needed for instant
palette swaps without re-running the heavy processing.
"""

import cv2
import numpy as np
from scipy import ndimage
from collections import Counter
import networkx as nx
from typing import Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Type alias for the optional progress callback
ProgressCB = Optional[Callable[[str, int], None]]


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
    filtered = lut[labeled]
    filtered = _fill_unlabeled(filtered)
    valid_labels = sorted(set(filtered.flatten()) - {0})
    return filtered, valid_labels


def _build_adjacency(filtered, valid_labels, radius=4):
    """Vectorized adjacency detection using NumPy array operations."""
    from collections import defaultdict

    left = filtered[:, :-radius].ravel()
    right = filtered[:, radius:].ravel()
    hmask = (left != right) & (left > 0) & (right > 0)

    top = filtered[:-radius, :].ravel()
    bottom = filtered[radius:, :].ravel()
    vmask = (top != bottom) & (top > 0) & (bottom > 0)

    p1 = np.concatenate([left[hmask], top[vmask]])
    p2 = np.concatenate([right[hmask], bottom[vmask]])

    if len(p1) == 0:
        return {r: set() for r in valid_labels}

    mins = np.minimum(p1, p2).astype(np.int64)
    maxs = np.maximum(p1, p2).astype(np.int64)
    packed = (mins << 32) | maxs
    unique_packed = np.unique(packed)

    u1 = (unique_packed >> 32).astype(int)
    u2 = (unique_packed & 0xFFFFFFFF).astype(int)

    adj = defaultdict(set)
    for s1, s2 in zip(u1.tolist(), u2.tolist()):
        adj[s1].add(s2)
        adj[s2].add(s1)

    for r in valid_labels:
        if r not in adj:
            adj[r] = set()

    return dict(adj)


def _kempe_chain(coloring, adjacency, start, a, b):
    """Connected component of `start` in the subgraph colored {a, b}."""
    chain = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for n in adjacency.get(x, ()):
            if n not in chain and coloring.get(n) in (a, b):
                chain.add(n)
                stack.append(n)
    return chain


def _graph_color(valid_labels, adjacency, max_colors):
    """
    Proper graph coloring capped at max_colors.

    DSATUR first; any node it colors beyond the cap is repaired with
    Kempe chain interchanges (flip-and-verify, revert on failure). The
    old implementation took DSATUR's result modulo max_colors, which
    silently produced ADJACENT SAME-COLOR REGIONS whenever DSATUR needed
    more than max_colors - the one thing a Four Color Theorem demo must
    never do.
    """
    G = nx.Graph()
    for r in valid_labels:
        G.add_node(r)
    for r, nbrs in adjacency.items():
        for n in nbrs:
            G.add_edge(r, n)
    coloring = nx.greedy_color(G, strategy="DSATUR")

    overflow = [r for r, c in coloring.items() if c >= max_colors]
    unrepaired = 0
    for v in overflow:
        nbrs = list(adjacency.get(v, ()))

        def nbr_colors():
            return {coloring[n] for n in nbrs if n in coloring}

        free = [c for c in range(max_colors) if c not in nbr_colors()]
        if free:
            coloring[v] = free[0]
            continue

        repaired = False
        for a in range(max_colors):
            if repaired:
                break
            for b in range(max_colors):
                if b == a or repaired:
                    continue
                for n in nbrs:
                    if coloring.get(n) != a:
                        continue
                    chain = _kempe_chain(coloring, adjacency, n, a, b)
                    if v in chain:
                        continue
                    for x in chain:
                        coloring[x] = b if coloring[x] == a else a
                    if a not in nbr_colors():
                        coloring[v] = a
                        repaired = True
                        break
                    # revert
                    for x in chain:
                        coloring[x] = b if coloring[x] == a else a
                if repaired:
                    break

        if not repaired:
            # Last resort: minimize visible conflicts
            counts = Counter(coloring.get(n) for n in nbrs)
            coloring[v] = min(range(max_colors),
                              key=lambda c: counts.get(c, 0))
            unrepaired += 1

    if overflow:
        logger.info(f"Graph coloring: repaired {len(overflow) - unrepaired}"
                    f"/{len(overflow)} overflow nodes via Kempe chains")
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
    progress_cb: ProgressCB = None,
) -> Tuple[np.ndarray, Dict, Dict]:
    """Process a photograph via K-means clustering.

    Returns (colored_image, stats, recolor_data).
    """
    h, w = image_rgb.shape[:2]
    logger.info(f"Photo pipeline: {w}x{h}")

    # --- Stage: enhance ---
    if progress_cb:
        progress_cb("Enhancing contrast", 10)

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    # --- Stage: cluster ---
    if progress_cb:
        progress_cb("Clustering pixels", 20)

    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    max_kmeans_samples = 100_000
    if len(pixels) > max_kmeans_samples:
        sample_idx = np.random.choice(len(pixels), max_kmeans_samples, replace=False)
        _, _, centers = cv2.kmeans(
            pixels[sample_idx], n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
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

    # --- Stage: edge detection ---
    if progress_cb:
        progress_cb("Detecting edges", 35)

    med = np.median(enhanced_gray)
    lo, hi = int(max(0, 0.67 * med)), int(min(255, 1.33 * med))
    edges_multi = np.zeros((h, w), dtype=np.uint8)
    for ks in [3, 5, 7]:
        bl = cv2.GaussianBlur(enhanced_gray, (ks, ks), 0)
        edges_multi = cv2.bitwise_or(edges_multi, cv2.Canny(bl, lo, hi))

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

    # --- Stage: region detection ---
    if progress_cb:
        progress_cb("Finding regions", 50)

    _, labeled = cv2.connectedComponents(cv2.bitwise_not(thickened), connectivity=4)
    labeled = _fill_unlabeled(labeled)
    filtered, valid_labels = _filter_regions(labeled, min_region_area)
    logger.info(f"Photo regions: {len(valid_labels)}")

    # --- Stage: graph colouring ---
    if progress_cb:
        progress_cb("Building adjacency graph", 60)

    adjacency = _build_adjacency(filtered, valid_labels, radius=1)

    if progress_cb:
        progress_cb("Solving graph coloring", 70)

    coloring, G = _graph_color(valid_labels, adjacency, max_colors)
    region_areas = _compute_areas(filtered, valid_labels)
    balanced = _area_balance(coloring, region_areas, adjacency, max_colors)

    # --- Stage: render ---
    if progress_cb:
        progress_cb("Rendering result", 85)

    outline_mask = thickened > 0

    max_label = int(filtered.max()) + 1
    color_lut = np.full((max_label, 3), 255, dtype=np.uint8)
    for r, c in balanced.items():
        color_lut[r] = palette[c % len(palette)]
    result = color_lut[filtered]
    result[outline_mask] = [0, 0, 0]

    # Soft alpha version of the outline for consistent downstream use
    line_alpha = cv2.GaussianBlur(
        (outline_mask.astype(np.uint8) * 255), (3, 3), 0.6
    )

    recolor_data = {
        "filtered": filtered,
        "balanced": balanced,
        "outline_mask": outline_mask,
        "line_alpha": line_alpha,
    }

    return result, _make_stats(valid_labels, balanced, G), recolor_data


# ======================================================================
# Pipeline 2: Coloring book / line art
# ======================================================================

def _auto_threshold_coloring_book(gray: np.ndarray) -> int:
    """
    Threshold separating line pixels from paper.

    Uses Otsu as a starting point, clamped to a sane range so scans with
    gray shading or off-white paper don't blow up.
    """
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(np.clip(otsu, 96, 200))


def _line_alpha_from_gray(gray: np.ndarray, threshold: int) -> np.ndarray:
    """
    Anti-aliased line opacity (uint8) derived from the ORIGINAL artwork.

    Pixels darker than (threshold - band) are fully opaque line;
    opacity ramps smoothly to zero above the threshold. This preserves
    the original smooth linework instead of reconstructing ragged
    1-px boundaries.
    """
    g = gray.astype(np.float32)
    band = 40.0
    hi = float(threshold) + 12.0
    lo = hi - band
    alpha = np.clip((hi - g) / (hi - lo), 0.0, 1.0)
    # smoothstep for softer ramp
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return (alpha * 255.0).astype(np.uint8)


def sort_palette_by_luminance(palette: List[List[int]]) -> List[List[int]]:
    """Palette sorted darkest -> lightest (Rec.601 luminance)."""
    return sorted(palette,
                  key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])


def _luminance_refine(coloring, adjacency, region_lum, palette,
                      region_areas, max_colors, passes: int = 3):
    """
    Kempe-chain local search: starting from a valid graph coloring, move
    regions toward the palette color closest to their photo luminance.

    A direct switch is used when the target color is free among the
    region's neighbors. When it is blocked, the (current, target)
    Kempe chain containing the region is flipped instead - a classic
    map-coloring move that ALWAYS preserves a proper coloring - and the
    flip is kept only if it improves the total area-weighted luminance
    fit. This escapes the dense-graph deadlock where every neighbor
    blocks every simple switch.
    """
    # Quantile mapping: rank regions by photo luminance and target each
    # area-weighted quarter of them at one palette rank (palette already
    # sorted dark -> light by the caller). Absolute luminance matching
    # fails here because most photo regions are similarly dark and would
    # all demand the same color, which adjacency cannot allow; quantiles
    # spread the demand evenly while preserving the dark->dark,
    # light->light ordering.
    regs = sorted(coloring.keys(), key=lambda r: region_lum.get(r, 128.0))
    total_area = sum(region_areas.get(r, 1) for r in regs) or 1
    target = {}
    acc = 0.0
    for r in regs:
        a = region_areas.get(r, 1)
        q = (acc + a * 0.5) / total_area
        target[r] = min(max_colors - 1, int(q * max_colors))
        acc += a

    def fit_cost(region, color):
        return abs(color - target.get(region, 0)) \
            * region_areas.get(region, 1)

    prefs = {
        r: sorted(range(max_colors), key=lambda c: abs(c - target[r]))
        for r in coloring
    }
    result = dict(coloring)
    order = sorted(result.keys(), key=lambda r: -region_areas.get(r, 0))

    for _ in range(passes):
        changed = 0
        for r in order:
            cur = result[r]
            for c in prefs[r]:
                if c == cur:
                    break  # already at the best reachable preference
                nbr_colors = {result[n] for n in adjacency.get(r, ())
                              if n in result}
                if c not in nbr_colors:
                    result[r] = c
                    changed += 1
                    break
                # Kempe chain flip on colors {cur, c}
                chain = {r}
                stack = [r]
                while stack:
                    x = stack.pop()
                    for n in adjacency.get(x, ()):
                        if n in result and n not in chain \
                                and result[n] in (cur, c):
                            chain.add(n)
                            stack.append(n)
                delta = 0.0
                for x in chain:
                    oldc = result[x]
                    newc = c if oldc == cur else cur
                    delta += fit_cost(x, oldc) - fit_cost(x, newc)
                if delta > 0:
                    for x in chain:
                        result[x] = c if result[x] == cur else cur
                    changed += 1
                    break
        if changed == 0:
            break
    return result


def composite_lines(colored: np.ndarray, line_alpha: np.ndarray,
                    line_color=(0, 0, 0)) -> np.ndarray:
    """Alpha-composite anti-aliased lines over a colored image."""
    a = (line_alpha.astype(np.float32) / 255.0)[:, :, None]
    lc = np.array(line_color, dtype=np.float32)[None, None, :]
    out = colored.astype(np.float32) * (1.0 - a) + lc * a
    return np.clip(out, 0, 255).astype(np.uint8)


def process_coloring_book(
    image_rgb: np.ndarray,
    palette: List[List[int]],
    min_region_area: int = 50,
    max_colors: int = 4,
    outline_thickness: str = "thin",
    progress_cb: ProgressCB = None,
) -> Tuple[np.ndarray, Dict, Dict]:
    """Process a coloring book / line art image.

    Returns (colored_image, stats, recolor_data).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    logger.info(f"Coloring book pipeline: {w}x{h}")

    # --- Stage: threshold -------------------------------------------------
    if progress_cb:
        progress_cb("Analysing line work", 10)

    threshold = _auto_threshold_coloring_book(gray)
    _, lines = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Seal tiny gaps in the linework so regions are enclosed.
    # NOTE: no Canny here. Canny on clean line art produces double edges
    # (one on each side of every stroke) which fragments the image into
    # sliver regions and wrecks the coloring quality.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sealed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)

    logger.info(
        f"Coloring book lines: {np.sum(sealed > 0) / sealed.size * 100:.1f}% "
        f"(threshold={threshold})"
    )

    # --- Stage: region detection -----------------------------------------
    if progress_cb:
        progress_cb("Finding regions", 40)

    _, labeled = cv2.connectedComponents(cv2.bitwise_not(sealed), connectivity=4)
    labeled = _fill_unlabeled(labeled)

    # Scale the minimum region size with resolution so speckles vanish
    # on large scans but detail survives on small images.
    auto_min = max(16, (h * w) // 40000)
    min_area = max(min_region_area, auto_min)

    filtered, valid_labels = _filter_regions(labeled, min_area)
    logger.info(f"Coloring book regions: {len(valid_labels)} (min_area={min_area})")

    # --- Stage: graph colouring ------------------------------------------
    if progress_cb:
        progress_cb("Building adjacency graph", 60)

    adjacency = _build_adjacency(filtered, valid_labels, radius=1)

    if progress_cb:
        progress_cb("Solving graph coloring", 70)

    coloring, G = _graph_color(valid_labels, adjacency, max_colors)
    region_areas = _compute_areas(filtered, valid_labels)
    balanced = _area_balance(coloring, region_areas, adjacency, max_colors)

    # --- Stage: render ----------------------------------------------------
    if progress_cb:
        progress_cb("Rendering result", 85)

    max_label = int(filtered.max()) + 1
    color_lut = np.full((max_label, 3), 255, dtype=np.uint8)
    for r, c in balanced.items():
        color_lut[r] = palette[c % len(palette)]
    flat = color_lut[filtered]

    # Overlay the ORIGINAL anti-aliased linework for crisp, smooth lines.
    line_alpha = _line_alpha_from_gray(gray, threshold)
    if outline_thickness == "none":
        line_alpha = np.zeros_like(line_alpha)
    result = composite_lines(flat, line_alpha)

    recolor_data = {
        "filtered": filtered,
        "balanced": balanced,
        "outline_mask": None,
        "line_alpha": line_alpha,
    }

    return result, _make_stats(valid_labels, balanced, G), recolor_data


# ======================================================================
# Pipeline 3: Photo -> line art -> colored (segmentation-aware)
# ======================================================================

def _kmeans_label_map(image_rgb: np.ndarray, k: int) -> np.ndarray:
    """
    Cluster the photo into k color groups (LAB space) and return an
    h x w uint8 label map, cleaned so boundaries are smooth.
    """
    h, w = image_rgb.shape[:2]

    # Flatten texture first so clusters follow objects, not grain
    small = image_rgb
    scale = 1.0
    if max(h, w) > 900:
        scale = 900.0 / max(h, w)
        small = cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    for _ in range(2):
        small = cv2.bilateralFilter(small, 9, 55, 9)

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)

    max_samples = 80_000
    if len(pixels) > max_samples:
        idx = np.random.default_rng(0).choice(len(pixels), max_samples,
                                              replace=False)
        _, _, centers = cv2.kmeans(pixels[idx], k, None, criteria, 3,
                                   cv2.KMEANS_PP_CENTERS)
    else:
        _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3,
                                   cv2.KMEANS_PP_CENTERS)

    dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1).astype(np.uint8)
    labels = labels.reshape(small.shape[:2])

    # Clean speckle in the label map so boundaries are meaningful
    labels = cv2.medianBlur(labels, 5)
    labels = cv2.medianBlur(labels, 5)

    if scale != 1.0:
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    return labels


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of boundaries between label values."""
    b = np.zeros(labels.shape, dtype=bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    return (b.astype(np.uint8)) * 255


def process_photo_as_lineart(
    image_rgb: np.ndarray,
    palette: List[List[int]],
    min_region_area: int = 50,
    max_colors: int = 4,
    line_thickness: str = "medium",
    detail_level: str = "detailed",
    contrast: float = 1.0,
    progress_cb: ProgressCB = None,
) -> Tuple[np.ndarray, Dict, Dict, np.ndarray]:
    """
    Photo -> line art -> colored, with color-segmentation awareness.

    Regions are derived from BOTH the drawn lines and a k-means color
    segmentation of the photo. Drawn lines alone rarely form closed
    contours, so without the segmentation split, distinct objects flood
    together into one giant single-color patch.

    Returns (colored_image, stats, recolor_data, lineart_rgb).
    """
    from core.photo_to_lineart import convert_photo_to_lineart

    h, w = image_rgb.shape[:2]
    logger.info(f"Photo-as-lineart pipeline: {w}x{h}")

    # --- Stage: color segmentation ---------------------------------------
    if progress_cb:
        progress_cb("Segmenting objects", 8)
    k = 10 if detail_level == "detailed" else 6
    k_labels = _kmeans_label_map(image_rgb, k)
    seg_edges = _label_boundaries(k_labels)

    # --- Stage: line art ---------------------------------------------------
    if progress_cb:
        progress_cb("Converting photo to line art", 18)
    lineart = convert_photo_to_lineart(
        image_rgb,
        line_thickness=line_thickness,
        detail_level=detail_level,
        contrast=contrast,
        segmentation_edges=seg_edges,
    )
    gray = cv2.cvtColor(lineart, cv2.COLOR_RGB2GRAY)
    line_mask = gray < 128

    # --- Stage: region detection (cluster-aware) --------------------------
    if progress_cb:
        progress_cb("Finding regions", 40)

    # Connected components computed per color cluster with the drawn
    # lines acting as barriers: two touching objects with different
    # colors in the photo stay separate regions even if no line divides
    # them, and one object crossed by texture lines is still split only
    # where lines actually are.
    labeled = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    open_mask = (~line_mask)
    for c in range(int(k_labels.max()) + 1):
        m = ((k_labels == c) & open_mask).astype(np.uint8)
        if not m.any():
            continue
        n, lab = cv2.connectedComponents(m, connectivity=4)
        add = lab > 0
        labeled[add] = lab[add] + (next_id - 1)
        next_id += n - 1

    labeled = _fill_unlabeled(labeled)

    auto_min = max(24, (h * w) // 30000)
    min_area = max(min_region_area, auto_min)
    filtered, valid_labels = _filter_regions(labeled, min_area)
    logger.info(f"Photo-as-lineart regions: {len(valid_labels)} "
                f"(min_area={min_area}, clusters={k})")

    # --- Stage: graph colouring ------------------------------------------
    if progress_cb:
        progress_cb("Building adjacency graph", 60)
    adjacency = _build_adjacency(filtered, valid_labels, radius=1)

    if progress_cb:
        progress_cb("Solving graph coloring", 70)
    coloring, G = _graph_color(valid_labels, adjacency, max_colors)
    region_areas = _compute_areas(filtered, valid_labels)

    # --- Stage: luminance matching ---------------------------------------
    # Per-region: give each region the palette color whose luminance is
    # closest to that region's brightness in the ORIGINAL photo, using a
    # validity-preserving local search on top of the DSATUR coloring.
    # Dark objects end up with the palette's dark colors, skies with its
    # light ones - the result reads like the photo. (Area balancing is
    # skipped here; fidelity matters more than equal color areas.)
    gray_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sums = np.bincount(filtered.ravel(), weights=gray_orig.ravel().astype(np.float64))
    cnts = np.bincount(filtered.ravel())
    region_lum = {r: (sums[r] / cnts[r]) if cnts[r] > 0 else 128.0
                  for r in valid_labels}
    render_palette = sort_palette_by_luminance(palette)
    balanced = _luminance_refine(
        coloring, adjacency, region_lum, render_palette,
        region_areas, max_colors,
    )

    # --- Stage: render ----------------------------------------------------
    if progress_cb:
        progress_cb("Rendering result", 85)
    max_label = int(filtered.max()) + 1
    color_lut = np.full((max_label, 3), 255, dtype=np.uint8)
    for r, c in balanced.items():
        color_lut[r] = render_palette[c % len(render_palette)]
    flat = color_lut[filtered]

    line_alpha = _line_alpha_from_gray(gray, 128)
    result = composite_lines(flat, line_alpha)

    recolor_data = {
        "filtered": filtered,
        "balanced": balanced,
        "outline_mask": None,
        "line_alpha": line_alpha,
        "palette_luminance_sort": True,
    }

    return result, _make_stats(valid_labels, balanced, G), recolor_data, lineart
