"""Detect closed regions in coloring book images."""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
import logging

logger = logging.getLogger(__name__)


class RegionDetector:
    def __init__(self, min_region_area: int = 100, is_line_art: bool = False):
        """
        Initialize region detector.

        Args:
            min_region_area: Minimum area in pixels for a valid region
            is_line_art: Whether processing line art (uses larger min area)
        """
        if is_line_art:
            self.min_region_area = max(min_region_area, 200)
        else:
            self.min_region_area = min_region_area

    def detect_regions(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
        """Detect closed regions in an image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        logger.info(f"Region detection – image shape: {image.shape}")

        edges = self._detect_edges(gray)
        cleaned_edges = self._clean_edges(edges)
        labeled_regions, contours = self._find_regions(cleaned_edges)

        logger.info(
            f"Regions found: {len(contours)} "
            f"(edge pixels: {int(np.sum(edges > 0))})"
        )

        stats = {
            "total_regions": len(contours),
            "edge_pixels": int(np.sum(edges > 0)),
            "average_region_area": (
                float(np.mean([cv2.contourArea(c) for c in contours]))
                if contours
                else 0
            ),
        }
        return labeled_regions, contours, stats

    # ------------------------------------------------------------------
    # Edge detection
    # ------------------------------------------------------------------

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, 50, 150)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(binary)
        return cv2.bitwise_or(edges, inverted)

    def _clean_edges(self, edges: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        denoised = cv2.medianBlur(closed, 3)
        try:
            denoised = cv2.ximgproc.thinning(denoised)
        except (AttributeError, cv2.error):
            pass  # ximgproc not available
        return denoised

    # ------------------------------------------------------------------
    # Region finding
    # ------------------------------------------------------------------

    def _find_regions(
        self, edges: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        regions = cv2.bitwise_not(edges)
        num_labels, labeled = cv2.connectedComponents(regions, connectivity=4)

        contours: List[np.ndarray] = []
        valid_labels: List[int] = []

        for label in range(1, num_labels):
            mask = (labeled == label).astype(np.uint8) * 255
            region_contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if region_contours:
                contour = region_contours[0]
                if cv2.contourArea(contour) >= self.min_region_area:
                    contours.append(contour)
                    valid_labels.append(label)

        relabeled = np.zeros_like(labeled)
        for new_label, old_label in enumerate(valid_labels, 1):
            relabeled[labeled == old_label] = new_label

        relabeled = self._fill_unlabeled(relabeled)
        return relabeled, contours

    # ------------------------------------------------------------------
    # Adjacency detection
    # ------------------------------------------------------------------

    def find_adjacent_regions(
        self, labeled_regions: np.ndarray
    ) -> Dict[int, Set[int]]:
        """Find adjacent regions using a small search radius to bridge thin edges."""
        h, w = labeled_regions.shape
        unique_regions = np.unique(labeled_regions)
        unique_regions = unique_regions[unique_regions > 0]
        num_regions = len(unique_regions)

        adjacency: Dict[int, Set[int]] = {int(r): set() for r in unique_regions}

        logger.info(f"Finding adjacencies for {num_regions} regions")

        if num_regions > 1000:
            search_radius = 3
        else:
            search_radius = 5

        for y in range(h):
            for x in range(w - search_radius):
                s1 = int(labeled_regions[y, x])
                s2 = int(labeled_regions[y, x + search_radius])
                if s1 > 0 and s2 > 0 and s1 != s2:
                    adjacency[s1].add(s2)
                    adjacency[s2].add(s1)

        for y in range(h - search_radius):
            for x in range(w):
                s1 = int(labeled_regions[y, x])
                s2 = int(labeled_regions[y + search_radius, x])
                if s1 > 0 and s2 > 0 and s1 != s2:
                    adjacency[s1].add(s2)
                    adjacency[s2].add(s1)

        diag = max(2, search_radius - 1)
        for y in range(h - diag):
            for x in range(w - diag):
                s1 = int(labeled_regions[y, x])
                s2 = int(labeled_regions[y + diag, x + diag])
                if s1 > 0 and s2 > 0 and s1 != s2:
                    adjacency[s1].add(s2)
                    adjacency[s2].add(s1)

        total_edges = sum(len(v) for v in adjacency.values()) // 2
        logger.info(f"Found {total_edges} adjacency edges")
        return adjacency

    # ------------------------------------------------------------------
    # Region area helper
    # ------------------------------------------------------------------

    @staticmethod
    def compute_region_areas(labeled_regions: np.ndarray) -> Dict[int, int]:
        """Return {region_id: pixel_count} for every region ≥ 1."""
        unique, counts = np.unique(labeled_regions, return_counts=True)
        return {int(r): int(c) for r, c in zip(unique, counts) if r > 0}

    # ------------------------------------------------------------------
    # Rendering helper
    # ------------------------------------------------------------------

    @staticmethod
    def render_colored_image(
        labeled_regions: np.ndarray,
        coloring: Dict[int, int],
        palette: list,
        line_art: np.ndarray = None,
        outline_thickness: str = 'thin',
    ) -> np.ndarray:
        """Render a coloured image with thin outlines on region boundaries only."""
        h, w = labeled_regions.shape
        n_colors = len(palette)

        try:
            from scipy import ndimage as _ndi
            unlabeled = labeled_regions == 0
            if np.any(unlabeled):
                _, idx = _ndi.distance_transform_edt(unlabeled, return_indices=True)
                labeled_regions = labeled_regions.copy()
                labeled_regions[unlabeled] = labeled_regions[idx[0][unlabeled], idx[1][unlabeled]]
        except ImportError:
            pass

        result = np.full((h, w, 3), 255, dtype=np.uint8)
        for region_id, color_idx in coloring.items():
            mask = labeled_regions == region_id
            result[mask] = palette[color_idx % n_colors]

        if outline_thickness == 'none':
            return result

        boundary = np.zeros((h, w), dtype=bool)
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            shifted = np.roll(np.roll(labeled_regions, dy, axis=0), dx, axis=1)
            boundary |= (labeled_regions != shifted)

        if line_art is not None:
            gray_la = cv2.cvtColor(line_art, cv2.COLOR_RGB2GRAY) if line_art.ndim == 3 else line_art
            la_edges = gray_la < 128
            boundary = boundary & la_edges

        if outline_thickness == 'medium':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool)

        result[boundary] = [0, 0, 0]
        return result

    # ------------------------------------------------------------------
    # Fill helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_unlabeled(labeled: np.ndarray) -> np.ndarray:
        """Fill unlabelled pixels with nearest labelled region."""
        try:
            from scipy import ndimage
        except ImportError:
            logger.warning("scipy not available, skipping unlabelled fill")
            return labeled

        unlabeled = labeled == 0
        if not np.any(unlabeled):
            return labeled

        _, indices = ndimage.distance_transform_edt(unlabeled, return_indices=True)
        filled = labeled.copy()
        filled[unlabeled] = labeled[indices[0][unlabeled], indices[1][unlabeled]]
        return filled