"""
ML-Enhanced Region Detection Module

Provides multiple segmentation strategies:
1. SLIC Superpixels + Merging - Good quality, CPU-friendly
2. Traditional Edge Detection - Fallback, fastest
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SegmentationMethod(Enum):
    SLIC = "slic"
    EDGE = "edge"
    AUTO = "auto"


@dataclass
class SegmentationResult:
    """Result from segmentation."""
    labeled_regions: np.ndarray  # H x W array where each pixel has region ID
    num_regions: int
    method_used: SegmentationMethod
    confidence: float  # 0-1 confidence in segmentation quality
    metadata: Dict


class MLRegionDetector:
    """
    ML-Enhanced Region Detector with multiple backend support.
    
    Usage:
        detector = MLRegionDetector(method='auto')
        result = detector.segment(image)
    """
    
    def __init__(
        self, 
        method: Literal['slic', 'edge', 'auto'] = 'auto',
        min_region_area: int = 100,
        target_regions: int = 50
    ):
        self.method = SegmentationMethod(method)
        self.min_region_area = min_region_area
        self.target_regions = target_regions
        
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment image into regions."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB (H, W, 3)")
        
        method = self._select_method(image)
        logger.info(f"Segmenting image {image.shape} using method: {method.value}")
        
        if method == SegmentationMethod.SLIC:
            return self._segment_slic(image)
        else:
            return self._segment_edge(image)
    
    def _select_method(self, image: np.ndarray) -> SegmentationMethod:
        """Auto-select best available method."""
        if self.method != SegmentationMethod.AUTO:
            return self.method
        
        h, w = image.shape[:2]
        if h * w > 500 * 500:
            return SegmentationMethod.SLIC
        
        if self._is_likely_lineart(image):
            return SegmentationMethod.EDGE
        
        return SegmentationMethod.SLIC
    
    def _is_likely_lineart(self, image: np.ndarray) -> bool:
        """Detect if image is likely line art."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        dark_ratio = hist[:50].sum()
        light_ratio = hist[200:].sum()
        
        return (dark_ratio + light_ratio) > 0.7
    
    def _segment_slic(self, image: np.ndarray) -> SegmentationResult:
        """Segment using SLIC superpixels with region merging."""
        try:
            from skimage.segmentation import slic
            from skimage.color import rgb2lab
        except ImportError:
            logger.warning("scikit-image not available, falling back to edge detection")
            return self._segment_edge(image)
        
        h, w = image.shape[:2]
        # Create more initial segments to allow for better merging control
        n_segments = min(self.target_regions * 5, (h * w) // 500)  # More segments, less aggressive reduction
        n_segments = max(n_segments, 50)  # Minimum 50 segments for better detail
        
        lab_image = rgb2lab(image)
        
        segments = slic(
            lab_image,
            n_segments=n_segments,
            compactness=10,
            sigma=1,
            start_label=1,
            channel_axis=2
        )
        
        # Optionally merge similar regions - but be conservative to preserve detail
        # For photos with gradients, merging can cause too few regions
        merged = self._merge_similar_regions(image, segments, color_threshold=12.0)  # Very conservative threshold
        
        # Get unique labels (SLIC uses start_label=1, so should be 1, 2, 3...)
        unique_labels = np.unique(merged)
        unique_labels = unique_labels[unique_labels > 0]  # Remove any 0s (shouldn't exist with SLIC)
        
        if len(unique_labels) == 0:
            # Fallback: create a single region
            logger.warning("No valid regions found after merging, creating single region")
            result = np.ones_like(merged, dtype=np.int32)
            return SegmentationResult(
                labeled_regions=result,
                num_regions=1,
                method_used=SegmentationMethod.SLIC,
                confidence=0.3,
                metadata={'initial_segments': n_segments, 'after_merge': 1, 'fallback': True}
            )
        
        # Keep original labels but ensure they're consecutive starting from 1
        # This matches the behavior of traditional RegionDetector
        label_map = {}
        for new_label, old_label in enumerate(unique_labels, start=1):
            label_map[old_label] = new_label
        
        relabeled = np.zeros_like(merged, dtype=np.int32)
        for old_label, new_label in label_map.items():
            relabeled[merged == old_label] = new_label
        
        return SegmentationResult(
            labeled_regions=relabeled.astype(np.int32),
            num_regions=len(unique_labels),
            method_used=SegmentationMethod.SLIC,
            confidence=0.8,
            metadata={'initial_segments': n_segments, 'after_merge': len(unique_labels)}
        )
    
    def _merge_similar_regions(
        self, 
        image: np.ndarray, 
        segments: np.ndarray,
        color_threshold: float = 15.0  # Reduced from 25.0 to be less aggressive
    ) -> np.ndarray:
        """Merge adjacent regions with similar colors."""
        merged = segments.copy()
        unique_segments = np.unique(segments)
        
        segment_colors = {}
        for seg_id in unique_segments:
            mask = segments == seg_id
            if np.any(mask):
                segment_colors[seg_id] = image[mask].mean(axis=0)
        
        adjacency = self._find_adjacent_segments(segments)
        merged_into = {s: s for s in unique_segments}
        
        for seg_id in unique_segments:
            if merged_into[seg_id] != seg_id:
                continue
                
            for neighbor in adjacency.get(seg_id, []):
                neighbor_root = merged_into[neighbor]
                if neighbor_root == seg_id:
                    continue
                
                if seg_id in segment_colors and neighbor_root in segment_colors:
                    color1 = segment_colors[seg_id]
                    color2 = segment_colors[neighbor_root]
                    color_diff = np.linalg.norm(color1 - color2)
                    
                    if color_diff < color_threshold:
                        for s, root in merged_into.items():
                            if root == neighbor_root:
                                merged_into[s] = seg_id
        
        for seg_id in unique_segments:
            root = merged_into[seg_id]
            if root != seg_id:
                merged[segments == seg_id] = root
        
        return merged
    
    def _find_adjacent_segments(self, segments: np.ndarray) -> Dict[int, set]:
        """Find which segments are adjacent."""
        adjacency = {}
        h, w = segments.shape
        
        for y in range(h):
            for x in range(w - 1):
                s1, s2 = int(segments[y, x]), int(segments[y, x + 1])
                if s1 != s2:
                    adjacency.setdefault(s1, set()).add(s2)
                    adjacency.setdefault(s2, set()).add(s1)
        
        for y in range(h - 1):
            for x in range(w):
                s1, s2 = int(segments[y, x]), int(segments[y + 1, x])
                if s1 != s2:
                    adjacency.setdefault(s1, set()).add(s2)
                    adjacency.setdefault(s2, set()).add(s1)
        
        return adjacency
    
    def _segment_edge(self, image: np.ndarray) -> SegmentationResult:
        """Segment using traditional edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, 50, 150)
        
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(binary)
        edges = cv2.bitwise_or(edges, inverted)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        regions = cv2.bitwise_not(edges)
        num_labels, labeled = cv2.connectedComponents(regions, connectivity=4)
        
        filtered = np.zeros_like(labeled)
        valid_label = 1
        
        for label in range(1, num_labels):
            mask = labeled == label
            area = np.sum(mask)
            if area >= self.min_region_area:
                filtered[mask] = valid_label
                valid_label += 1
        
        # Fill unlabeled regions with nearest labeled region
        if np.any(filtered == 0):
            filtered = self._fill_unlabeled(filtered)
        
        # Ensure we have at least one region
        if valid_label == 1:
            logger.warning("No valid regions found, creating single region")
            filtered = np.ones_like(filtered, dtype=np.int32)
            return SegmentationResult(
                labeled_regions=filtered,
                num_regions=1,
                method_used=SegmentationMethod.EDGE,
                confidence=0.3,
                metadata={'edge_pixels': int(np.sum(edges > 0)), 'fallback': True}
            )
        
        return SegmentationResult(
            labeled_regions=filtered.astype(np.int32),
            num_regions=valid_label - 1,
            method_used=SegmentationMethod.EDGE,
            confidence=0.6,
            metadata={'edge_pixels': int(np.sum(edges > 0))}
        )
    
    def _fill_unlabeled(self, labeled: np.ndarray) -> np.ndarray:
        """Fill unlabeled pixels with nearest labeled region."""
        try:
            from scipy import ndimage
        except ImportError:
            logger.warning("scipy not available, skipping unlabeled fill")
            return labeled
        
        unlabeled = labeled == 0
        if not np.any(unlabeled):
            return labeled
        
        distances, indices = ndimage.distance_transform_edt(unlabeled, return_indices=True)
        filled = labeled.copy()
        filled[unlabeled] = labeled[indices[0][unlabeled], indices[1][unlabeled]]
        
        return filled


def create_detector(method: str = 'auto', min_region_area: int = 100, target_regions: int = 50) -> MLRegionDetector:
    """Factory function to create detector."""
    return MLRegionDetector(method=method, min_region_area=min_region_area, target_regions=target_regions)

