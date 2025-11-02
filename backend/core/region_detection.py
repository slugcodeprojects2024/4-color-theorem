"""Detect closed regions in coloring book images."""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
import logging

logger = logging.getLogger(__name__)

class RegionDetector:
    def __init__(self, min_region_area: int = 100):
        """
        Initialize region detector.
        
        Args:
            min_region_area: Minimum area in pixels for a valid region
        """
        self.min_region_area = min_region_area
        
    def detect_regions(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
        """
        Detect closed regions in an image.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            labeled_regions: Image with labeled regions
            contours: List of contour arrays
            stats: Dictionary with detection statistics
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Detect edges
        edges = self._detect_edges(gray)
        
        # Clean up edges
        cleaned_edges = self._clean_edges(edges)
        
        # Find regions
        labeled_regions, contours = self._find_regions(cleaned_edges)
        
        # Compute statistics
        stats = {
            "total_regions": len(contours),
            "edge_pixels": np.sum(edges > 0),
            "average_region_area": np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
        }
        
        return labeled_regions, contours, stats
    
    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Detect edges in grayscale image."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Alternative: threshold-based edge detection for clean line art
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(binary)
        
        # Combine both methods
        combined = cv2.bitwise_or(edges, inverted)
        
        return combined
    
    def _clean_edges(self, edges: np.ndarray) -> np.ndarray:
        """Clean up edges by closing gaps and removing noise."""
        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        denoised = cv2.medianBlur(closed, 3)
        
        # Thin the edges
        thinned = cv2.ximgproc.thinning(denoised) if hasattr(cv2, 'ximgproc') else denoised
        
        return thinned
    
    def _find_regions(self, edges: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Find closed regions from edge image."""
        # Invert edges to get regions
        regions = cv2.bitwise_not(edges)
        
        # Find connected components
        num_labels, labeled = cv2.connectedComponents(regions, connectivity=4)
        
        # Find contours for each region
        contours = []
        valid_labels = []
        
        for label in range(1, num_labels):  # Skip background (0)
            mask = (labeled == label).astype(np.uint8) * 255
            region_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if region_contours:
                contour = region_contours[0]
                area = cv2.contourArea(contour)
                
                if area >= self.min_region_area:
                    contours.append(contour)
                    valid_labels.append(label)
        
        # Relabel regions to be consecutive
        relabeled = np.zeros_like(labeled)
        for new_label, old_label in enumerate(valid_labels, 1):
            relabeled[labeled == old_label] = new_label
            
        return relabeled, contours
    
    def find_adjacent_regions(self, labeled_regions: np.ndarray) -> Dict[int, Set[int]]:
        """
        Find which regions are adjacent to each other.
        
        Args:
            labeled_regions: Image with labeled regions
            
        Returns:
            Dictionary mapping region_id to set of adjacent region_ids
        """
        adjacency = {}
        h, w = labeled_regions.shape
        
        # Check all neighboring pixels
        for y in range(h):
            for x in range(w):
                current = int(labeled_regions[y, x])
                if current == 0:  # Skip background
                    continue
                    
                if current not in adjacency:
                    adjacency[current] = set()
                
                # Check 4-connected neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor = int(labeled_regions[ny, nx])
                        if neighbor != 0 and neighbor != current:
                            adjacency[current].add(neighbor)
        
        return adjacency