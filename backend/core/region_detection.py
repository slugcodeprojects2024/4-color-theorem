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
            is_line_art: Whether processing line art (needs larger min area)
        """
        # For line art, use larger minimum area to reduce complexity
        if is_line_art:
            self.min_region_area = max(min_region_area, 200)  # At least 200px for line art
        else:
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
        
        print(f"Debug - Image shape: {image.shape}")
        print(f"Debug - Gray shape: {gray.shape}")
        
        # Detect edges
        edges = self._detect_edges(gray)
        print(f"Debug - Edge pixels found: {np.sum(edges > 0)}")
        
        # Save debug image
        cv2.imwrite('debug_edges.png', edges)
        
        # Clean up edges
        cleaned_edges = self._clean_edges(edges)
        
        # Find regions
        labeled_regions, contours = self._find_regions(cleaned_edges)
        print(f"Debug - Number of regions found: {len(contours)}")
        print(f"Debug - Unique labels: {np.unique(labeled_regions)}")
        
        # Save debug labeled image
        cv2.imwrite('debug_labeled.png', labeled_regions * 50)  # Scale for visibility
        
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
        Find which regions are adjacent to each other, accounting for thin separating lines.
        Optimized for large numbers of regions.
        
        Args:
            labeled_regions: Image with labeled regions
            
        Returns:
            Dictionary mapping region_id to set of adjacent region_ids
        """
        adjacency = {}
        h, w = labeled_regions.shape
        
        # Get unique regions
        unique_regions = np.unique(labeled_regions)
        unique_regions = unique_regions[unique_regions > 0]  # Skip background
        num_regions = len(unique_regions)
        
        logger.info(f"Finding adjacencies for {num_regions} regions")
        
        # Initialize adjacency dict
        for region in unique_regions:
            adjacency[int(region)] = set()
        
        # For very large numbers of regions, use a more efficient method
        if num_regions > 1000:
            logger.info("Using optimized adjacency detection for large graph")
            # Use 4-connected pixel check (faster for large graphs)
            search_radius = 5  # Smaller radius for speed
            sample_rate = max(1, int(num_regions / 500))  # Sample more sparsely for large graphs
        else:
            search_radius = 10
            sample_rate = 5
        
        edges_found = 0
        
        # For each region, find its boundary pixels
        for idx, region in enumerate(unique_regions):
            if idx % 100 == 0:
                logger.info(f"Processing region {idx}/{num_regions}")
            
            region = int(region)
            # Find boundary pixels of this region
            mask = (labeled_regions == region).astype(np.uint8)
            
            # Find contour of this region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            # Sample points along the contour (more sparse for large graphs)
            contour = contours[0]
            step = max(1, len(contour) // (100 * sample_rate))  # Limit samples per region
            for point in contour[::step]:
                x, y = point[0]
                
                # Look in a radius around this boundary point
                for dy in range(-search_radius, search_radius + 1, 2):  # Step by 2 for speed
                    for dx in range(-search_radius, search_radius + 1, 2):
                        if dx == 0 and dy == 0:
                            continue
                            
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            neighbor = int(labeled_regions[ny, nx])
                            if neighbor > 0 and neighbor != region:
                                if neighbor not in adjacency[region]:
                                    adjacency[region].add(neighbor)
                                    edges_found += 1
        
        total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
        logger.info(f"Found {total_edges} unique edges from {edges_found} edge checks")
        
        return adjacency