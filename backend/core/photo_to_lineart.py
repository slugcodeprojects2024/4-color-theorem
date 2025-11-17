"""
Photo to Line Art Converter
Converts regular photos into coloring book style line art
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PhotoToLineArt:
    """Convert photos to line art suitable for coloring."""
    
    def __init__(self):
        self.default_line_thickness = 'medium'
        self.default_detail_level = 'detailed'
        self.default_contrast = 1.0
    
    def convert(
        self,
        image: np.ndarray,
        line_thickness: str = 'medium',
        detail_level: str = 'detailed',
        contrast: float = 1.0
    ) -> np.ndarray:
        """
        Convert a photo to line art.
        
        Args:
            image: Input RGB image
            line_thickness: 'thin', 'medium', or 'thick'
            detail_level: 'simple' or 'detailed'
            contrast: Contrast multiplier (0.5 to 2.0)
            
        Returns:
            Line art image (grayscale, suitable for coloring)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Step 1: Enhance contrast
        gray = self._adjust_contrast(gray, contrast)
        
        # Step 2: Apply bilateral filter to reduce noise while preserving edges
        if detail_level == 'detailed':
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        else:
            filtered = cv2.bilateralFilter(gray, 15, 100, 100)  # More smoothing for simple
        
        # Step 3: Edge detection using multiple techniques
        edges_canny = self._canny_edges(filtered, detail_level)
        edges_adaptive = self._adaptive_threshold_edges(filtered, detail_level)
        
        # Step 4: Combine edge detection methods
        combined = self._combine_edges(edges_canny, edges_adaptive, detail_level)
        
        # Step 5: Adjust line thickness
        line_art = self._adjust_line_thickness(combined, line_thickness)
        
        # Step 6: Invert (black lines on white background)
        line_art = 255 - line_art
        
        # Step 7: Clean up (remove small noise)
        line_art = self._cleanup_noise(line_art, detail_level)
        
        # Convert back to RGB for consistency
        line_art_rgb = cv2.cvtColor(line_art, cv2.COLOR_GRAY2RGB)
        
        return line_art_rgb
    
    def _adjust_contrast(self, image: np.ndarray, contrast: float) -> np.ndarray:
        """Adjust image contrast."""
        contrast = np.clip(contrast, 0.5, 2.0)
        if contrast == 1.0:
            return image
        
        # Apply contrast adjustment
        alpha = contrast  # Contrast control (1.0 = no change)
        beta = 0  # Brightness control
        
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    
    def _canny_edges(self, image: np.ndarray, detail_level: str) -> np.ndarray:
        """Extract edges using Canny edge detection."""
        if detail_level == 'detailed':
            # More sensitive for detailed images
            low_threshold = 50
            high_threshold = 150
        else:
            # Less sensitive for simple images
            low_threshold = 80
            high_threshold = 200
        
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges
    
    def _adaptive_threshold_edges(self, image: np.ndarray, detail_level: str) -> np.ndarray:
        """Extract edges using adaptive thresholding."""
        # Apply Gaussian blur first
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        if detail_level == 'detailed':
            block_size = 11
            c_value = 2
        else:
            block_size = 15
            c_value = 3
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c_value
        )
        
        # Extract edges from threshold
        edges = cv2.Canny(thresh, 50, 150)
        return edges
    
    def _combine_edges(
        self,
        edges1: np.ndarray,
        edges2: np.ndarray,
        detail_level: str
    ) -> np.ndarray:
        """Combine multiple edge detection results."""
        # Weight the different methods
        if detail_level == 'detailed':
            weight1 = 0.6  # Canny
            weight2 = 0.4  # Adaptive
        else:
            weight1 = 0.4  # Canny
            weight2 = 0.6  # Adaptive (better for simple images)
        
        # Combine
        combined = cv2.addWeighted(edges1, weight1, edges2, weight2, 0)
        
        # Threshold to binary
        _, combined = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)
        
        return combined
    
    def _adjust_line_thickness(
        self,
        edges: np.ndarray,
        line_thickness: str
    ) -> np.ndarray:
        """Adjust the thickness of lines."""
        if line_thickness == 'thin':
            kernel_size = 1
            iterations = 0
        elif line_thickness == 'medium':
            kernel_size = 3
            iterations = 1
        else:  # thick
            kernel_size = 5
            iterations = 2
        
        if iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            thickened = cv2.dilate(edges, kernel, iterations=iterations)
        else:
            thickened = edges
        
        return thickened
    
    def _cleanup_noise(self, image: np.ndarray, detail_level: str) -> np.ndarray:
        """Remove small noise artifacts."""
        # Remove very small connected components
        if detail_level == 'simple':
            min_area = 50  # Remove smaller components for simple mode
        else:
            min_area = 20  # Keep more detail
        
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for valid contours
        mask = np.zeros_like(image)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply mask
        cleaned = cv2.bitwise_and(image, mask)
        
        return cleaned


def convert_photo_to_lineart(
    image: np.ndarray,
    line_thickness: str = 'medium',
    detail_level: str = 'detailed',
    contrast: float = 1.0
) -> np.ndarray:
    """
    Convenience function to convert photo to line art.
    
    Args:
        image: Input RGB image
        line_thickness: 'thin', 'medium', or 'thick'
        detail_level: 'simple' or 'detailed'
        contrast: Contrast multiplier (0.5 to 2.0)
        
    Returns:
        Line art image (RGB, black lines on white background)
    """
    converter = PhotoToLineArt()
    return converter.convert(image, line_thickness, detail_level, contrast)

