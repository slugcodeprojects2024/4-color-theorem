"""
Stained Glass Effect for Backend Processing
Creates authentic stained glass appearance with lead lines between regions
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StainedGlassEffect:
    """Create authentic stained glass effect with lead lines and glass texture."""
    
    def __init__(self):
        self.lead_color = (20, 20, 25)  # Dark gray/black for lead lines
        self.lead_thickness = 3
        
    def apply_effect(self, 
                    image: np.ndarray, 
                    labeled_regions: Optional[np.ndarray] = None,
                    intensity: float = 0.8) -> np.ndarray:
        """
        Apply stained glass effect to an image.
        
        Args:
            image: Input image (RGB)
            labeled_regions: Optional labeled regions array for accurate lead lines
            intensity: Effect intensity (0.0 to 1.0)
            
        Returns:
            Image with stained glass effect
        """
        result = image.copy()
        h, w = result.shape[:2]
        
        # Step 1: Create lead lines
        if labeled_regions is not None:
            lead_lines, lead_alpha = self.create_lead_lines_from_regions(labeled_regions, intensity)
        else:
            lead_lines, lead_alpha = self.create_lead_lines_from_edges(result, intensity)
        
        # Step 2: Apply glass texture to the colored regions
        result = self.apply_glass_texture(result, intensity)
        
        # Step 3: Add lighting effects
        result = self.apply_lighting_effects(result, intensity)
        
        # Step 4: Overlay lead lines
        result = self.overlay_lead_lines(result, (lead_lines, lead_alpha))
        
        # Step 5: Final glass effects
        result = self.apply_final_effects(result, intensity)
        
        return result
    
    def create_lead_lines_from_regions(self, labeled_regions: np.ndarray, intensity: float):
        """
        Create lead lines from labeled regions (most accurate method).
        """
        h, w = labeled_regions.shape
        lead_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Find boundaries between different regions
        # Use morphological gradient to detect region boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        gradient = cv2.morphologyEx(labeled_regions.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        
        # Threshold to create binary lead lines
        _, lead_mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY)
        
        # Dilate to make lead lines thicker
        thickness_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (self.lead_thickness, self.lead_thickness))
        lead_mask = cv2.dilate(lead_mask, thickness_kernel, iterations=1)
        
        # Apply Gaussian blur for softer edges
        lead_mask = cv2.GaussianBlur(lead_mask, (5, 5), 1)
        
        # Create colored lead lines with metallic appearance
        lead_lines = np.zeros((h, w, 3), dtype=np.uint8)
        mask_3d = np.stack([lead_mask, lead_mask, lead_mask], axis=2)
        
        # Base lead color
        lead_lines[mask_3d > 128] = self.lead_color
        
        # Add metallic highlights
        highlight_mask = cv2.Laplacian(lead_mask, cv2.CV_64F)
        highlight_mask = np.abs(highlight_mask)
        highlight_mask = (highlight_mask * 0.3).astype(np.uint8)
        
        lead_lines[:, :, 0] = np.minimum(lead_lines[:, :, 0] + highlight_mask, 255)
        lead_lines[:, :, 1] = np.minimum(lead_lines[:, :, 1] + highlight_mask, 255)
        lead_lines[:, :, 2] = np.minimum(lead_lines[:, :, 2] + highlight_mask + 5, 255)  # Slight blue tint
        
        # Scale intensity
        lead_alpha = (lead_mask.astype(np.float32) / 255.0) * intensity
        
        return lead_lines, lead_alpha
    
    def create_lead_lines_from_edges(self, image: np.ndarray, intensity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lead lines using edge detection (fallback method).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to make them thicker
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (self.lead_thickness, self.lead_thickness))
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Blur for softer appearance
        thick_edges = cv2.GaussianBlur(thick_edges, (5, 5), 1)
        
        # Create colored lead lines
        h, w = image.shape[:2]
        lead_lines = np.zeros((h, w, 3), dtype=np.uint8)
        mask_3d = np.stack([thick_edges, thick_edges, thick_edges], axis=2)
        
        lead_lines[mask_3d > 128] = self.lead_color
        
        # Alpha channel for blending
        lead_alpha = (thick_edges.astype(np.float32) / 255.0) * intensity
        
        return lead_lines, lead_alpha
    
    def apply_glass_texture(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """
        Apply glass-like texture to the image.
        """
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        # Create glass texture using Perlin-like noise
        texture = self.generate_glass_texture(w, h)
        
        # Apply texture as overlay
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - intensity * 0.3) + \
                             texture * intensity * 0.3 * 255
        
        # Add subtle color variations (glass imperfections)
        noise = np.random.normal(0, 10 * intensity, (h, w, 3))
        result += noise
        
        # Add ripple effect (simulating hand-blown glass)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ripple = np.sin(np.sqrt(x**2 + y**2) * 0.01) * 10 * intensity
        result[:, :, :] += ripple[:, :, np.newaxis]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def generate_glass_texture(self, width: int, height: int) -> np.ndarray:
        """
        Generate a glass-like texture pattern.
        """
        # Create base noise
        texture = np.zeros((height, width), dtype=np.float32)
        
        # Multi-scale noise for realistic glass texture
        scales = [4, 8, 16, 32]
        weights = [0.5, 0.25, 0.15, 0.1]
        
        for scale, weight in zip(scales, weights):
            # Create noise at this scale
            noise = np.random.randn(height // scale + 1, width // scale + 1)
            # Upscale to full resolution
            noise_full = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
            texture += noise_full * weight
        
        # Normalize to 0-1 range
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        
        # Apply smoothing for glass-like appearance
        texture = cv2.GaussianBlur(texture, (5, 5), 1)
        
        return texture
    
    def apply_lighting_effects(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """
        Apply lighting effects to simulate light passing through glass.
        """
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        # Create light gradient (simulating directional light)
        light_gradient = np.zeros((h, w, 3), dtype=np.float32)
        
        # Top-left light source
        for y in range(h):
            for x in range(w):
                distance = np.sqrt((x / w) ** 2 + (y / h) ** 2)
                brightness = max(0, 1 - distance * 0.7)
                light_gradient[y, x] = [255, 250, 200] * brightness * intensity * 0.3
        
        # Apply light gradient with soft light blending
        result = cv2.addWeighted(result, 1.0, light_gradient, 0.5, 0)
        
        # Add specular highlights
        highlights = self.create_specular_highlights(w, h, intensity)
        result = cv2.add(result, highlights)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def create_specular_highlights(self, width: int, height: int, intensity: float) -> np.ndarray:
        """
        Create specular highlights for glass surface.
        """
        highlights = np.zeros((height, width, 3), dtype=np.float32)
        
        # Add random bright spots
        num_highlights = int(5 + intensity * 10)
        for _ in range(num_highlights):
            cx = np.random.randint(0, width)
            cy = np.random.randint(0, height)
            radius = np.random.randint(20, 100)
            
            # Create radial gradient for highlight
            y, x = np.ogrid[:height, :width]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            
            # Gaussian falloff
            distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            falloff = np.exp(-(distances ** 2) / (radius ** 2))
            
            highlight_intensity = falloff * intensity * 50
            highlights[:, :, 0] += highlight_intensity
            highlights[:, :, 1] += highlight_intensity
            highlights[:, :, 2] += highlight_intensity * 1.1  # Slight blue tint
        
        return highlights
    
    def overlay_lead_lines(self, image: np.ndarray, lead_data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Overlay lead lines onto the image.
        """
        lead_lines, lead_alpha = lead_data
        result = image.copy().astype(np.float32)
        
        # Expand alpha to 3 channels
        alpha_3d = np.stack([lead_alpha, lead_alpha, lead_alpha], axis=2)
        
        # Blend lead lines with image
        result = result * (1 - alpha_3d) + lead_lines.astype(np.float32) * alpha_3d
        
        return result.astype(np.uint8)
    
    def apply_final_effects(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """
        Apply final effects for authentic stained glass appearance.
        """
        result = image.copy()
        h, w = result.shape[:2]
        
        # Increase color saturation (stained glass has rich colors)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        result[:, :, 1] *= (1 + intensity * 0.3)  # Increase saturation
        result[:, :, 2] *= (1 + intensity * 0.1)  # Slight brightness boost
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
        
        # Add vignette effect (darker edges)
        vignette = np.ones((h, w), dtype=np.float32)
        cv, ch = w // 2, h // 2
        for y in range(h):
            for x in range(w):
                distance = np.sqrt((x - cv) ** 2 + (y - ch) ** 2)
                max_dist = np.sqrt(cv ** 2 + ch ** 2)
                vignette[y, x] = 1 - (distance / max_dist) * intensity * 0.3
        
        vignette = np.stack([vignette, vignette, vignette], axis=2)
        result = (result.astype(np.float32) * vignette).astype(np.uint8)
        
        # Final contrast adjustment
        result = cv2.convertScaleAbs(result, alpha=1 + intensity * 0.2, beta=0)
        
        return result


def apply_stained_glass(image: np.ndarray, 
                        labeled_regions: Optional[np.ndarray] = None,
                        intensity: float = 0.8) -> np.ndarray:
    """
    Convenience function to apply stained glass effect.
    
    Args:
        image: Input image (RGB)
        labeled_regions: Optional labeled regions for accurate lead lines
        intensity: Effect intensity (0.0 to 1.0)
        
    Returns:
        Image with stained glass effect
    """
    effect = StainedGlassEffect()
    return effect.apply_effect(image, labeled_regions, intensity)
