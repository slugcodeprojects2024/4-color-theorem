"""Apply stained glass effect to colored images using advanced image processing."""
import numpy as np
import cv2
from typing import Tuple
import random
from skimage import filters, morphology, exposure, feature
from skimage.filters import gaussian, unsharp_mask
from skimage.morphology import disk, square


def apply_stained_glass_effect(
    colored_image: np.ndarray,
    labeled_regions: np.ndarray,
    intensity: float = 0.8
) -> np.ndarray:
    """
    Apply a realistic stained glass effect to the colored image.
    
    Args:
        colored_image: RGB image with colored regions (uint8, 0-255)
        labeled_regions: Labeled regions array (same shape as image, single channel)
        intensity: Intensity of the effect (0.0 to 1.0)
        
    Returns:
        Stained glass effect applied image (RGB, uint8)
    """
    result = colored_image.copy().astype(np.float32)
    h, w = result.shape[:2]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Fast, lightweight backend version - detailed effects moved to frontend WebGL
    # Step 1: Quick texture (simplified, faster)
    result = _add_glass_texture_fast(result, intensity)
    
    # Step 2: Basic lighting (simplified)
    result = _add_lighting_effects_fast(result, labeled_regions, intensity)
    
    # Step 3: Quick edge highlights (basic)
    result = _add_edge_highlights_fast(result, labeled_regions, intensity)
    
    # Ensure values are in valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def _add_glass_texture(image: np.ndarray, intensity: float) -> np.ndarray:
    """
    Add glass-like texture with irregular light patterns using advanced filtering.
    Uses Perlin-like noise and bilateral filtering for realistic glass texture.
    """
    h, w = image.shape[:2]
    
    # Convert to float for processing
    result = image.astype(np.float32)
    
    # Create multi-scale Perlin-like noise for organic glass texture
    texture = np.zeros((h, w, 3), dtype=np.float32)
    
    # Large-scale variations (like light passing through thick glass)
    # Use Gaussian blur on noise for smooth, organic patterns
    noise_large = np.random.rand(h // 8, w // 8, 3).astype(np.float32)
    noise_large = cv2.resize(noise_large, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_large = gaussian(noise_large, sigma=20, channel_axis=2, preserve_range=True)
    texture += noise_large * 20 * intensity
    
    # Medium-scale variations (glass imperfections and bubbles)
    noise_medium = np.random.rand(h // 16, w // 16, 3).astype(np.float32)
    noise_medium = cv2.resize(noise_medium, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_medium = gaussian(noise_medium, sigma=8, channel_axis=2, preserve_range=True)
    texture += noise_medium * 12 * intensity
    
    # Fine-scale noise (surface texture and ripples)
    noise_fine = np.random.rand(h, w, 3).astype(np.float32)
    noise_fine = gaussian(noise_fine, sigma=2, channel_axis=2, preserve_range=True)
    texture += noise_fine * 6 * intensity
    
    # Apply texture with edge-preserving blending using bilateral filter
    # This maintains sharp edges while adding texture
    result_with_texture = result + texture
    result_with_texture = np.clip(result_with_texture, 0, 255).astype(np.uint8)
    
    # Use bilateral filter to blend texture while preserving edges
    result = cv2.bilateralFilter(result_with_texture, d=9, sigmaColor=75, sigmaSpace=75)
    result = result.astype(np.float32)
    
    return result


def _add_lighting_effects(
    image: np.ndarray,
    labeled_regions: np.ndarray,
    intensity: float
) -> np.ndarray:
    """
    Add advanced lighting effects: highlights, shadows, and Fresnel effect.
    Uses Gaussian gradients and exposure adjustments for realistic lighting.
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    # Create directional lighting gradient (light from top-left)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Normalize coordinates to [-1, 1]
    x_norm = (x_coords - w / 2) / (w / 2)
    y_norm = (y_coords - h / 2) / (h / 2)
    
    # Create smooth lighting gradient using Gaussian-like falloff
    light_gradient = 1.0 - np.sqrt(x_norm**2 + y_norm**2) * 0.2 * intensity
    light_gradient = np.clip(light_gradient, 0.7, 1.3)
    
    # Add directional component (brighter in top-left)
    directional = 1.0 + (-x_norm * 0.3 - y_norm * 0.3) * intensity * 0.4
    lighting = light_gradient * directional
    
    # Add per-region lighting variation (each glass piece has different properties)
    unique_regions = np.unique(labeled_regions)
    np.random.seed(42)  # Reproducible
    
    for region_id in unique_regions:
        if region_id == 0:
            continue
        
        mask = labeled_regions == region_id
        if np.sum(mask) == 0:
            continue
        
        # Random lighting variation (thickness, clarity, etc.)
        region_lighting = 1.0 + (np.random.rand() - 0.5) * 0.25 * intensity
        
        # Smooth the lighting variation within region using Gaussian
        region_mask_float = mask.astype(np.float32)
        smoothed_mask = gaussian(region_mask_float, sigma=5, preserve_range=True)
        smoothed_mask = smoothed_mask / smoothed_mask.max() if smoothed_mask.max() > 0 else smoothed_mask
        
        lighting[mask] = lighting[mask] * (1.0 + (region_lighting - 1.0) * smoothed_mask[mask])
    
    # Apply lighting with exposure adjustment for realism
    result = result * lighting[:, :, np.newaxis]
    
    # Add Fresnel effect (edges are brighter due to light refraction)
    # This is a key characteristic of glass
    edge_distance = cv2.distanceTransform(
        (labeled_regions > 0).astype(np.uint8),
        cv2.DIST_L2,
        5
    )
    # Normalize and invert (closer to edge = brighter)
    edge_distance_norm = 1.0 - np.clip(edge_distance / 10.0, 0, 1)
    fresnel_effect = 1.0 + edge_distance_norm * 0.15 * intensity
    
    result = result * fresnel_effect[:, :, np.newaxis]
    result = np.clip(result, 0, 255)
    
    return result


def _add_color_variation(
    image: np.ndarray,
    labeled_regions: np.ndarray,
    intensity: float
) -> np.ndarray:
    """
    Add color variation within regions to simulate the natural variation in stained glass.
    Real stained glass has slight color variations and gradients.
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    unique_regions = np.unique(labeled_regions)
    np.random.seed(42)  # Reproducible randomness
    
    for region_id in unique_regions:
        if region_id == 0:  # Skip background
            continue
        
        mask = labeled_regions == region_id
        if np.sum(mask) == 0:
            continue
        
        # Get region bounds
        region_coords = np.where(mask)
        if len(region_coords[0]) == 0:
            continue
        
        min_y, max_y = region_coords[0].min(), region_coords[0].max()
        min_x, max_x = region_coords[1].min(), region_coords[1].max()
        
        # Create gradient within region (simulating light passing through)
        region_h, region_w = max_y - min_y + 1, max_x - min_x + 1
        
        # Radial gradient from center (like light source)
        center_y, center_x = (min_y + max_y) / 2, (min_x + max_x) / 2
        y_grad, x_grad = np.meshgrid(
            np.arange(min_y, max_y + 1),
            np.arange(min_x, max_x + 1),
            indexing='ij'
        )
        
        # Distance from center
        dist = np.sqrt((y_grad - center_y)**2 + (x_grad - center_x)**2)
        max_dist = np.sqrt(region_h**2 + region_w**2) / 2
        gradient = 1.0 - (dist / (max_dist + 1)) * 0.15 * intensity
        
        # Add subtle color shift (like real glass)
        color_shift = (np.random.rand(3) - 0.5) * 10 * intensity
        
        # Apply to region
        region_mask = mask[min_y:max_y+1, min_x:max_x+1]
        if region_mask.shape[0] > 0 and region_mask.shape[1] > 0:
            gradient_masked = gradient[region_mask]
            for c in range(3):
                result[min_y:max_y+1, min_x:max_x+1, c][region_mask] = \
                    result[min_y:max_y+1, min_x:max_x+1, c][region_mask] * gradient_masked + \
                    color_shift[c] * gradient_masked
    
    result = np.clip(result, 0, 255)
    return result


def _add_edge_highlights(
    image: np.ndarray,
    labeled_regions: np.ndarray,
    intensity: float
) -> np.ndarray:
    """
    Add advanced edge highlights using morphological operations and distance transforms.
    Simulates lead came (metal framework) and light refraction at edges.
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    # Use Canny edge detection on region boundaries for cleaner edges
    # Create a mask of region boundaries
    region_boundaries = np.zeros((h, w), dtype=np.uint8)
    
    # Find edges using morphological gradient (more accurate than pixel checking)
    for region_id in np.unique(labeled_regions):
        if region_id == 0:
            continue
        
        mask = (labeled_regions == region_id).astype(np.uint8) * 255
        # Morphological gradient finds edges
        kernel = disk(2)
        gradient = morphology.binary_dilation(mask, kernel).astype(np.uint8) - \
                   morphology.binary_erosion(mask, kernel).astype(np.uint8)
        region_boundaries = cv2.bitwise_or(region_boundaries, gradient)
    
    # Use distance transform for smooth edge falloff
    dist_transform = cv2.distanceTransform(
        255 - region_boundaries,
        cv2.DIST_L2,
        5
    )
    
    # Create smooth highlight gradient (brighter closer to edge)
    edge_highlight_map = np.clip(1.0 - dist_transform / 8.0, 0, 1)
    edge_highlight_map = gaussian(edge_highlight_map, sigma=1.5, preserve_range=True)
    
    # Add bright highlight to edges (light refraction)
    highlight_strength = 0.5 * intensity
    for c in range(3):
        result[:, :, c] = np.minimum(
            result[:, :, c] + edge_highlight_map * 255 * highlight_strength,
            255
        )
    
    # Add lead came (dark center line) using morphological operations
    # Thicken the boundary, then erode to get center
    kernel_lead = disk(2)
    thick_boundaries = morphology.binary_dilation(
        (region_boundaries > 0).astype(bool),
        kernel_lead
    )
    lead_center = morphology.binary_erosion(
        thick_boundaries,
        kernel_lead
    )
    
    # Darken the lead came
    lead_strength = 0.4 * intensity
    lead_darkening = np.array([40, 40, 40], dtype=np.float32) * lead_strength
    
    for c in range(3):
        result[:, :, c][lead_center] = np.maximum(
            result[:, :, c][lead_center] - lead_darkening[c],
            0
        )
    
    result = np.clip(result, 0, 255)
    return result


def _add_glass_texture_fast(image: np.ndarray, intensity: float) -> np.ndarray:
    """Fast, simplified texture addition."""
    h, w = image.shape[:2]
    result = image.copy()
    
    # Simple noise texture (much faster than multi-scale)
    noise = np.random.rand(h, w, 3).astype(np.float32) * 10 * intensity
    result = result + noise
    result = np.clip(result, 0, 255)
    
    return result


def _add_lighting_effects_fast(
    image: np.ndarray,
    labeled_regions: np.ndarray,
    intensity: float
) -> np.ndarray:
    """Fast, simplified lighting effects."""
    h, w = image.shape[:2]
    result = image.copy()
    
    # Simple directional lighting gradient
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_norm = (x_coords - w / 2) / (w / 2)
    y_norm = (y_coords - h / 2) / (h / 2)
    
    # Simple lighting
    lighting = 1.0 + (-x_norm * 0.2 - y_norm * 0.2) * intensity * 0.3
    result = result * lighting[:, :, np.newaxis]
    result = np.clip(result, 0, 255)
    
    return result


def _add_edge_highlights_fast(
    image: np.ndarray,
    labeled_regions: np.ndarray,
    intensity: float
) -> np.ndarray:
    """Fast, simplified edge highlights."""
    result = image.copy()
    h, w = image.shape[:2]
    
    # Simple edge detection using difference
    edge_mask = np.zeros((h, w), dtype=bool)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            current = labeled_regions[y, x]
            if (labeled_regions[y-1, x] != current or
                labeled_regions[y+1, x] != current or
                labeled_regions[y, x-1] != current or
                labeled_regions[y, x+1] != current):
                if current != 0:
                    edge_mask[y, x] = True
    
    # Simple highlight
    kernel = np.ones((2, 2), np.uint8)
    edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    highlight = 255 * 0.3 * intensity
    for c in range(3):
        result[:, :, c][edge_dilated] = np.minimum(
            result[:, :, c][edge_dilated] + highlight,
            255
        )
    
    result = np.clip(result, 0, 255)
    return result

