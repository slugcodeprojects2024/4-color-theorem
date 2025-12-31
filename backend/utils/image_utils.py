"""
Image utility functions for optimization and processing
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Maximum dimensions for processing (to prevent timeouts)
MAX_PROCESSING_DIMENSION = 3000  # Increased for better quality on large images
MAX_UPLOAD_DIMENSION = 10000


def optimize_image_size(
    image: np.ndarray,
    max_dimension: int = MAX_PROCESSING_DIMENSION,
    maintain_aspect: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Resize image if it exceeds maximum dimensions.
    
    Args:
        image: Input image (RGB numpy array)
        max_dimension: Maximum width or height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Tuple of (resized_image, metadata_dict)
    """
    h, w = image.shape[:2]
    original_size = (w, h)
    
    # Check if resizing is needed
    if w <= max_dimension and h <= max_dimension:
        return image, {
            'resized': False,
            'original_size': original_size,
            'final_size': original_size,
            'scale_factor': 1.0
        }
    
    # Calculate new dimensions
    if maintain_aspect:
        if w > h:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
        else:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
    else:
        new_w = min(w, max_dimension)
        new_h = min(h, max_dimension)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    scale_factor = new_w / w
    
    logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h} (scale: {scale_factor:.2f})")
    
    return resized, {
        'resized': True,
        'original_size': original_size,
        'final_size': (new_w, new_h),
        'scale_factor': scale_factor
    }


def validate_image_file(file_contents: bytes, max_size_mb: float = 50.0) -> Tuple[bool, Optional[str]]:
    """
    Validate image file before processing.
    
    Args:
        file_contents: Image file bytes
        max_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file size
    size_mb = len(file_contents) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"Image file too large ({size_mb:.1f}MB). Maximum size is {max_size_mb}MB."
    
    # Try to open and validate image
    try:
        from io import BytesIO
        img = Image.open(BytesIO(file_contents))
        img.verify()
        
        # Check dimensions
        img = Image.open(BytesIO(file_contents))  # Reopen after verify
        w, h = img.size
        
        if w > MAX_UPLOAD_DIMENSION or h > MAX_UPLOAD_DIMENSION:
            return False, f"Image dimensions too large ({w}x{h}). Maximum is {MAX_UPLOAD_DIMENSION}x{MAX_UPLOAD_DIMENSION}."
        
        if w < 10 or h < 10:
            return False, f"Image dimensions too small ({w}x{h}). Minimum is 10x10."
        
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def upscale_image(
    image: np.ndarray,
    scale_factor: float,
    method: str = 'lanczos'
) -> np.ndarray:
    """
    Upscale image by a scale factor.
    
    Args:
        image: Input image
        scale_factor: Factor to upscale (e.g., 2.0 for 2x)
        method: Upscaling method ('lanczos', 'cubic', 'linear')
        
    Returns:
        Upscaled image
    """
    h, w = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    if method == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    elif method == 'cubic':
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR
    
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    return upscaled

