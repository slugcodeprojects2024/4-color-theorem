"""
Smart Image Preprocessing Module

Automatically analyzes and enhances images for optimal segmentation:
- Auto contrast/brightness adjustment
- Noise reduction with edge preservation
- Image quality assessment
- Photo vs Line Art detection
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ImageType(Enum):
    LINE_ART = "line_art"
    PHOTO = "photo"
    ILLUSTRATION = "illustration"
    UNKNOWN = "unknown"


@dataclass
class ImageAnalysis:
    """Analysis results for an image."""
    image_type: ImageType
    quality_score: float
    contrast: float
    brightness: float
    sharpness: float
    noise_level: float
    dominant_colors: list
    has_transparency: bool
    recommendations: list


@dataclass 
class PreprocessingResult:
    """Result from preprocessing."""
    image: np.ndarray
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    analysis: ImageAnalysis
    applied_operations: list


class SmartPreprocessor:
    """Intelligent image preprocessor."""
    
    def __init__(self, target_size: Optional[int] = None, auto_enhance: bool = True):
        self.target_size = target_size
        self.auto_enhance = auto_enhance
    
    def process(self, image: np.ndarray) -> PreprocessingResult:
        """Analyze and preprocess image."""
        original_size = (image.shape[1], image.shape[0])
        applied_ops = []
        
        analysis = self.analyze(image)
        
        if self.auto_enhance:
            image, ops = self._auto_enhance(image, analysis)
            applied_ops.extend(ops)
        
        if self.target_size:
            image, resized = self._smart_resize(image, self.target_size)
            if resized:
                applied_ops.append(f"resize_to_{self.target_size}")
        
        return PreprocessingResult(
            image=image,
            original_size=original_size,
            final_size=(image.shape[1], image.shape[0]),
            analysis=analysis,
            applied_operations=applied_ops
        )
    
    def analyze(self, image: np.ndarray) -> ImageAnalysis:
        """Analyze image characteristics."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image_type = self._detect_image_type(image, gray)
        contrast = self._calculate_contrast(gray)
        brightness = self._calculate_brightness(gray)
        sharpness = self._calculate_sharpness(gray)
        noise_level = self._estimate_noise(gray)
        dominant_colors = self._find_dominant_colors(image)
        quality_score = self._calculate_quality_score(contrast, brightness, sharpness, noise_level)
        recommendations = self._generate_recommendations(image_type, contrast, brightness, sharpness, noise_level)
        
        return ImageAnalysis(
            image_type=image_type,
            quality_score=quality_score,
            contrast=contrast,
            brightness=brightness,
            sharpness=sharpness,
            noise_level=noise_level,
            dominant_colors=dominant_colors,
            has_transparency=False,
            recommendations=recommendations
        )
    
    def _detect_image_type(self, image: np.ndarray, gray: np.ndarray) -> ImageType:
        """Detect whether image is line art, photo, or illustration."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        
        dark_peak = hist[:50].sum()
        light_peak = hist[200:].sum()
        mid_values = hist[50:200].sum()
        
        if dark_peak + light_peak > 0.7 and mid_values < 0.3:
            return ImageType.LINE_ART
        
        color_std = np.std(image, axis=(0, 1)).mean()
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if color_std > 40 and edge_density < 0.15:
            return ImageType.PHOTO
        
        if edge_density > 0.05:
            return ImageType.ILLUSTRATION
        
        return ImageType.UNKNOWN
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        return min(np.std(gray) / 75, 1.0)
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        return np.mean(gray) / 255
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return min(laplacian.var() / 1000, 1.0)
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        sample = gray[h//4:3*h//4, w//4:3*w//4]
        median = np.median(sample)
        mad = np.median(np.abs(sample.astype(float) - median))
        return min(mad / 20, 1.0)
    
    def _find_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> list:
        pixels = image.reshape(-1, 3).astype(np.float32)
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        
        colors = []
        for idx in sorted_indices:
            color = centers[idx].astype(int).tolist()
            percentage = counts[idx] / len(labels) * 100
            colors.append({
                'rgb': color,
                'hex': '#{:02x}{:02x}{:02x}'.format(*color),
                'percentage': round(percentage, 1)
            })
        
        return colors
    
    def _calculate_quality_score(self, contrast, brightness, sharpness, noise) -> float:
        contrast_score = contrast
        brightness_score = 1 - abs(brightness - 0.5) * 2
        sharpness_score = sharpness
        noise_score = 1 - noise
        
        return round(contrast_score * 0.3 + brightness_score * 0.2 + sharpness_score * 0.3 + noise_score * 0.2, 2)
    
    def _generate_recommendations(self, image_type, contrast, brightness, sharpness, noise) -> list:
        recs = []
        if contrast < 0.3:
            recs.append("increase_contrast")
        if brightness < 0.3:
            recs.append("increase_brightness")
        elif brightness > 0.7:
            recs.append("decrease_brightness")
        if sharpness < 0.2 and image_type != ImageType.LINE_ART:
            recs.append("sharpen")
        if noise > 0.3:
            recs.append("denoise")
        if image_type == ImageType.PHOTO:
            recs.append("consider_line_art_conversion")
        return recs
    
    def _auto_enhance(self, image: np.ndarray, analysis: ImageAnalysis) -> Tuple[np.ndarray, list]:
        result = image.copy()
        applied = []
        
        if "increase_contrast" in analysis.recommendations:
            result = self._enhance_contrast(result)
            applied.append("contrast_enhanced")
        
        if "denoise" in analysis.recommendations:
            result = cv2.bilateralFilter(result, 9, 75, 75)
            applied.append("denoised")
        
        return result, applied
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _smart_resize(self, image: np.ndarray, max_dimension: int) -> Tuple[np.ndarray, bool]:
        h, w = image.shape[:2]
        if w <= max_dimension and h <= max_dimension:
            return image, False
        
        scale = max_dimension / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), True


def preprocess_for_coloring(image: np.ndarray, target_size: Optional[int] = 2000, enhance: bool = True) -> PreprocessingResult:
    """Convenience function to preprocess an image for coloring."""
    return SmartPreprocessor(target_size=target_size, auto_enhance=enhance).process(image)

