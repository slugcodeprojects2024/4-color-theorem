"""
ML-Based Color Palette Suggestions

Analyzes image content and suggests appropriate color palettes.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PaletteStyle(Enum):
    VIBRANT = "vibrant"
    PASTEL = "pastel"
    EARTH = "earth"
    MONOCHROME = "monochrome"
    COMPLEMENTARY = "complementary"
    WARM = "warm"
    COOL = "cool"
    CUSTOM = "custom"


@dataclass
class ColorPalette:
    """A color palette with metadata."""
    name: str
    colors: List[Tuple[int, int, int]]
    style: PaletteStyle
    description: str
    score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'colors': self.colors,
            'colors_hex': [self._rgb_to_hex(c) for c in self.colors],
            'style': self.style.value,
            'description': self.description,
            'score': self.score
        }
    
    @staticmethod
    def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)


PREDEFINED_PALETTES = {
    'vibrant': ColorPalette('Vibrant', [(220, 20, 60), (0, 191, 255), (50, 205, 50), (255, 215, 0)], PaletteStyle.VIBRANT, 'Bold, saturated colors'),
    'pastel': ColorPalette('Pastel Dream', [(255, 182, 193), (176, 224, 230), (152, 251, 152), (255, 255, 224)], PaletteStyle.PASTEL, 'Soft, gentle tones'),
    'earth': ColorPalette('Earth Tones', [(160, 82, 45), (107, 142, 35), (210, 180, 140), (139, 90, 43)], PaletteStyle.EARTH, 'Natural, organic colors'),
    'ocean': ColorPalette('Ocean Blue', [(0, 105, 148), (72, 202, 228), (144, 224, 239), (202, 240, 248)], PaletteStyle.COOL, 'Cool ocean tones'),
    'sunset': ColorPalette('Sunset Glow', [(255, 140, 0), (255, 20, 147), (255, 69, 0), (255, 192, 203)], PaletteStyle.WARM, 'Warm sunset colors'),
    'forest': ColorPalette('Forest', [(34, 139, 34), (85, 107, 47), (139, 90, 43), (154, 205, 50)], PaletteStyle.EARTH, 'Lush forest greens'),
    'neon': ColorPalette('Neon Pop', [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0)], PaletteStyle.VIBRANT, 'Electric neon colors'),
    'monochrome': ColorPalette('Grayscale', [(64, 64, 64), (128, 128, 128), (192, 192, 192), (224, 224, 224)], PaletteStyle.MONOCHROME, 'Classic grayscale'),
}


class PaletteSuggester:
    """Suggests color palettes based on image content."""
    
    def __init__(self):
        self.palettes = PREDEFINED_PALETTES
    
    def suggest(self, image: np.ndarray, n_suggestions: int = 5) -> List[ColorPalette]:
        """Suggest palettes for an image."""
        analysis = self._analyze_image(image)
        
        scored_palettes = []
        for name, palette in self.palettes.items():
            score = self._score_palette(palette, analysis)
            palette_copy = ColorPalette(
                name=palette.name, colors=palette.colors, style=palette.style,
                description=palette.description, score=score
            )
            scored_palettes.append(palette_copy)
        
        scored_palettes.sort(key=lambda p: p.score, reverse=True)
        
        custom = self._generate_custom_palette(image, analysis)
        if custom:
            scored_palettes.insert(0, custom)
        
        return scored_palettes[:n_suggestions]
    
    def _analyze_image(self, image: np.ndarray) -> Dict:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        avg_h = np.mean(hsv[:, :, 0])
        avg_s = np.mean(hsv[:, :, 1])
        avg_v = np.mean(hsv[:, :, 2])
        
        warm_mask = (hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 150)
        warm_ratio = np.mean(warm_mask)
        
        dominant = self._extract_dominant_colors(image, n=5)
        scene_type = self._detect_scene_type(image, hsv)
        
        return {
            'avg_hue': avg_h, 'avg_saturation': avg_s, 'avg_value': avg_v,
            'warm_ratio': warm_ratio, 'dominant_colors': dominant, 'scene_type': scene_type,
            'is_high_saturation': avg_s > 100, 'is_bright': avg_v > 150, 'is_dark': avg_v < 100,
        }
    
    def _extract_dominant_colors(self, image: np.ndarray, n: int = 5) -> List[Tuple[int, int, int]]:
        pixels = image.reshape(-1, 3).astype(np.float32)
        if len(pixels) > 10000:
            pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        unique, counts = np.unique(labels, return_counts=True)
        return [tuple(centers[idx].astype(int).tolist()) for idx in np.argsort(-counts)]
    
    def _detect_scene_type(self, image: np.ndarray, hsv: np.ndarray) -> str:
        h, w = image.shape[:2]
        
        top_region = hsv[:h//3, :, :]
        blue_ratio = np.mean((top_region[:, :, 0] > 90) & (top_region[:, :, 0] < 130))
        green_ratio = np.mean((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 50))
        warm_ratio = np.mean((hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 150))
        
        if blue_ratio > 0.3:
            return 'ocean'
        elif green_ratio > 0.3:
            return 'nature'
        elif warm_ratio > 0.5:
            return 'warm'
        return 'neutral'
    
    def _score_palette(self, palette: ColorPalette, analysis: Dict) -> float:
        score = 0.5
        scene_type = analysis['scene_type']
        
        if scene_type == 'ocean' and palette.style == PaletteStyle.COOL:
            score += 0.3
        elif scene_type == 'nature' and palette.style == PaletteStyle.EARTH:
            score += 0.3
        elif scene_type == 'warm' and palette.style == PaletteStyle.WARM:
            score += 0.3
        
        if analysis['is_high_saturation'] and palette.style == PaletteStyle.VIBRANT:
            score += 0.15
        elif not analysis['is_high_saturation'] and palette.style == PaletteStyle.PASTEL:
            score += 0.15
        
        return min(score, 1.0)
    
    def _generate_custom_palette(self, image: np.ndarray, analysis: Dict) -> Optional[ColorPalette]:
        dominant = analysis['dominant_colors']
        if len(dominant) < 4:
            return None
        
        colors = list(dominant[:4])
        
        return ColorPalette(
            name='From Image', colors=colors, style=PaletteStyle.CUSTOM,
            description='Generated from your image colors', score=0.85
        )


def suggest_palettes(image: np.ndarray, n: int = 5) -> List[Dict]:
    """Convenience function to get palette suggestions."""
    return [p.to_dict() for p in PaletteSuggester().suggest(image, n)]

