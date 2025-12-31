"""
Smart Color AI - Server-side OpenCV Analysis (Layer 1)

Provides instant pattern-based analysis as fallback/enhancement.
Works with any hosting, no ML dependencies required.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ImageStyle(Enum):
    CHILDREN_SIMPLE = "children_simple"
    CHILDREN_DETAILED = "children_detailed"
    ADULT_MANDALA = "adult_mandala"
    KAWAII = "kawaii"
    GEOMETRIC = "geometric"
    ZENTANGLE = "zentangle"
    REALISTIC = "realistic"
    UNKNOWN = "unknown"


@dataclass
class PatternAnalysis:
    """Results from OpenCV pattern analysis."""
    style: ImageStyle
    has_symmetry: bool
    symmetry_score: float
    edge_density: float
    complexity_score: float
    dominant_colors: List[Tuple[int, int, int]]
    has_circular_patterns: bool
    has_nature_elements: bool
    estimated_subjects: List[str]
    confidence: float


# Color palettes for different detected patterns
STYLE_PALETTES = {
    ImageStyle.CHILDREN_SIMPLE: [
        {"name": "Cheerful Primary", "colors": [(255, 0, 0), (255, 255, 0), (0, 0, 255), (0, 255, 0)],
         "description": "Bright primary colors for simple designs"},
        {"name": "Soft Pastels", "colors": [(255, 182, 193), (176, 224, 230), (255, 255, 224), (221, 160, 221)],
         "description": "Gentle colors for young children"},
    ],
    ImageStyle.KAWAII: [
        {"name": "Kawaii Pink", "colors": [(255, 182, 193), (255, 105, 180), (255, 218, 185), (176, 224, 230)],
         "description": "Cute pastel pinks and soft colors"},
        {"name": "Sweet Candy", "colors": [(255, 182, 193), (255, 160, 122), (255, 218, 185), (221, 160, 221)],
         "description": "Candy-inspired sweet tones"},
    ],
    ImageStyle.ADULT_MANDALA: [
        {"name": "Jewel Tones", "colors": [(128, 0, 128), (0, 128, 128), (178, 34, 34), (255, 215, 0)],
         "description": "Rich, sophisticated colors"},
        {"name": "Zen Garden", "colors": [(106, 90, 205), (100, 149, 237), (60, 179, 113), (255, 215, 0)],
         "description": "Calming, meditative palette"},
        {"name": "Royal Purple", "colors": [(75, 0, 130), (138, 43, 226), (255, 20, 147), (255, 215, 0)],
         "description": "Regal purple tones"},
    ],
    ImageStyle.GEOMETRIC: [
        {"name": "Bold Contrast", "colors": [(220, 20, 60), (0, 0, 139), (255, 215, 0), (34, 139, 34)],
         "description": "High-contrast geometric colors"},
        {"name": "Modern Minimal", "colors": [(0, 0, 0), (255, 255, 255), (255, 99, 71), (70, 130, 180)],
         "description": "Clean, modern palette"},
    ],
    ImageStyle.ZENTANGLE: [
        {"name": "Ink & Accent", "colors": [(0, 0, 0), (128, 128, 128), (70, 130, 180), (255, 215, 0)],
         "description": "Subtle with pops of color"},
        {"name": "Earth Zen", "colors": [(139, 90, 43), (160, 82, 45), (85, 107, 47), (210, 180, 140)],
         "description": "Natural earth tones"},
    ],
}

# Generic palettes for any image
UNIVERSAL_PALETTES = [
    {"name": "Nature's Palette", "colors": [(135, 206, 235), (34, 139, 34), (255, 215, 0), (139, 90, 43)],
     "description": "Sky, grass, sun, and earth"},
    {"name": "Warm Sunset", "colors": [(255, 99, 71), (255, 140, 0), (255, 215, 0), (255, 182, 193)],
     "description": "Warm, inviting tones"},
    {"name": "Cool Ocean", "colors": [(0, 105, 148), (72, 202, 228), (64, 224, 208), (176, 224, 230)],
     "description": "Cool blues and teals"},
    {"name": "Forest Walk", "colors": [(34, 139, 34), (85, 107, 47), (139, 90, 43), (255, 215, 0)],
     "description": "Deep forest greens"},
    {"name": "Vibrant Pop", "colors": [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0)],
     "description": "Bold, eye-catching colors"},
]


class OpenCVAnalyzer:
    """
    OpenCV-based image analyzer for instant pattern detection.
    This is Layer 1 - fast, works anywhere, no ML required.
    """
    
    def analyze(self, image: np.ndarray) -> PatternAnalysis:
        """Analyze image patterns using OpenCV."""
        if len(image.shape) == 2:
            gray = image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        h, w = gray.shape[:2]
        
        # Core analysis
        edge_density = self._analyze_edges(gray)
        symmetry_score = self._analyze_symmetry(gray)
        complexity = self._analyze_complexity(gray)
        has_circles = self._detect_circles(gray)
        dominant_colors = self._extract_dominant_colors(image)
        
        # Estimate style
        style = self._estimate_style(edge_density, symmetry_score, complexity, has_circles)
        
        # Estimate subjects based on patterns
        subjects = self._estimate_subjects(gray, has_circles, symmetry_score, edge_density)
        
        # Calculate confidence
        confidence = self._calculate_confidence(style, symmetry_score, edge_density)
        
        return PatternAnalysis(
            style=style,
            has_symmetry=symmetry_score > 0.5,
            symmetry_score=symmetry_score,
            edge_density=edge_density,
            complexity_score=complexity,
            dominant_colors=dominant_colors,
            has_circular_patterns=has_circles,
            has_nature_elements=self._detect_nature_elements(gray),
            estimated_subjects=subjects,
            confidence=confidence
        )
    
    def _analyze_edges(self, gray: np.ndarray) -> float:
        """Calculate edge density."""
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def _analyze_symmetry(self, gray: np.ndarray) -> float:
        """Calculate symmetry score (0-1)."""
        h, w = gray.shape
        
        # Horizontal symmetry
        left = gray[:, :w//2]
        right = cv2.flip(gray[:, w//2:], 1)
        min_w = min(left.shape[1], right.shape[1])
        h_diff = np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
        h_sym = 1 - (np.mean(h_diff) / 255)
        
        # Vertical symmetry
        top = gray[:h//2, :]
        bottom = cv2.flip(gray[h//2:, :], 0)
        min_h = min(top.shape[0], bottom.shape[0])
        v_diff = np.abs(top[:min_h, :].astype(float) - bottom[:min_h, :].astype(float))
        v_sym = 1 - (np.mean(v_diff) / 255)
        
        # Radial symmetry (for mandalas)
        center = (w // 2, h // 2)
        radial_scores = []
        for angle in [45, 60, 90, 120]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h))
            diff = np.abs(gray.astype(float) - rotated.astype(float))
            radial_scores.append(1 - np.mean(diff) / 255)
        
        r_sym = max(radial_scores) if radial_scores else 0
        
        return max(h_sym, v_sym, r_sym)
    
    def _analyze_complexity(self, gray: np.ndarray) -> float:
        """Analyze pattern complexity (0-1)."""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Contour complexity
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        
        # Normalize
        complexity = min(edge_density * 5 + (contour_count / 1000), 1.0)
        return complexity
    
    def _detect_circles(self, gray: np.ndarray) -> bool:
        """Detect circular patterns."""
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=10, maxRadius=min(gray.shape)//3
        )
        return circles is not None and len(circles[0]) >= 2
    
    def _detect_nature_elements(self, gray: np.ndarray) -> bool:
        """Detect potential nature elements (rough heuristic)."""
        h, w = gray.shape
        
        # Check for sky-like region (bright top)
        top_brightness = np.mean(gray[:h//3, :])
        
        # Check for ground-like region (edges at bottom)
        bottom_edges = cv2.Canny(gray[2*h//3:, :], 50, 150)
        bottom_edge_density = np.sum(bottom_edges > 0) / bottom_edges.size
        
        return top_brightness > 200 or bottom_edge_density > 0.05
    
    def _extract_dominant_colors(self, image: np.ndarray, n: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using k-means."""
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Sample for performance
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        # Sort by frequency
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        
        return [tuple(centers[idx].astype(int).tolist()) for idx in sorted_indices]
    
    def _estimate_style(self, edge_density: float, symmetry: float, 
                        complexity: float, has_circles: bool) -> ImageStyle:
        """Estimate image style from patterns."""
        
        # High symmetry + circles = likely mandala
        if symmetry > 0.7 and has_circles:
            return ImageStyle.ADULT_MANDALA
        
        # High symmetry + high complexity = zentangle or mandala
        if symmetry > 0.6 and complexity > 0.5:
            return ImageStyle.ADULT_MANDALA
        
        # High complexity alone = zentangle
        if complexity > 0.7:
            return ImageStyle.ZENTANGLE
        
        # Low edge density + simple = children's simple
        if edge_density < 0.08 and complexity < 0.3:
            return ImageStyle.CHILDREN_SIMPLE
        
        # Medium complexity with circles = could be kawaii
        if has_circles and complexity < 0.5 and edge_density < 0.12:
            return ImageStyle.KAWAII
        
        # High symmetry alone = geometric
        if symmetry > 0.5:
            return ImageStyle.GEOMETRIC
        
        # Default
        if complexity > 0.4:
            return ImageStyle.CHILDREN_DETAILED
        
        return ImageStyle.UNKNOWN
    
    def _estimate_subjects(self, gray: np.ndarray, has_circles: bool,
                           symmetry: float, edge_density: float) -> List[str]:
        """Estimate likely subjects (rough heuristics)."""
        subjects = []
        
        if symmetry > 0.7:
            subjects.append("symmetrical_pattern")
        
        if has_circles:
            subjects.append("circular_elements")
        
        if self._detect_nature_elements(gray):
            subjects.append("possible_nature_scene")
        
        # These are just hints - the browser AI will do better
        if edge_density > 0.15:
            subjects.append("detailed_drawing")
        elif edge_density < 0.05:
            subjects.append("simple_drawing")
        
        return subjects if subjects else ["unknown"]
    
    def _calculate_confidence(self, style: ImageStyle, symmetry: float, 
                             edge_density: float) -> float:
        """Calculate analysis confidence."""
        base = 0.4
        
        if style != ImageStyle.UNKNOWN:
            base += 0.2
        
        if symmetry > 0.6:
            base += 0.15
        
        if 0.05 < edge_density < 0.2:
            base += 0.1
        
        return min(base, 0.85)  # Cap at 0.85 - browser AI can do better


def get_palettes_for_analysis(analysis: PatternAnalysis) -> List[Dict]:
    """Get color palette suggestions based on analysis."""
    palettes = []
    
    # Style-specific palettes
    if analysis.style in STYLE_PALETTES:
        for p in STYLE_PALETTES[analysis.style]:
            palettes.append({
                **p,
                "score": 0.9 * analysis.confidence,
                "source": "pattern_analysis",
                "is_smart": True
            })
    
    # Add palette based on dominant colors
    if analysis.dominant_colors:
        palettes.append({
            "name": "From Image Colors",
            "colors": analysis.dominant_colors[:4],
            "description": "Extracted from image",
            "score": 0.7,
            "source": "color_extraction",
            "is_smart": False
        })
    
    # Add universal palettes with lower scores
    for p in UNIVERSAL_PALETTES[:3]:
        palettes.append({
            **p,
            "score": 0.5,
            "source": "universal",
            "is_smart": False
        })
    
    # Sort by score
    palettes.sort(key=lambda x: x["score"], reverse=True)
    
    return palettes[:6]


def analyze_image_opencv(image: np.ndarray) -> Dict:
    """Main entry point for OpenCV analysis."""
    analyzer = OpenCVAnalyzer()
    analysis = analyzer.analyze(image)
    palettes = get_palettes_for_analysis(analysis)
    
    return {
        "layer": "opencv",
        "style": analysis.style.value,
        "symmetry_score": float(analysis.symmetry_score),
        "complexity_score": float(analysis.complexity_score),
        "has_symmetry": bool(analysis.has_symmetry),
        "has_circular_patterns": bool(analysis.has_circular_patterns),
        "estimated_subjects": list(analysis.estimated_subjects),
        "dominant_colors": [[int(c[0]), int(c[1]), int(c[2])] for c in analysis.dominant_colors],
        "confidence": float(analysis.confidence),
        "suggested_palettes": palettes,
        "note": "Pattern-based analysis. Enable browser AI for better results."
    }

