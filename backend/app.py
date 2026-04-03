"""FastAPI backend for 4-color theorem app."""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import numpy as np
from PIL import Image
import io
import base64
import cv2
import time
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Import our core modules
from core.region_detection import RegionDetector
from core.graph_builder import GraphBuilder
from core.four_color_solver import FourColorSolver
from core.photo_to_lineart import convert_photo_to_lineart
from core.ml_segmentation import MLRegionDetector, create_detector
from core.smart_preprocessing import SmartPreprocessor
from core.palette_suggester import suggest_palettes, PREDEFINED_PALETTES
from core.smart_color_ai import analyze_image_opencv
from core.photo_processor import process_photo  # NEW: unified photo pipeline
from utils.image_utils import optimize_image_size, validate_image_file, upscale_image

app = FastAPI(title="4-Color Theorem API")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for WSL compatibility
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our processors (will be recreated per request with appropriate settings)
graph_builder = GraphBuilder()
# Color solver will be created per request with appropriate max_colors


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "4-Color Theorem API", "version": "0.4.0"}

@app.get("/api/palettes")
async def get_palettes():
    """Get all available color palettes."""
    return {
        "palettes": [{
            "id": name,
            "name": palette.name,
            "colors": palette.colors,
            "colors_hex": [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in palette.colors],
            "style": palette.style.value,
            "description": palette.description
        } for name, palette in PREDEFINED_PALETTES.items()]
    }

@app.post("/api/analyze")
@limiter.limit("60/minute")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...)
):
    """Analyze image and suggest palettes."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        
        preprocessor = SmartPreprocessor(auto_enhance=False)
        result = preprocessor.process(image_np)
        analysis = result.analysis
        
        suggestions = suggest_palettes(image_np, n=6)
        
        return JSONResponse(content={
            "success": True,
            "analysis": {
                "image_type": analysis.image_type.value,
                "quality_score": analysis.quality_score,
                "contrast": analysis.contrast,
                "brightness": analysis.brightness,
                "sharpness": analysis.sharpness,
                "noise_level": analysis.noise_level,
                "dominant_colors": analysis.dominant_colors,
                "recommendations": analysis.recommendations
            },
            "suggested_palettes": suggestions
        })
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-colors")
@limiter.limit("120/minute")
async def analyze_colors(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Analyze image using server-side OpenCV (Layer 1).
    Returns pattern analysis and initial palette suggestions.
    Browser AI (Layer 2) can enhance these results client-side.
    """
    try:
        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 50MB)")
        
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run OpenCV analysis
        result = analyze_image_opencv(image_rgb)
        
        return JSONResponse(content={
            "success": True,
            "analysis": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        import traceback
        error_detail = str(e)
        if "analyze_image_opencv" in error_detail or "smart_color_ai" in error_detail:
            error_detail = f"Color analysis failed: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/suggest-palettes")
@limiter.limit("60/minute")
async def suggest_palettes_endpoint(
    request: Request,
    file: UploadFile = File(...),
    n: str = Form("5")
):
    """Get ML-based palette suggestions for an image."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        
        n_int = min(int(n), 10)  # Max 10 suggestions
        suggestions = suggest_palettes(image_np, n=n_int)
        
        return JSONResponse(content={
            "success": True,
            "suggestions": suggestions
        })
    except Exception as e:
        logger.error(f"Palette suggestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preview")
@limiter.limit("120/minute")
async def preview_image(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("vibrant"),
    convert_to_lineart: str = Form("false"),
    max_dimension: str = Form("400")
):
    """Generate low-resolution preview quickly. Rate: 30/min."""
    start_time = time.time()
    
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Validate image file
        is_valid, error_msg = validate_image_file(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        
        # Resize to preview size (max 400px)
        preview_max = min(int(max_dimension), 400)
        image_np, _ = optimize_image_size(image_np, max_dimension=preview_max)
        
        # Convert form parameters
        convert_lineart = convert_to_lineart.lower() in ("true", "1", "yes", "on")
        
        # Process through pipeline (no upscaling for preview)
        result = process_pipeline(
            image_np,
            style,
            stained_glass_enabled=False,  # No stained glass for preview
            convert_lineart=convert_lineart,
            line_thickness='medium',
            detail_level='detailed',
            contrast=1.0,
            size_metadata=None  # No upscaling for preview
        )
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return JSONResponse(content={
            "success": True,
            "preview": result["image"],
            "preview_size": (image_np.shape[1], image_np.shape[0]),
            "processing_time_ms": processing_time
        })
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = str(e)
        logger.error(f"Preview error: {error_detail}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/process")
@limiter.limit("60/minute")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("vibrant"),
    stained_glass: str = Form("false"),
    convert_to_lineart: str = Form("false"),
    line_thickness: str = Form("medium"),
    detail_level: str = Form("detailed"),
    contrast: str = Form("1.0"),
    use_ml: str = Form("false"),
    segmentation_method: str = Form("auto"),
    target_regions: str = Form("50"),
    use_five_colors: str = Form("false"),
    custom_colors: str = Form(None)
):
    """Process uploaded image through coloring pipeline (4 or 5 colors)."""
    try:
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Validate image file
        is_valid, error_msg = validate_image_file(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
            
        # Convert to numpy array
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        
        # Optimize image size for processing (increased to 3000px for better quality)
        image_np, size_metadata = optimize_image_size(image_np, max_dimension=3000)
        
        # Convert form parameters
        stained_glass_enabled = stained_glass.lower() in ("true", "1", "yes", "on")
        convert_lineart = convert_to_lineart.lower() in ("true", "1", "yes", "on")
        contrast_float = float(contrast) if contrast else 1.0
        
        # Process through pipeline
        use_ml_seg = use_ml.lower() in ("true", "1", "yes", "on")
        target_regions_int = int(target_regions) if target_regions.isdigit() else 50
        use_five_colors_bool = use_five_colors.lower() in ("true", "1", "yes", "on")
        
        # Parse custom colors if provided
        custom_colors_list = None
        if custom_colors:
            try:
                import json
                custom_colors_list = json.loads(custom_colors)
                # Validate colors format: should be list of [R, G, B] lists
                if isinstance(custom_colors_list, list) and len(custom_colors_list) > 0:
                    for color in custom_colors_list:
                        if not (isinstance(color, list) and len(color) == 3):
                            custom_colors_list = None
                            break
                else:
                    custom_colors_list = None
            except:
                custom_colors_list = None
        
        result = process_pipeline(
            image_np, 
            style, 
            stained_glass_enabled,
            convert_lineart=convert_lineart,
            line_thickness=line_thickness,
            detail_level=detail_level,
            contrast=contrast_float,
            size_metadata=size_metadata,
            use_ml_segmentation=use_ml_seg,
            segmentation_method=segmentation_method,
            target_regions=target_regions_int,
            use_five_colors=use_five_colors_bool,
            custom_colors=custom_colors_list
        )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = str(e)
        logger.error(f"Processing error: {error_detail}\n{traceback.format_exc()}")
        
        # Provide helpful error messages
        if "timeout" in error_detail.lower() or "time" in error_detail.lower():
            error_detail = "Processing took too long. Try a smaller image or simpler settings."
        elif "memory" in error_detail.lower():
            error_detail = "Image too large for processing. Please resize and try again."
        elif "format" in error_detail.lower() or "decode" in error_detail.lower():
            error_detail = "Invalid image format. Please use PNG, JPG, or JPEG."
        
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/preview-lineart")
async def preview_lineart(
    file: UploadFile = File(...),
    line_thickness: str = Form("medium"),
    detail_level: str = Form("detailed"),
    contrast: str = Form("1.0")
):
    """Preview line art conversion without full processing."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        
        contrast_float = float(contrast) if contrast else 1.0
        
        # Convert to line art
        line_art = convert_photo_to_lineart(
            image_np,
            line_thickness=line_thickness,
            detail_level=detail_level,
            contrast=contrast_float
        )
        
        # Convert to base64
        result_pil = Image.fromarray(line_art)
        buffered = io.BytesIO()
        result_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse(content={
            "success": True,
            "image": f"data:image/png;base64,{img_base64}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_pipeline(
    image_np: np.ndarray, 
    style: str, 
    stained_glass_enabled: bool = False,
    convert_lineart: bool = False,
    line_thickness: str = 'medium',
    detail_level: str = 'detailed',
    contrast: float = 1.0,
    size_metadata: Optional[Dict] = None,
    use_ml_segmentation: bool = False,
    segmentation_method: str = 'auto',
    target_regions: int = 50,
    use_five_colors: bool = False,
    custom_colors: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    """Main processing pipeline with optional ML segmentation."""
    
    max_colors = 5 if use_five_colors else 4
    
    # =====================================================================
    # PHOTO PATH: use unified pipeline (K-means → regions → colour)
    # This produces dramatically better results than line-art conversion
    # because region boundaries follow actual colour transitions in the
    # photo rather than being re-detected from a lossy intermediate.
    # =====================================================================
    if convert_lineart:
        logger.info("Using unified photo processing pipeline")
        
        # Build palette
        palette = _get_palette(style, custom_colors, max_colors)
        
        colored_image, photo_stats = process_photo(
            image_np,
            palette=palette,
            n_clusters=8,
            min_region_area=200,
            max_colors=max_colors,
        )
        
        # Apply stained glass effect if enabled
        if stained_glass_enabled:
            try:
                from effects.stained_glass import apply_stained_glass
                logger.info("Applying stained glass effect...")
                colored_image = apply_stained_glass(colored_image, intensity=0.8)
            except Exception as e:
                logger.warning(f"Stained glass failed: {e}")
        
        # Upscale result if image was resized for processing
        if size_metadata and size_metadata.get('resized', False):
            scale_factor = 1.0 / size_metadata['scale_factor']
            colored_image = upscale_image(colored_image, scale_factor, method='lanczos')
        
        # Convert result to base64
        result_pil = Image.fromarray(colored_image)
        buffered = io.BytesIO()
        result_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        stats_dict = {
            "regions": photo_stats["regions"],
            "colors_used": photo_stats["colors_used"],
            "max_colors_allowed": max_colors,
            "color_mode": f"{max_colors}-color",
            "graph_nodes": photo_stats["graph_nodes"],
            "graph_edges": photo_stats["graph_edges"],
            "pipeline": "unified_photo",
        }
        if size_metadata:
            stats_dict["image_resized"] = size_metadata.get('resized', False)
            stats_dict["original_size"] = size_metadata.get('original_size')
            stats_dict["processed_size"] = size_metadata.get('final_size')
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "stats": stats_dict,
        }
    
    # =====================================================================
    # LINE-ART / COLORING-BOOK PATH: traditional edge-based pipeline
    # (unchanged from before — works well for actual line art images)
    # =====================================================================
    
    # Step 1: Detect regions
    is_line_art = False  # convert_lineart is handled above now
    
    if use_ml_segmentation:
        # Use ML-based segmentation
        logger.info(f"Using ML segmentation (method: {segmentation_method}, target_regions: {target_regions})")
        ml_detector = create_detector(method=segmentation_method, min_region_area=100, target_regions=target_regions)
        seg_result = ml_detector.segment(image_np)
        labeled_regions = seg_result.labeled_regions
        num_regions = seg_result.num_regions
        
        # Convert to contours for stats
        contours = []
        for region_id in range(1, num_regions + 1):
            mask = (labeled_regions == region_id).astype(np.uint8) * 255
            region_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if region_contours:
                contours.extend(region_contours)
        
        stats = {
            "total_regions": num_regions,
            "edge_pixels": 0,
            "average_region_area": np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
        }
        
        # Build adjacency from labeled regions
        h, w = labeled_regions.shape
        adjacency = {}
        unique_regions = np.unique(labeled_regions)
        unique_regions = unique_regions[unique_regions > 0]
        
        for region in unique_regions:
            adjacency[int(region)] = set()
        
        # 4-connected adjacency check
        for y in range(h - 1):
            for x in range(w - 1):
                r1, r2 = int(labeled_regions[y, x]), int(labeled_regions[y, x + 1])
                if r1 != r2 and r1 > 0 and r2 > 0:
                    adjacency[r1].add(r2)
                    adjacency[r2].add(r1)
                
                r1, r2 = int(labeled_regions[y, x]), int(labeled_regions[y + 1, x])
                if r1 != r2 and r1 > 0 and r2 > 0:
                    adjacency[r1].add(r2)
                    adjacency[r2].add(r1)
    else:
        # Use traditional edge-based detection
        detector = RegionDetector(min_region_area=100, is_line_art=is_line_art)
        labeled_regions, contours, stats = detector.detect_regions(image_np)
        num_regions = len(contours)
        
        # Check complexity and warn if too many regions
        if num_regions > 2000:
            logger.warning(f"Very complex image detected: {num_regions} regions. Processing may be slow.")
            if num_regions > 5000:
                logger.warning("Too many regions, increasing minimum area threshold")
                detector = RegionDetector(min_region_area=300, is_line_art=is_line_art)
                labeled_regions, contours, stats = detector.detect_regions(image_np)
                num_regions = len(contours)
                logger.info(f"After threshold increase: {num_regions} regions")
        
        # Step 2: Build adjacency graph
        adjacency = detector.find_adjacent_regions(labeled_regions)
    
    graph = graph_builder.build_graph(adjacency)
    
    # Log graph complexity
    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    if graph.number_of_nodes() > 2000:
        logger.warning(f"Large graph detected: {graph.number_of_nodes()} nodes. Coloring may take time.")
    
    # Step 3: Solve coloring (4 or 5 colors)
    color_solver = FourColorSolver(max_colors=max_colors)
    coloring = color_solver.solve(graph)
    
    # Step 4: Apply colors
    colored_image = apply_colors(labeled_regions, coloring, style, custom_colors=custom_colors)
    
    # Step 5: Apply stained glass effect if enabled
    if stained_glass_enabled:
        from effects.stained_glass import apply_stained_glass
        print("Applying stained glass effect...")
        colored_image = apply_stained_glass(colored_image, labeled_regions, intensity=0.8)
    
    # Upscale result if image was resized for processing
    if size_metadata and size_metadata.get('resized', False):
        scale_factor = 1.0 / size_metadata['scale_factor']
        colored_image = upscale_image(colored_image, scale_factor, method='lanczos')
    
    # Convert result to base64
    result_pil = Image.fromarray(colored_image)
    buffered = io.BytesIO()
    result_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Get num_regions from contours if using traditional method, otherwise from ML result
    if use_ml_segmentation:
        num_regions_final = num_regions
    else:
        num_regions_final = len(contours)
    
    colors_used = len(set(coloring.values()))
    stats_dict = {
        "regions": num_regions_final,
        "colors_used": colors_used,
        "max_colors_allowed": max_colors,
        "color_mode": "5-color" if use_five_colors else "4-color",
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "pipeline": "line_art",
    }
    
    # Add size metadata if available
    if size_metadata:
        stats_dict["image_resized"] = size_metadata.get('resized', False)
        stats_dict["original_size"] = size_metadata.get('original_size')
        stats_dict["processed_size"] = size_metadata.get('final_size')
    
    return {
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "stats": stats_dict
    }


def _get_palette(
    style: str,
    custom_colors: Optional[List[List[int]]],
    max_colors: int,
) -> List[List[int]]:
    """Resolve palette from style name or custom colours."""
    if custom_colors and len(custom_colors) > 0:
        palette = list(custom_colors)
        while len(palette) < max_colors:
            palette.append(palette[-1] if palette else [128, 128, 128])
        return palette

    palettes = {
        "vibrant": [[220,20,60],[0,191,255],[50,205,50],[255,215,0]],
        "pastel":  [[255,182,193],[176,224,230],[152,251,152],[255,255,224]],
        "earth":   [[160,82,45],[107,142,35],[210,180,140],[139,90,43]],
        "monochrome": [[64,64,64],[128,128,128],[192,192,192],[224,224,224]],
        "ocean":   [[0,119,190],[64,224,208],[0,191,255],[25,25,112]],
        "sunset":  [[255,140,0],[255,20,147],[255,69,0],[255,192,203]],
        "forest":  [[34,139,34],[85,107,47],[107,142,35],[139,90,43]],
        "neon":    [[255,0,255],[0,255,255],[255,255,0],[0,255,0]],
    }
    return palettes.get(style, palettes["vibrant"])


def apply_colors(labeled_regions: np.ndarray, coloring: Dict[int, int], style: str, custom_colors: Optional[List[List[int]]] = None) -> np.ndarray:
    """Apply colors to regions based on style or custom colors."""
    # Use custom colors if provided
    if custom_colors and len(custom_colors) > 0:
        palette = custom_colors
        # Ensure we have at least 4 colors (or 5 if needed)
        while len(palette) < 4:
            palette.append(palette[-1] if palette else [128, 128, 128])
    else:
        # Define color palettes
        palettes = {
        "vibrant": [
            [220, 20, 60],    # Crimson
            [0, 191, 255],    # Deep Sky Blue
            [50, 205, 50],    # Lime Green
            [255, 215, 0]     # Gold
        ],
        "pastel": [
            [255, 182, 193], # Light Pink
            [176, 224, 230], # Powder Blue
            [152, 251, 152], # Pale Green
            [255, 255, 224]  # Light Yellow
        ],
        "earth": [
            [160, 82, 45],   # Sienna
            [107, 142, 35],  # Olive Drab
            [210, 180, 140], # Tan
            [139, 90, 43]    # Saddle Brown
        ],
        "monochrome": [
            [64, 64, 64],    # Dark Gray
            [128, 128, 128], # Gray
            [192, 192, 192], # Light Gray
            [224, 224, 224]  # Very Light Gray
        ],
        "ocean": [
            [0, 119, 190],   # Ocean Blue
            [64, 224, 208],  # Turquoise
            [0, 191, 255],    # Deep Sky Blue
            [25, 25, 112]     # Midnight Blue
        ],
        "sunset": [
            [255, 140, 0],   # Dark Orange
            [255, 20, 147],   # Deep Pink
            [255, 69, 0],     # Red Orange
            [255, 192, 203]  # Pink
        ],
        "forest": [
            [34, 139, 34],   # Forest Green
            [85, 107, 47],   # Dark Olive Green
            [107, 142, 35],  # Olive Drab
            [139, 90, 43]    # Saddle Brown
        ],
        "neon": [
            [255, 0, 255],   # Magenta
            [0, 255, 255],   # Cyan
            [255, 255, 0],   # Yellow
            [0, 255, 0]      # Lime
        ]
    }
    
    palette = palettes.get(style, palettes["vibrant"])
    
    # Create output image
    h, w = labeled_regions.shape
    colored = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
    
    # Apply colors to each region
    max_colors = len(palette)
    for region_id, color_id in coloring.items():
        if region_id > 0:  # Skip background (0)
            mask = labeled_regions == region_id
            colored[mask] = palette[color_id % max_colors]
    
    return colored

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)