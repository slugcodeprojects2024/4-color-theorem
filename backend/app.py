"""FastAPI backend for 4-color theorem app."""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import io
import base64
import cv2
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Import our core modules
from core.region_detection import RegionDetector
from core.graph_builder import GraphBuilder
from core.four_color_solver import FourColorSolver
from core.photo_to_lineart import convert_photo_to_lineart
from utils.image_utils import optimize_image_size, validate_image_file, upscale_image

app = FastAPI(title="4-Color Theorem API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for WSL compatibility
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our processors
region_detector = RegionDetector()
graph_builder = GraphBuilder()
color_solver = FourColorSolver()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "4-Color Theorem API", "version": "0.1.0"}

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    style: str = Form("vibrant"),
    stained_glass: str = Form("false"),
    convert_to_lineart: str = Form("false"),
    line_thickness: str = Form("medium"),
    detail_level: str = Form("detailed"),
    contrast: str = Form("1.0")
):
    """Process uploaded image through 4-color pipeline."""
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
        
        # Optimize image size for processing
        image_np, size_metadata = optimize_image_size(image_np, max_dimension=2000)
        
        # Convert form parameters
        stained_glass_enabled = stained_glass.lower() in ("true", "1", "yes", "on")
        convert_lineart = convert_to_lineart.lower() in ("true", "1", "yes", "on")
        contrast_float = float(contrast) if contrast else 1.0
        
        # Process through pipeline
        result = process_pipeline(
            image_np, 
            style, 
            stained_glass_enabled,
            convert_lineart=convert_lineart,
            line_thickness=line_thickness,
            detail_level=detail_level,
            contrast=contrast_float,
            size_metadata=size_metadata
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
    size_metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """Main processing pipeline."""
    
    # Step 0: Convert photo to line art if requested
    if convert_lineart:
        print(f"Converting photo to line art (thickness: {line_thickness}, detail: {detail_level}, contrast: {contrast})")
        image_np = convert_photo_to_lineart(
            image_np,
            line_thickness=line_thickness,
            detail_level=detail_level,
            contrast=contrast
        )
    
    # Step 1: Detect regions
    labeled_regions, contours, stats = region_detector.detect_regions(image_np)
    
    # Step 2: Build adjacency graph
    adjacency = region_detector.find_adjacent_regions(labeled_regions)
    graph = graph_builder.build_graph(adjacency)
    
    # Step 3: Solve 4-coloring
    coloring = color_solver.solve(graph)
    
    # Step 4: Apply colors
    colored_image = apply_colors(labeled_regions, coloring, style)
    
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
    
    stats_dict = {
        "regions": len(contours),
        "colors_used": len(set(coloring.values())),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges()
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

def apply_colors(labeled_regions: np.ndarray, coloring: Dict[int, int], style: str) -> np.ndarray:
    """Apply colors to regions based on style."""
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
    for region_id, color_id in coloring.items():
        if region_id > 0:  # Skip background (0)
            mask = labeled_regions == region_id
            colored[mask] = palette[color_id % 4]
    
    return colored

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)