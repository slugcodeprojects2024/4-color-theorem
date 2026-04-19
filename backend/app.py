"""FastAPI backend for 4-color theorem app."""
import os
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
import time
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our core modules
from core.region_detection import RegionDetector
from core.graph_builder import GraphBuilder
from core.four_color_solver import FourColorSolver
from core.photo_to_lineart import convert_photo_to_lineart
from core.photo_processor import process_photo
from utils.image_utils import optimize_image_size, validate_image_file, upscale_image

app = FastAPI(title="4-Color Theorem API")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration via env var (comma-separated list of allowed origins)
# Defaults to "*" for local dev. In production set ALLOWED_ORIGINS to your frontend URL.
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
if allowed_origins_env == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

logger.info(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize processors
graph_builder = GraphBuilder()

# Mount static files only if the directory exists
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files mounted at /static")
else:
    logger.info("No static/ directory, skipping static mount")


@app.get("/")
async def root():
    return {"message": "4-Color Theorem API", "version": "0.5.0"}


@app.get("/health")
async def health():
    """Health check endpoint for Fly.io."""
    return {"status": "ok"}


@app.post("/api/preview")
@limiter.limit("120/minute")
async def preview_image(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("vibrant"),
    convert_to_lineart: str = Form("false"),
    max_dimension: str = Form("400")
):
    """Generate low-resolution preview quickly."""
    start_time = time.time()

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        is_valid, error_msg = validate_image_file(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)

        preview_max = min(int(max_dimension), 400)
        image_np, _ = optimize_image_size(image_np, max_dimension=preview_max)

        convert_lineart = convert_to_lineart.lower() in ("true", "1", "yes", "on")

        result = process_pipeline(
            image_np,
            style,
            stained_glass_enabled=False,
            convert_lineart=convert_lineart,
            line_thickness='medium',
            detail_level='detailed',
            contrast=1.0,
            size_metadata=None
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
    use_five_colors: str = Form("false"),
    custom_colors: str = Form(None)
):
    """Process uploaded image through coloring pipeline (4 or 5 colors)."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        is_valid, error_msg = validate_image_file(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)

        image_np, size_metadata = optimize_image_size(image_np, max_dimension=3000)

        stained_glass_enabled = stained_glass.lower() in ("true", "1", "yes", "on")
        convert_lineart = convert_to_lineart.lower() in ("true", "1", "yes", "on")
        contrast_float = float(contrast) if contrast else 1.0
        use_five_colors_bool = use_five_colors.lower() in ("true", "1", "yes", "on")

        # Parse custom colors if provided
        custom_colors_list = None
        if custom_colors:
            try:
                import json
                custom_colors_list = json.loads(custom_colors)
                if isinstance(custom_colors_list, list) and len(custom_colors_list) > 0:
                    for color in custom_colors_list:
                        if not (isinstance(color, list) and len(color) == 3):
                            custom_colors_list = None
                            break
                else:
                    custom_colors_list = None
            except Exception:
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

        line_art = convert_photo_to_lineart(
            image_np,
            line_thickness=line_thickness,
            detail_level=detail_level,
            contrast=contrast_float
        )

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
    use_five_colors: bool = False,
    custom_colors: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    """Main processing pipeline."""

    max_colors = 5 if use_five_colors else 4

    # PHOTO PATH: unified pipeline (K-means → regions → colour)
    if convert_lineart:
        logger.info("Using unified photo processing pipeline")

        palette = _get_palette(style, custom_colors, max_colors)

        colored_image, photo_stats = process_photo(
            image_np,
            palette=palette,
            n_clusters=8,
            min_region_area=200,
            max_colors=max_colors,
        )

        if stained_glass_enabled:
            try:
                from effects.stained_glass import apply_stained_glass
                logger.info("Applying stained glass effect...")
                colored_image = apply_stained_glass(colored_image, intensity=0.8)
            except Exception as e:
                logger.warning(f"Stained glass failed: {e}")

        if size_metadata and size_metadata.get('resized', False):
            scale_factor = 1.0 / size_metadata['scale_factor']
            colored_image = upscale_image(colored_image, scale_factor, method='lanczos')

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

    # LINE-ART / COLORING-BOOK PATH
    detector = RegionDetector(min_region_area=100, is_line_art=False)
    labeled_regions, contours, stats = detector.detect_regions(image_np)
    num_regions = len(contours)

    if num_regions > 2000:
        logger.warning(f"Very complex image: {num_regions} regions")
        if num_regions > 5000:
            logger.warning("Too many regions, increasing minimum area threshold")
            detector = RegionDetector(min_region_area=300, is_line_art=False)
            labeled_regions, contours, stats = detector.detect_regions(image_np)
            num_regions = len(contours)

    adjacency = detector.find_adjacent_regions(labeled_regions)
    graph = graph_builder.build_graph(adjacency)

    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    color_solver = FourColorSolver(max_colors=max_colors)
    coloring = color_solver.solve(graph)

    colored_image = apply_colors(labeled_regions, coloring, style, custom_colors=custom_colors)

    if stained_glass_enabled:
        from effects.stained_glass import apply_stained_glass
        logger.info("Applying stained glass effect...")
        colored_image = apply_stained_glass(colored_image, labeled_regions, intensity=0.8)

    if size_metadata and size_metadata.get('resized', False):
        scale_factor = 1.0 / size_metadata['scale_factor']
        colored_image = upscale_image(colored_image, scale_factor, method='lanczos')

    result_pil = Image.fromarray(colored_image)
    buffered = io.BytesIO()
    result_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    colors_used = len(set(coloring.values()))
    stats_dict = {
        "regions": num_regions,
        "colors_used": colors_used,
        "max_colors_allowed": max_colors,
        "color_mode": f"{max_colors}-color",
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "pipeline": "line_art",
    }

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
        "vibrant": [[220, 20, 60], [0, 191, 255], [50, 205, 50], [255, 215, 0]],
        "pastel":  [[255, 182, 193], [176, 224, 230], [152, 251, 152], [255, 255, 224]],
        "earth":   [[160, 82, 45], [107, 142, 35], [210, 180, 140], [139, 90, 43]],
        "monochrome": [[64, 64, 64], [128, 128, 128], [192, 192, 192], [224, 224, 224]],
        "ocean":   [[0, 119, 190], [64, 224, 208], [0, 191, 255], [25, 25, 112]],
        "sunset":  [[255, 140, 0], [255, 20, 147], [255, 69, 0], [255, 192, 203]],
        "forest":  [[34, 139, 34], [85, 107, 47], [107, 142, 35], [139, 90, 43]],
        "neon":    [[255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 255, 0]],
    }
    return palettes.get(style, palettes["vibrant"])


def apply_colors(
    labeled_regions: np.ndarray,
    coloring: Dict[int, int],
    style: str,
    custom_colors: Optional[List[List[int]]] = None,
) -> np.ndarray:
    """Apply colors to regions based on style or custom colors."""
    palette = _get_palette(style, custom_colors, max_colors=4)

    h, w = labeled_regions.shape
    colored = np.ones((h, w, 3), dtype=np.uint8) * 255

    n_colors = len(palette)
    for region_id, color_id in coloring.items():
        if region_id > 0:
            mask = labeled_regions == region_id
            colored[mask] = palette[color_id % n_colors]

    return colored


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)