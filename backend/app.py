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

from core.photo_to_lineart import convert_photo_to_lineart
from core.photo_processor import (
    process_photo,
    process_coloring_book,
    is_coloring_book,
)
from utils.image_utils import optimize_image_size, validate_image_file, upscale_image

app = FastAPI(title="4-Color Theorem API")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {"message": "4-Color Theorem API", "version": "0.7.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/preview")
@limiter.limit("120/minute")
async def preview_image(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("vibrant"),
    convert_to_lineart: str = Form("false"),
    max_dimension: str = Form("400"),
):
    start_time = time.time()
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        is_valid, error_msg = validate_image_file(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)
        preview_max = min(int(max_dimension), 400)
        image_np, _ = optimize_image_size(image_np, max_dimension=preview_max)

        force_photo = convert_to_lineart.lower() in ("true", "1", "yes", "on")

        result = process_pipeline(
            image_np, style,
            stained_glass_enabled=False,
            force_photo_pipeline=force_photo,
            size_metadata=None,
        )

        processing_time = round((time.time() - start_time) * 1000, 2)
        return JSONResponse(content={
            "success": True,
            "preview": result["image"],
            "preview_size": (image_np.shape[1], image_np.shape[0]),
            "processing_time_ms": processing_time,
        })
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Preview error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


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
    custom_colors: str = Form(None),
):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        is_valid, error_msg = validate_image_file(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)
        image_np, size_metadata = optimize_image_size(image_np, max_dimension=3000)

        stained_glass_enabled = stained_glass.lower() in ("true", "1", "yes", "on")
        force_photo = convert_to_lineart.lower() in ("true", "1", "yes", "on")
        use_five_colors_bool = use_five_colors.lower() in ("true", "1", "yes", "on")

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
            image_np, style,
            stained_glass_enabled=stained_glass_enabled,
            force_photo_pipeline=force_photo,
            size_metadata=size_metadata,
            use_five_colors=use_five_colors_bool,
            custom_colors=custom_colors_list,
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = str(e)
        logger.error(f"Processing error: {error_detail}\n{traceback.format_exc()}")
        if "timeout" in error_detail.lower() or "time" in error_detail.lower():
            error_detail = "Processing took too long. Try a smaller image."
        elif "memory" in error_detail.lower():
            error_detail = "Image too large. Please resize and try again."
        elif "format" in error_detail.lower() or "decode" in error_detail.lower():
            error_detail = "Invalid image format. Please use PNG, JPG, or JPEG."
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/api/preview-lineart")
async def preview_lineart(
    file: UploadFile = File(...),
    line_thickness: str = Form("medium"),
    detail_level: str = Form("detailed"),
    contrast: str = Form("1.0"),
):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)
        contrast_float = float(contrast) if contrast else 1.0
        line_art = convert_photo_to_lineart(
            image_np, line_thickness=line_thickness,
            detail_level=detail_level, contrast=contrast_float,
        )
        result_pil = Image.fromarray(line_art)
        buffered = io.BytesIO()
        result_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return JSONResponse(content={
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# Core pipeline router
# ======================================================================

def process_pipeline(
    image_np: np.ndarray,
    style: str,
    stained_glass_enabled: bool = False,
    force_photo_pipeline: bool = False,
    size_metadata: Optional[Dict] = None,
    use_five_colors: bool = False,
    custom_colors: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """
    Route to the correct pipeline:
    - If force_photo_pipeline is True (user toggled "Convert Photo to Line Art"),
      always use the photo pipeline.
    - Otherwise, auto-detect: coloring book images → coloring book pipeline,
      photos → photo pipeline.
    """
    max_colors = 5 if use_five_colors else 4
    palette = _get_palette(style, custom_colors, max_colors)

    # Decide which pipeline to use
    if force_photo_pipeline:
        use_photo = True
        logger.info("Pipeline: photo (forced by user toggle)")
    else:
        use_photo = not is_coloring_book(image_np)
        logger.info(f"Pipeline: {'photo' if use_photo else 'coloring_book'} (auto-detected)")

    if use_photo:
        colored_image, stats = process_photo(
            image_np, palette=palette, n_clusters=8,
            min_region_area=200, max_colors=max_colors,
        )
        pipeline_name = "photo"
    else:
        colored_image, stats = process_coloring_book(
            image_np, palette=palette,
            min_region_area=50, max_colors=max_colors,
        )
        pipeline_name = "coloring_book"

    # Stained glass (optional)
    if stained_glass_enabled:
        try:
            from effects.stained_glass import apply_stained_glass
            logger.info("Applying stained glass effect")
            colored_image = apply_stained_glass(colored_image, intensity=0.8)
        except Exception as e:
            logger.warning(f"Stained glass failed: {e}")

    # Upscale if needed
    if size_metadata and size_metadata.get("resized", False):
        scale_factor = 1.0 / size_metadata["scale_factor"]
        colored_image = upscale_image(colored_image, scale_factor, method="lanczos")

    # Encode
    result_pil = Image.fromarray(colored_image)
    buffered = io.BytesIO()
    result_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    stats_dict = {
        "regions": stats["regions"],
        "colors_used": stats["colors_used"],
        "max_colors_allowed": max_colors,
        "color_mode": f"{max_colors}-color",
        "graph_nodes": stats["graph_nodes"],
        "graph_edges": stats["graph_edges"],
        "pipeline": pipeline_name,
    }
    if size_metadata:
        stats_dict["image_resized"] = size_metadata.get("resized", False)
        stats_dict["original_size"] = size_metadata.get("original_size")
        stats_dict["processed_size"] = size_metadata.get("final_size")

    return {
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "stats": stats_dict,
    }


def _get_palette(style, custom_colors, max_colors):
    if custom_colors and len(custom_colors) > 0:
        palette = list(custom_colors)
        while len(palette) < max_colors:
            palette.append(palette[-1] if palette else [128, 128, 128])
        return palette
    palettes = {
        "vibrant": [[220, 20, 60], [0, 191, 255], [50, 205, 50], [255, 215, 0]],
        "pastel": [[255, 182, 193], [176, 224, 230], [152, 251, 152], [255, 255, 224]],
        "earth": [[160, 82, 45], [107, 142, 35], [210, 180, 140], [139, 90, 43]],
        "monochrome": [[64, 64, 64], [128, 128, 128], [192, 192, 192], [224, 224, 224]],
        "ocean": [[0, 119, 190], [64, 224, 208], [0, 191, 255], [25, 25, 112]],
        "sunset": [[255, 140, 0], [255, 20, 147], [255, 69, 0], [255, 192, 203]],
        "forest": [[34, 139, 34], [85, 107, 47], [107, 142, 35], [139, 90, 43]],
        "neon": [[255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 255, 0]],
    }
    return palettes.get(style, palettes["vibrant"])


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)