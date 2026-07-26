"""FastAPI backend for 4-color theorem app."""
import asyncio
import os
import io
import base64
import time
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.photo_to_lineart import convert_photo_to_lineart
from core.photo_processor import (
    process_photo,
    process_coloring_book,
    is_coloring_book,
)
from core.recolor_cache import recolor_cache, render_from_cache
from core.animation import build_animation_payload
from utils.image_utils import optimize_image_size, validate_image_file, upscale_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Thread pool for CPU-bound processing (SSE streaming)
_executor = ThreadPoolExecutor(max_workers=4)


# ======================================================================
# Health / root
# ======================================================================

@app.get("/")
async def root():
    return {"message": "4-Color Theorem API", "version": "0.8.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ======================================================================
# Preview (unchanged)
# ======================================================================

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


# ======================================================================
# Process (original synchronous endpoint, now returns session_id)
# ======================================================================

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

        custom_colors_list = _parse_custom_colors(custom_colors)

        result = process_pipeline(
            image_np, style,
            stained_glass_enabled=stained_glass_enabled,
            force_photo_pipeline=force_photo,
            size_metadata=size_metadata,
            use_five_colors=use_five_colors_bool,
            custom_colors=custom_colors_list,
            line_thickness=line_thickness,
            detail_level=detail_level,
            contrast=float(contrast) if contrast else 1.0,
            include_animation=True,
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


# ======================================================================
# Process with streaming progress (SSE)
# ======================================================================

@app.post("/api/process-stream")
@limiter.limit("60/minute")
async def process_image_stream(
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
    """
    Same processing as /api/process, but returns Server-Sent Events
    with progress updates followed by the final result.

    Events:
      data: {"type":"progress","stage":"Detecting edges","progress":35}
      data: {"type":"result","data":{...same as /api/process response...}}
      data: {"type":"error","message":"..."}
    """
    # Parse all inputs synchronously before streaming
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
        custom_colors_list = _parse_custom_colors(custom_colors)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Set up the progress bridge: worker thread -> async queue
    progress_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def progress_cb(stage: str, pct: int):
        loop.call_soon_threadsafe(
            progress_queue.put_nowait,
            {"type": "progress", "stage": stage, "progress": pct},
        )

    def do_work():
        return process_pipeline(
            image_np, style,
            stained_glass_enabled=stained_glass_enabled,
            force_photo_pipeline=force_photo,
            size_metadata=size_metadata,
            use_five_colors=use_five_colors_bool,
            custom_colors=custom_colors_list,
            line_thickness=line_thickness,
            detail_level=detail_level,
            contrast=float(contrast) if contrast else 1.0,
            include_animation=True,
            progress_cb=progress_cb,
        )

    async def event_stream():
        future = loop.run_in_executor(_executor, do_work)

        # Relay progress until processing finishes
        while not future.done():
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.4)
                yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

        # Drain any remaining progress messages
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            yield f"data: {json.dumps(msg)}\n\n"

        # Emit the final result (or error)
        try:
            result = future.result()
            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
        except Exception as exc:
            logger.error(f"Stream processing error: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # prevent proxy buffering
        },
    )


# ======================================================================
# Recolor (fast palette swap using cached data)
# ======================================================================

@app.post("/api/recolor")
@limiter.limit("120/minute")
async def recolor_image(
    request: Request,
    session_id: str = Form(...),
    style: str = Form("vibrant"),
    custom_colors: str = Form(None),
    use_five_colors: str = Form("false"),
):
    """
    Instantly re-render a previously processed image with a new palette.
    Skips edge detection, region finding, and graph coloring entirely.
    """
    cached = recolor_cache.get(session_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="Session expired or not found. Please re-process the image.",
        )

    use_five = use_five_colors.lower() in ("true", "1", "yes", "on")
    max_colors = 5 if use_five else 4
    custom_colors_list = _parse_custom_colors(custom_colors)
    palette = _get_palette(style, custom_colors_list, max_colors)

    start = time.time()
    colored_image = render_from_cache(
        cached["filtered"],
        cached["balanced"],
        cached["outline_mask"],
        palette,
        line_alpha=cached.get("line_alpha"),
    )

    # Stained glass (reapply if it was on)
    if cached.get("stained_glass"):
        try:
            from effects.stained_glass_v2 import apply_stained_glass
            colored_image = apply_stained_glass(
                colored_image, intensity=0.8,
                line_alpha=cached.get("line_alpha"))
        except Exception as e:
            logger.warning(f"Stained glass failed during recolor: {e}")

    # Upscale if the original was resized
    size_metadata = cached.get("size_metadata")
    if size_metadata and size_metadata.get("resized", False):
        scale_factor = 1.0 / size_metadata["scale_factor"]
        colored_image = upscale_image(colored_image, scale_factor, method="lanczos")

    # Encode
    result_pil = Image.fromarray(colored_image)
    buffered = io.BytesIO()
    result_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    elapsed_ms = round((time.time() - start) * 1000, 2)
    logger.info(f"Recolor completed in {elapsed_ms}ms")

    region_colors = {
        str(r): palette[c % len(palette)]
        for r, c in cached["balanced"].items()
    }

    return JSONResponse(content={
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "stats": cached["stats"],
        "session_id": session_id,
        "recolor_time_ms": elapsed_ms,
        "region_colors": region_colors,
    })


# ======================================================================
# Preview line art (unchanged)
# ======================================================================

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
    line_thickness: str = "medium",
    detail_level: str = "detailed",
    contrast: float = 1.0,
    include_animation: bool = False,
    progress_cb: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Route to the correct pipeline:

    - "Convert Photo to Line Art" toggled on (force_photo_pipeline):
        photo -> line art -> coloring book pipeline.
        The intermediate line art is returned as `lineart`.
    - Otherwise auto-detect:
        coloring book images -> coloring book pipeline
        photos               -> photo (posterize) pipeline

    Returns a dict with image, stats, session_id, optional lineart, and an
    optional `animation` payload for the real-time coloring window.
    """
    max_colors = 5 if use_five_colors else 4
    palette = _get_palette(style, custom_colors, max_colors)

    lineart_b64 = None

    if force_photo_pipeline:
        # FIXED: this used to route to the k-means photo pipeline and
        # never actually produced line art.
        logger.info("Pipeline: photo -> line art -> coloring book")
        if progress_cb:
            progress_cb("Converting photo to line art", 5)
        lineart = convert_photo_to_lineart(
            image_np,
            line_thickness=line_thickness,
            detail_level=detail_level,
            contrast=contrast,
        )
        lineart_b64 = _encode_png(lineart)
        colored_image, stats, recolor_data = process_coloring_book(
            lineart, palette=palette,
            min_region_area=50, max_colors=max_colors,
            progress_cb=progress_cb,
        )
        pipeline_name = "photo_to_lineart"
    elif not is_coloring_book(image_np):
        logger.info("Pipeline: photo (auto-detected)")
        colored_image, stats, recolor_data = process_photo(
            image_np, palette=palette, n_clusters=8,
            min_region_area=200, max_colors=max_colors,
            progress_cb=progress_cb,
        )
        pipeline_name = "photo"
    else:
        logger.info("Pipeline: coloring_book (auto-detected)")
        colored_image, stats, recolor_data = process_coloring_book(
            image_np, palette=palette,
            min_region_area=50, max_colors=max_colors,
            progress_cb=progress_cb,
        )
        pipeline_name = "coloring_book"

    if progress_cb:
        progress_cb("Applying effects", 88)

    # Stained glass (optional, server-side; vectorized v2)
    if stained_glass_enabled:
        try:
            from effects.stained_glass_v2 import apply_stained_glass
            logger.info("Applying stained glass effect (v2)")
            colored_image = apply_stained_glass(
                colored_image, intensity=0.8,
                line_alpha=recolor_data.get("line_alpha"),
            )
        except Exception as e:
            logger.warning(f"Stained glass failed: {e}")

    # Animation payload (before upscaling - animation runs at processed res)
    animation = None
    if include_animation:
        try:
            if progress_cb:
                progress_cb("Preparing animation data", 92)
            animation = build_animation_payload(
                recolor_data["filtered"],
                recolor_data["balanced"],
                palette,
                recolor_data.get("line_alpha"),
            )
        except Exception as e:
            logger.warning(f"Animation payload failed: {e}")

    # Upscale if needed
    if size_metadata and size_metadata.get("resized", False):
        scale_factor = 1.0 / size_metadata["scale_factor"]
        colored_image = upscale_image(colored_image, scale_factor, method="lanczos")

    if progress_cb:
        progress_cb("Encoding image", 96)

    img_base64 = _encode_png(colored_image)

    # Cache recolor data
    recolor_data["stained_glass"] = stained_glass_enabled
    recolor_data["size_metadata"] = size_metadata
    recolor_data["stats"] = stats
    session_id = recolor_cache.put(recolor_data)

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

    if progress_cb:
        progress_cb("Complete", 100)

    result = {
        "success": True,
        "image": img_base64,
        "stats": stats_dict,
        "session_id": session_id,
    }
    if lineart_b64:
        result["lineart"] = lineart_b64
    if animation:
        result["animation"] = animation
    return result


def _encode_png(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr)
    buffered = io.BytesIO()
    pil.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()


# ======================================================================
# Helpers
# ======================================================================

def _parse_custom_colors(raw: Optional[str]) -> Optional[List[List[int]]]:
    """Parse a JSON string of [[r,g,b], ...] into a list, or return None."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            for color in parsed:
                if not (isinstance(color, list) and len(color) == 3):
                    return None
            return parsed
    except Exception:
        pass
    return None


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