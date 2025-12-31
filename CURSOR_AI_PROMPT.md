# Four Color Theorem Web Application - Cursor AI Prompt

## Project Overview

You are continuing development on a **Four Color Theorem web application** that automatically colors images (coloring book pages, photos converted to line art) using graph theory. The app detects regions in images, builds an adjacency graph, and applies the 4-color theorem to ensure no adjacent regions share the same color.

**Tech Stack:**
- **Frontend:** React 18, CSS (no Tailwind), Axios
- **Backend:** FastAPI, Python 3.11+, OpenCV, NetworkX, NumPy, Pillow
- **Effects:** WebGL-based stained glass effect (frontend), photo-to-lineart conversion (backend)

**Current Status:** Phase 1 complete - Core features working, includes color palettes, image optimization, error handling, and export options.

---

## Project Structure

```
four-color-theorem-web/
├── backend/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── four_color_solver.py    # Graph coloring algorithms
│   │   ├── graph_builder.py        # Build NetworkX graph from regions
│   │   ├── region_detection.py     # Detect closed regions in images
│   │   └── photo_to_lineart.py     # Convert photos to coloring book style
│   ├── effects/
│   │   ├── __init__.py
│   │   ├── stained_glass.py        # Backend stained glass effect
│   │   ├── glass_textures.py       # (empty - for future)
│   │   └── lighting_effects.py     # (empty - for future)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py          # Image optimization utilities
│   │   ├── cache.py                # (empty - for future caching)
│   │   └── color_utils.py          # (empty - for future)
│   ├── app.py                      # FastAPI application
│   └── requirements.txt
├── frontend/
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── components/
│       │   ├── ImageUploader.js
│       │   ├── ProcessButton.js
│       │   ├── ProgressIndicator.js
│       │   ├── ResultViewer.js
│       │   ├── StyleSelector.js
│       │   ├── StainedGlassToggle.js
│       │   ├── LineArtConverter.js + .css
│       │   └── ExportOptions.js + .css
│       ├── effects/
│       │   └── stainedGlassEffect.js        # WebGL stained glass
│       ├── services/
│       │   └── api.js                       # API client
│       ├── styles/
│       │   └── App.css
│       ├── App.js
│       └── index.js
└── PROJECT_ROADMAP.md
```

---

## Core Algorithm Flow

```
1. Image Upload
      ↓
2. [Optional] Photo → Line Art Conversion
      ↓
3. Region Detection (edge detection + connected components)
      ↓
4. Build Adjacency Graph (which regions touch each other)
      ↓
5. Graph Coloring (Welsh-Powell → Backtracking → NetworkX fallback)
      ↓
6. Apply Colors from Selected Palette
      ↓
7. [Optional] Apply Stained Glass Effect
      ↓
8. Upscale to Original Size (if resized)
      ↓
9. Return Result
```

---

## Key Files - Implementation Details

### Backend: `backend/core/four_color_solver.py`

Implements the graph coloring logic with three strategies:
1. **Welsh-Powell**: Fast greedy algorithm, sorts nodes by degree
2. **Backtracking**: Exact 4-coloring for small graphs (< 100 nodes)
3. **NetworkX DSATUR**: Fallback for large graphs or when others fail

Key features:
- Normalizes colors to 0-3 range
- Skips backtracking for graphs > 100 nodes (performance safeguard)
- Max iteration limit (10,000) to prevent hanging

### Backend: `backend/core/region_detection.py`

Detects closed regions using:
- Canny edge detection + threshold-based edge detection
- Morphological operations to close gaps
- Connected components analysis
- Contour-based adjacency detection with search radius

Key features:
- Minimum region area filter (default 100 pixels)
- Handles thin separating lines between regions
- Returns labeled regions array and contour list

### Backend: `backend/app.py`

Main FastAPI application with:
- `/api/process` - Full resolution processing
- `/api/preview-lineart` - Preview line art conversion
- Image size optimization (auto-resize to 2000px max, upscale result)
- Error handling with helpful messages
- 8 color palettes: vibrant, pastel, earth, monochrome, ocean, sunset, forest, neon

### Frontend: `frontend/src/App.js`

Main React component managing:
- Image upload and selection
- Style/palette selection
- Line art conversion settings
- Stained glass toggle
- Processing state and progress
- Result display

### Frontend: `frontend/src/effects/stainedGlassEffect.js`

WebGL-based stained glass effect:
- Sobel edge detection for lead lines
- Glass texture with noise
- Lighting gradients and vignette
- GPU-accelerated processing

---

## Phase 1 Features (Completed)

1. **Additional Color Palettes** - Ocean, Sunset, Forest, Neon palettes added
2. **Image Size Optimization** - Auto-resize large images (2000px max), upscale results
3. **Enhanced Error Handling** - Client-side validation, detailed error messages
4. **Basic Export Options** - PNG/JPG download with quality and resolution controls
5. **Progress Indicator** - Percentage-based progress display

---

## Phase 2 Roadmap (Next Steps)

### 2.1 Processing Presets
- Save/load setting combinations (style + line art + stained glass)
- localStorage persistence
- Default presets included
- Preset gallery UI

### 2.2 Image History
- Session-based history of last 10 processed images
- Quick reload previous results
- History panel UI

### 2.3 Quick Preview
- Low-res preview endpoint (400px max)
- Fast preview before full processing
- Preview button in UI

### 2.4 API Hardening
- Rate limiting (30 preview/min, 10 process/min)
- Request validation
- Enhanced error handling
- Security headers

---

## Phase 3 Roadmap (Future)

### 3.1 ML-Enhanced Region Detection
- Integrate SAM (Segment Anything Model) or U-Net
- Better handling of photos vs line art
- Automatic region merging for over-segmentation

### 3.2 ML-Based Color Palette Suggestions
- Analyze image content to suggest appropriate palettes
- Custom palette generation from image colors

### 3.3 Smart Image Preprocessing
- Auto contrast/brightness normalization
- Background removal
- Image quality assessment

---

## Known Issues / Technical Debt

1. **Region Detection Quality** - Current edge-based detection creates many small regions on textured images. Need smarter region merging or ML-based segmentation.

2. **Color Palette Naming** - Some palettes may not match their names exactly (e.g., "ocean" palette). May need palette renaming or content-aware selection.

3. **Stained Glass Performance** - WebGL effect runs on main thread. Could be moved to Web Worker for better UX.

4. **Missing Tests** - No unit tests for core algorithms yet.

5. **No Rate Limiting** - Backend doesn't have rate limiting implemented yet.

6. **No Presets/History** - Phase 2 features not yet implemented.

---

## How to Run

```bash
# Backend
cd backend
pip install -r requirements.txt
python app.py
# Or: uvicorn app:app --reload --port 8000

# Frontend
cd frontend
npm install
npm start
```

Backend runs on `http://localhost:8000`  
Frontend runs on `http://localhost:3000`

---

## Development Guidelines

When making changes:

1. **Maintain Code Style**
   - React functional components with hooks
   - Python type hints
   - Clear function docstrings

2. **Architecture**
   - Keep WebGL stained glass effect on frontend (GPU-accelerated)
   - Preserve modular architecture (core/, effects/, utils/)
   - Backend handles heavy computation, frontend handles UI/effects

3. **Error Handling**
   - Add appropriate error handling and logging
   - Provide helpful error messages to users
   - Validate inputs on both client and server

4. **Performance**
   - Use image size optimization for large images
   - Skip expensive algorithms (backtracking) for large graphs
   - Consider Web Workers for heavy frontend processing

5. **Testing**
   - Test with various image types (line art, photos, complex images)
   - Verify color constraints (no adjacent regions share color)
   - Check edge cases (very large images, many regions)

---

## Your Task

Continue development on this project. Prioritize based on:

1. **Phase 2 features** - Implement presets, image history, quick preview, and API hardening
2. **User-facing improvements** - Enhance the artistic output and user experience
3. **Bug fixes and performance** - Address known issues and optimize processing
4. **Phase 3 features** - ML enhancements if core functionality is solid

When implementing new features:
- Follow the existing code patterns
- Update the roadmap when features are completed
- Add appropriate error handling
- Test thoroughly before marking as complete

---

## Quick Reference

### Color Palettes Available
- `vibrant` - Bright, bold colors (Crimson, Deep Sky Blue, Lime Green, Gold)
- `pastel` - Soft, gentle colors (Light Pink, Powder Blue, Pale Green, Light Yellow)
- `earth` - Natural, earthy tones (Sienna, Olive Drab, Tan, Saddle Brown)
- `monochrome` - Grayscale shades
- `ocean` - Blues and aquas (Ocean Blue, Turquoise, Deep Sky Blue, Midnight Blue)
- `sunset` - Warm oranges and pinks (Dark Orange, Deep Pink, Red Orange, Pink)
- `forest` - Greens and browns (Forest Green, Dark Olive Green, Olive Drab, Saddle Brown)
- `neon` - Bright fluorescent colors (Magenta, Cyan, Yellow, Lime)

### API Endpoints
- `POST /api/process` - Full image processing
  - Parameters: `file`, `style`, `stained_glass`, `convert_to_lineart`, `line_thickness`, `detail_level`, `contrast`
- `POST /api/preview-lineart` - Preview line art conversion
  - Parameters: `file`, `line_thickness`, `detail_level`, `contrast`

### Key Constants
- Max image dimension: 2000px (processing), 4000px (upload)
- Max file size: 10MB
- Min region area: 100 pixels
- Max backtracking iterations: 10,000
- Large graph threshold: 100 nodes

---

This document should be updated as the project evolves. Keep it accurate to the current state of the codebase.

