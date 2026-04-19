# Four Color Theorem Web Application

A web application that uses the Four Color Theorem to automatically color images using graph theory and graph coloring algorithms.
https://www.fourcolordemo.com/

## Features

- **Automatic Image Coloring**: Uses graph theory to color images with 4 or 5 colors
- **Line Art Conversion**: Convert photos to line art before coloring
- **Stained Glass Effect**: Optional visual effect for colored images
- **Multiple Color Palettes**: Vibrant, pastel, earth tones, monochrome, ocean, sunset, forest, and neon
- **Image History**: Session-based history of processed images
- **Export Options**: Download images in PNG or JPG with quality/resolution controls

## Prerequisites

- **Python 3.11+** (Python 3.11 or higher recommended)
- **Node.js 16+** and npm (Node Package Manager)
- **Git** (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd four-color-theorem-web
```

### 2. Backend Setup

Navigate to the backend directory and install Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

**Note**: On some systems, you may need to use `pip3` instead of `pip`.

If you encounter issues with OpenCV installation, you may need to install system dependencies:
- **Linux (Ubuntu/Debian)**: `sudo apt-get install python3-opencv`
- **macOS**: `brew install opencv`
- **Windows**: OpenCV should install via pip, but you may need Visual C++ Redistributables

### 3. Frontend Setup

Open a new terminal, navigate to the frontend directory, and install Node.js dependencies:

```bash
cd frontend
npm install
```

This will install all required dependencies including React, Axios, and Transformers.js for browser-based AI.

## Running the Application

The application consists of two parts: the backend server and the frontend web interface. You need to run both simultaneously.

### Start the Backend Server

In the `backend` directory:

```bash
python app.py
```

Or on some systems:

```bash
python3 app.py
```

The backend server will start on `http://localhost:8000`

You should see output like:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Start the Frontend Development Server

Open a **new terminal window** and navigate to the `frontend` directory:

```bash
cd frontend
npm start
```

The frontend will start on `http://localhost:3000` and should automatically open in your browser.

If it doesn't open automatically, navigate to: `http://localhost:3000`

## Using the Application

1. **Upload an Image**: Click "Choose Image" or drag and drop an image file
2. **Select Style**: Choose a color palette style (vibrant, pastel, earth, etc.)
3. **Optional Features**:
   - Enable "Smart Color Suggestions" for AI-powered palette recommendations
   - Enable "ML Segmentation" for better region detection
   - Enable "5-Color Mode" for complex images
   - Enable "Line Art Converter" to convert photos to line art first
   - Enable "Stained Glass Effect" for visual enhancement
4. **Process Image**: Click "Process Image" to generate the colored result
5. **Download**: Use the download button to save your colored image

### Smart Color Suggestions

When enabled, the AI color suggestion system provides:
- **Layer 1 (Instant)**: Server-side OpenCV pattern analysis for immediate suggestions
- **Layer 2 (Optional)**: Browser-based AI using CLIP model (~150MB download, cached after first use)

Click on any suggested palette to automatically process the image with those colors.

## Development

### Backend Development

The backend is built with:
- FastAPI (Python web framework)
- OpenCV (image processing)
- NetworkX (graph algorithms)
- scikit-image (ML segmentation)
- Transformers.js compatible API

### Frontend Development

The frontend is built with:
- React 18
- Axios (HTTP client)
- Transformers.js (browser-based AI)
- CSS3

### Project Structure

```
four-color-theorem-web/
├── backend/              # Python FastAPI backend
│   ├── app.py           # Main application file
│   ├── core/            # Core algorithms (graph coloring, region detection)
│   ├── effects/         # Visual effects (stained glass)
│   ├── utils/           # Utility functions
│   └── requirements.txt # Python dependencies
├── frontend/            # React frontend
│   ├── src/
│   │   ├── App.js       # Main React component
│   │   ├── components/  # React components
│   │   ├── services/    # API services
│   │   └── workers/     # Web Workers (AI processing)
│   └── package.json     # Node.js dependencies
└── README.md            # This file
```

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**
- Stop any other services using port 8000
- Or modify the port in `backend/app.py` (look for `uvicorn.run`)

**Import errors:**
- Ensure all requirements are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.11+)

**OpenCV errors:**
- Reinstall OpenCV: `pip uninstall opencv-python opencv-contrib-python && pip install -r requirements.txt`

### Frontend Issues

**Port 3000 already in use:**
- The terminal will prompt you to use a different port (usually 3001)
- Or stop the process using port 3000

**Module not found errors:**
- Run `npm install` again in the frontend directory
- Delete `node_modules` and `package-lock.json`, then run `npm install`

**CORS errors:**
- Ensure the backend server is running on port 8000
- Check that `frontend/package.json` has the correct proxy setting

### General Issues

**Images not processing:**
- Check that both backend and frontend servers are running
- Check browser console for errors (F12 → Console tab)
- Check backend terminal for error messages
- Ensure image format is supported (PNG, JPG, JPEG)

**Rate limiting errors:**
- The server has rate limits to prevent abuse
- Current limits: 60 requests/minute for processing, 120/minute for analysis
- Wait a moment and try again

## API Endpoints

- `GET /` - API status
- `POST /api/process` - Process image with coloring algorithm
- `POST /api/analyze-colors` - Analyze image for color suggestions (OpenCV)
- `GET /api/palettes` - Get available color palettes
- `POST /api/preview` - Generate low-resolution preview

## License

See LICENSE file for details.

## Contributing

Please refer to PROJECT_ROADMAP.md for current development status and planned features.
