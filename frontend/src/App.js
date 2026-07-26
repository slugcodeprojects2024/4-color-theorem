import React, { useState, useEffect, useCallback } from 'react';
import ImageUploader from './components/ImageUploader';
import ProcessButton from './components/ProcessButton';
import ProgressIndicator from './components/ProgressIndicator';
import ResultExperience from './components/ResultExperience';
import StyleSelector from './components/StyleSelector';
import StainedGlassToggle from './components/StainedGlassToggle';
import LineArtConverter from './components/LineArtConverter';
import CustomColorPicker from './components/CustomColorPicker';
import ImageHistory from './components/ImageHistory';
import EducationalPage from './components/EducationalPage';
import GalleryPage from './components/GalleryPage';
import {
  processImageWithProgress,
  recolorImage,
  checkServerStatus,
} from './services/api';
import { saveImageToHistory } from './components/ImageHistory';

function App() {
  // --- Image state ---
  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [stats, setStats] = useState(null);
  const [animation, setAnimation] = useState(null);
  const [lineartImage, setLineartImage] = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [error, setError] = useState(null);

  // --- Processing state ---
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressStage, setProgressStage] = useState(null);
  const [progressPercent, setProgressPercent] = useState(null);
  const [isRecoloring, setIsRecoloring] = useState(false);

  // --- Session (for recolor) ---
  const [sessionId, setSessionId] = useState(null);

  // --- Settings ---
  const [selectedStyle, setSelectedStyle] = useState('vibrant');
  const [stainedGlassEnabled, setStainedGlassEnabled] = useState(false);
  const [lineArtEnabled, setLineArtEnabled] = useState(false);
  const [lineArtSettings, setLineArtSettings] = useState({
    lineThickness: 'medium',
    detailLevel: 'detailed',
    contrast: 1.0,
  });
  const [customColorsEnabled, setCustomColorsEnabled] = useState(false);
  const [customColors, setCustomColors] = useState([
    [220, 20, 60],
    [0, 191, 255],
    [50, 205, 50],
    [255, 215, 0],
  ]);

  // --- Pages ---
  const [serverStatus, setServerStatus] = useState(null);
  const [showEducationalPage, setShowEducationalPage] = useState(false);
  const [showGalleryPage, setShowGalleryPage] = useState(false);

  // Check server on mount
  useEffect(() => {
    checkServerStatus()
      .then(() => setServerStatus('connected'))
      .catch(() => setServerStatus('disconnected'));
  }, []);

  // ------------------------------------------------------------------
  // Handlers
  // ------------------------------------------------------------------

  const handleImageSelect = (imageFile) => {
    setSelectedImage(imageFile);
    setProcessedImage(null);
    setStats(null);
    setAnimation(null);
    setLineartImage(null);
    setOriginalUrl(imageFile ? URL.createObjectURL(imageFile) : null);
    setError(null);
    setSessionId(null);
    setProgressStage(null);
    setProgressPercent(null);
  };

  const getCurrentSettings = useCallback(
    () => ({
      style: selectedStyle,
      stainedGlass: stainedGlassEnabled,
      lineArt: lineArtEnabled ? lineArtSettings : { enabled: false },
      customColors: customColorsEnabled ? customColors : null,
    }),
    [selectedStyle, stainedGlassEnabled, lineArtEnabled, lineArtSettings, customColorsEnabled, customColors]
  );

  // Full pipeline processing (with streaming progress)
  const handleProcess = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    try {
      await checkServerStatus();
      setServerStatus('connected');
    } catch (err) {
      setServerStatus('disconnected');
      setError(
        'Server is not available. Please ensure the backend is running on port 8000.'
      );
      return;
    }

    setIsProcessing(true);
    setError(null);
    setProgressStage('Starting…');
    setProgressPercent(0);

    try {
      const result = await processImageWithProgress(
        selectedImage,
        {
          style: selectedStyle,
          stainedGlassEnabled: false, // stained glass handled on frontend
          lineArtSettings: lineArtEnabled
            ? { enabled: true, ...lineArtSettings }
            : null,
          customColors: customColorsEnabled ? customColors : null,
        },
        // Progress callback
        (stage, pct) => {
          setProgressStage(stage);
          setProgressPercent(pct);
        }
      );

      if (!result || !result.image) {
        throw new Error('Server returned invalid response - no image data');
      }

      setProcessedImage(result.image);
      setStats(result.stats);
      setSessionId(result.session_id || null);
      setAnimation(result.animation || null);
      setLineartImage(result.lineart || null);

      try {
        saveImageToHistory(result.image, getCurrentSettings(), result.stats);
      } catch (err) {
        console.warn('Failed to save to history:', err);
      }
    } catch (err) {
      let errorMessage = err.message || 'Failed to process image';

      if (errorMessage.includes('too large') || errorMessage.includes('size')) {
        errorMessage =
          'Image is too large. Please use an image smaller than 50MB or 10000×10000px.';
      } else if (
        errorMessage.includes('format') ||
        errorMessage.includes('Invalid')
      ) {
        errorMessage = 'Invalid image format. Please use PNG, JPG, or JPEG.';
      } else if (
        errorMessage.includes('timeout') ||
        errorMessage.includes('long')
      ) {
        errorMessage =
          'Processing took too long. Try a smaller image or disable some effects.';
      } else if (
        errorMessage.includes('Network') ||
        errorMessage.includes('Cannot connect')
      ) {
        errorMessage =
          'Cannot connect to server. Please ensure the backend is running.';
      }

      setError(errorMessage);
      console.error('Processing error:', err);
    } finally {
      setIsProcessing(false);
      setProgressStage(null);
      setProgressPercent(null);
    }
  };

  // Fast recolor using cached pipeline data
  const handleRecolor = async () => {
    if (!sessionId) return;

    setIsRecoloring(true);
    setError(null);

    try {
      const result = await recolorImage(
        sessionId,
        selectedStyle,
        customColorsEnabled ? customColors : null
      );

      setProcessedImage(result.image);
      setStats(result.stats);
      // session_id stays the same

      if (result.region_colors) {
        setAnimation((prev) =>
          prev ? { ...prev, region_colors: result.region_colors } : prev
        );
      }

      try {
        saveImageToHistory(result.image, getCurrentSettings(), result.stats);
      } catch (err) {
        console.warn('Failed to save to history:', err);
      }
    } catch (err) {
      if (err.message === 'SESSION_EXPIRED') {
        setSessionId(null);
        setError(
          'Recolor session expired. Click "Color Image" to re-process.'
        );
      } else {
        setError(err.message || 'Recolor failed');
      }
      console.error('Recolor error:', err);
    } finally {
      setIsRecoloring(false);
    }
  };

  const handleStyleChange = (style) => {
    setSelectedStyle(style);
  };

  const handleStainedGlassToggle = (enabled) => {
    setStainedGlassEnabled(enabled);
  };

  const handleLineArtToggle = (enabled) => {
    setLineArtEnabled(enabled);
  };

  const handleLineArtSettingsChange = (newSettings) => {
    setLineArtSettings(newSettings);
  };

  const handleCustomColorsToggle = (enabled) => {
    setCustomColorsEnabled(enabled);
  };

  const handleCustomColorsChange = (colors) => {
    setCustomColors(colors);
  };

  const handleSelectHistoryItem = (item) => {
    setProcessedImage(item.image);
    setStats(item.stats);
    setAnimation(null);
    setLineartImage(null);
    setSessionId(null); // history items don't have a recolor session
    if (item.settings) {
      setSelectedStyle(item.settings.style || 'vibrant');
      setStainedGlassEnabled(item.settings.stainedGlass || false);
      if (item.settings.lineArt?.enabled) {
        setLineArtEnabled(true);
        setLineArtSettings({
          lineThickness: item.settings.lineArt.lineThickness || 'medium',
          detailLevel: item.settings.lineArt.detailLevel || 'detailed',
          contrast: item.settings.lineArt.contrast || 1.0,
        });
      } else {
        setLineArtEnabled(false);
      }
      if (item.settings.customColors) {
        setCustomColorsEnabled(true);
        setCustomColors(item.settings.customColors);
      } else {
        setCustomColorsEnabled(false);
      }
    }
  };

  // ------------------------------------------------------------------
  // Pages
  // ------------------------------------------------------------------

  if (showEducationalPage) {
    return <EducationalPage onBack={() => setShowEducationalPage(false)} />;
  }
  if (showGalleryPage) {
    return <GalleryPage onBack={() => setShowGalleryPage(false)} />;
  }

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  return (
    <div className="App">
      <header className="App-header">
        <h1>Four Color Theorem</h1>
        <p className="subtitle">
          Automatic Image Coloring using Graph Theory
        </p>
        <div
          style={{
            display: 'flex',
            gap: '10px',
            marginTop: '15px',
            flexWrap: 'wrap',
            justifyContent: 'center',
          }}
        >
          <button
            className="educational-link"
            onClick={() => setShowEducationalPage(true)}
            style={{
              padding: '10px 20px',
              background: 'rgba(255, 255, 255, 0.2)',
              border: '2px solid white',
              borderRadius: '6px',
              color: 'white',
              fontSize: '14px',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
            }}
          >
            Learn About the Theorem
          </button>
          <button
            className="gallery-link"
            onClick={() => setShowGalleryPage(true)}
            style={{
              padding: '10px 20px',
              background: 'rgba(255, 255, 255, 0.2)',
              border: '2px solid white',
              borderRadius: '6px',
              color: 'white',
              fontSize: '14px',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
            }}
          >
            View Example Gallery
          </button>
        </div>
      </header>

      <main className="App-main">
        {serverStatus === 'disconnected' && (
          <div
            className="server-warning"
            style={{
              background: '#fff3cd',
              border: '1px solid #ffc107',
              borderRadius: '8px',
              padding: '12px',
              margin: '20px 0',
              color: '#856404',
            }}
          >
            <strong>Server Not Connected</strong>
            <p style={{ margin: '8px 0 0 0', fontSize: '0.9rem' }}>
              Backend server is not running. Please start it with:{' '}
              <code
                style={{
                  background: '#f0f0f0',
                  padding: '2px 6px',
                  borderRadius: '3px',
                }}
              >
                python app.py
              </code>{' '}
              in the backend folder.
            </p>
          </div>
        )}

        <div className="upload-section">
          <ImageUploader onImageSelect={handleImageSelect} />
          {selectedImage && (
            <div className="preview-section">
              <h3>Original Image</h3>
              <img
                src={originalUrl || URL.createObjectURL(selectedImage)}
                alt="Selected"
                className="preview-image"
              />
            </div>
          )}
        </div>

        {selectedImage && (
          <div className="controls-section">
            <StyleSelector
              selectedStyle={selectedStyle}
              onStyleChange={handleStyleChange}
            />

            <CustomColorPicker
              enabled={customColorsEnabled}
              onToggle={handleCustomColorsToggle}
              colors={customColors}
              onColorsChange={handleCustomColorsChange}
            />

            <LineArtConverter
              enabled={lineArtEnabled}
              onToggle={handleLineArtToggle}
              settings={lineArtSettings}
              onSettingsChange={handleLineArtSettingsChange}
            />

            <StainedGlassToggle
              enabled={stainedGlassEnabled}
              onToggle={handleStainedGlassToggle}
            />

            <ProcessButton
              onProcess={handleProcess}
              disabled={isProcessing || isRecoloring}
            />

            {/* Recolor button: shown when there's a cached session */}
            {sessionId && processedImage && (
              <button
                className="process-button"
                onClick={handleRecolor}
                disabled={isProcessing || isRecoloring}
                style={{
                  marginTop: '10px',
                  background: isRecoloring
                    ? '#ccc'
                    : 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
                  color: isRecoloring ? '#666' : '#1a1a2e',
                }}
              >
                {isRecoloring ? 'Recoloring…' : 'Recolor (Instant Palette Swap)'}
              </button>
            )}
          </div>
        )}

        <ImageHistory onSelectHistoryItem={handleSelectHistoryItem} />

        {isProcessing && (
          <ProgressIndicator
            progress={progressPercent}
            stage={progressStage}
          />
        )}

        {error && (
          <div className="error-message">
            <p>Error: {error}</p>
          </div>
        )}

        {processedImage && (
          <ResultExperience
            image={processedImage}
            stats={stats}
            animation={animation}
            lineart={lineartImage}
            originalUrl={originalUrl}
            defaultTab={
              stainedGlassEnabled ? 'glass' : animation ? 'animation' : 'result'
            }
          />
        )}
      </main>

      <footer className="App-footer">
        <p>
          Upload a coloring book style image to automatically color it with 4
          colors
        </p>
      </footer>
    </div>
  );
}

export default App;