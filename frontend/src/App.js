import React, { useState, useEffect } from 'react';
import ImageUploader from './components/ImageUploader';
import ProcessButton from './components/ProcessButton';
import ProgressIndicator from './components/ProgressIndicator';
import ResultViewer from './components/ResultViewer';
import StyleSelector from './components/StyleSelector';
import StainedGlassToggle from './components/StainedGlassToggle';
import LineArtConverter from './components/LineArtConverter';
import ImageHistory from './components/ImageHistory';
import SegmentationSettings from './components/SegmentationSettings';
import FiveColorToggle from './components/FiveColorToggle';
import SmartColorSuggester from './components/SmartColorSuggester';
import EducationalPage from './components/EducationalPage';
import GalleryPage from './components/GalleryPage';
import { processImage, checkServerStatus } from './services/api';
import { applyStainedGlassEffect } from './effects/stainedGlassEffect';
import { saveImageToHistory } from './components/ImageHistory';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState('vibrant');
  const [stainedGlassEnabled, setStainedGlassEnabled] = useState(false);
  const [lineArtEnabled, setLineArtEnabled] = useState(false);
  const [lineArtSettings, setLineArtSettings] = useState({
    lineThickness: 'medium',
    detailLevel: 'detailed',
    contrast: 1.0
  });
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [serverStatus, setServerStatus] = useState(null);
  const [useMLSegmentation, setUseMLSegmentation] = useState(false);
  const [segmentationMethod, setSegmentationMethod] = useState('auto');
  const [targetRegions, setTargetRegions] = useState(50);
  const [useFiveColors, setUseFiveColors] = useState(false);
  const [selectedPalette, setSelectedPalette] = useState(null);
  const [enableSmartColorSuggestions, setEnableSmartColorSuggestions] = useState(false);
  const [showEducationalPage, setShowEducationalPage] = useState(false);
  const [showGalleryPage, setShowGalleryPage] = useState(false);

  // Check server status on mount
  useEffect(() => {
    checkServerStatus()
      .then(() => setServerStatus('connected'))
      .catch(() => setServerStatus('disconnected'));
  }, []);

  const handleImageSelect = (imageFile) => {
    setSelectedImage(imageFile);
    setProcessedImage(null);
    setStats(null);
    setError(null);
  };

  const handleProcessWithColors = async (colors) => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    // Check server status before processing
    try {
      await checkServerStatus();
      setServerStatus('connected');
    } catch (err) {
      setServerStatus('disconnected');
      setError('Server is not available. Please ensure the backend is running on port 8000.');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Prepare line art settings if enabled
      const lineArtConfig = lineArtEnabled ? {
        enabled: true,
        ...lineArtSettings
      } : null;
      
      // Prepare ML segmentation settings
      const mlConfig = useMLSegmentation ? {
        enabled: true,
        method: segmentationMethod,
        targetRegions: targetRegions
      } : null;
      
      // Process image with custom colors
      const result = await processImage(
        selectedImage, 
        selectedStyle, 
        false, // Stained glass handled on frontend
        lineArtConfig,
        mlConfig,
        useFiveColors,
        colors // Pass custom colors
      );
      
      console.log('Processing result received:', {
        success: result.success,
        hasImage: !!result.image,
        imageLength: result.image?.length,
        hasStats: !!result.stats,
        stats: result.stats
      });
      
      if (!result || !result.image) {
        throw new Error('Server returned invalid response - no image data received');
      }
      
      let finalImage = result.image;
      
      // Apply WebGL stained glass effect on frontend if enabled
      if (stainedGlassEnabled) {
        try {
          console.log('Applying stained glass effect (intensity: 1.0)...');
          finalImage = await applyStainedGlassEffect(result.image, 1.0);
          console.log('Stained glass effect applied successfully');
        } catch (stainedGlassError) {
          console.warn('Stained glass effect failed, using original:', stainedGlassError);
          finalImage = result.image;
        }
      }
      
      setProcessedImage(finalImage);
      setStats(result.stats);
      
      // Save to history
      try {
        saveImageToHistory(finalImage, getCurrentSettings(), result.stats);
      } catch (err) {
        console.warn('Failed to save to history:', err);
      }
    } catch (err) {
      let errorMessage = err.message || 'Failed to process image';
      
      if (errorMessage.includes('too large') || errorMessage.includes('size')) {
        errorMessage = 'Image is too large. Please use an image smaller than 50MB or 10000x10000px.';
      } else if (errorMessage.includes('format') || errorMessage.includes('Invalid')) {
        errorMessage = 'Invalid image format. Please use PNG, JPG, or JPEG.';
      } else if (errorMessage.includes('timeout') || errorMessage.includes('long')) {
        errorMessage = 'Processing took too long. Try a smaller image or disable some effects.';
      } else if (errorMessage.includes('Network') || errorMessage.includes('Cannot connect')) {
        errorMessage = 'Cannot connect to server. Please ensure the backend is running on port 8000.';
      }
      
      setError(errorMessage);
      console.error('Processing error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProcess = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    // Check server status before processing
    try {
      await checkServerStatus();
      setServerStatus('connected');
    } catch (err) {
      setServerStatus('disconnected');
      setError('Server is not available. Please ensure the backend is running on port 8000.');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Prepare line art settings if enabled
      const lineArtConfig = lineArtEnabled ? {
        enabled: true,
        ...lineArtSettings
      } : null;
      
      // Prepare ML segmentation settings
      const mlConfig = useMLSegmentation ? {
        enabled: true,
        method: segmentationMethod,
        targetRegions: targetRegions
      } : null;
      
      // Process image with all settings
      const result = await processImage(
        selectedImage, 
        selectedStyle, 
        false, // Stained glass handled on frontend
        lineArtConfig,
        mlConfig,
        useFiveColors
      );
      
      console.log('Processing result received:', {
        success: result.success,
        hasImage: !!result.image,
        imageLength: result.image?.length,
        hasStats: !!result.stats,
        stats: result.stats
      });
      
      if (!result || !result.image) {
        throw new Error('Server returned invalid response - no image data received');
      }
      
      let finalImage = result.image;
      
      // Apply WebGL stained glass effect on frontend if enabled (GPU-accelerated)
      if (stainedGlassEnabled) {
        try {
          console.log('Applying stained glass effect (intensity: 1.0)...');
          console.log('Original image data URL length:', result.image.length);
          
          // Apply effect with high intensity for maximum visibility
          finalImage = await applyStainedGlassEffect(result.image, 1.0);
          
          console.log('Stained glass effect applied successfully');
          console.log('Final image data URL length:', finalImage.length);
          
          // Verify the image changed
          if (finalImage === result.image) {
            console.warn('Warning: Stained glass effect may not have been applied (images are identical)');
          }
        } catch (effectError) {
          console.error('Stained glass effect failed:', effectError);
          console.error('Error details:', effectError.stack);
          // If effect fails, still show the colored image
          finalImage = result.image;
        }
      }
      
      setProcessedImage(finalImage);
      setStats(result.stats);
      
      // Save to history (with error handling)
      try {
        saveImageToHistory(finalImage, getCurrentSettings(), result.stats);
      } catch (err) {
        console.warn('Failed to save to history:', err);
        // Don't show error to user - history is optional
      }
    } catch (err) {
      let errorMessage = err.message || 'Failed to process image';
      
      // Provide helpful error messages
      if (errorMessage.includes('too large') || errorMessage.includes('size')) {
        errorMessage = 'Image is too large. Please use an image smaller than 50MB or 10000x10000px.';
      } else if (errorMessage.includes('format') || errorMessage.includes('Invalid')) {
        errorMessage = 'Invalid image format. Please use PNG, JPG, or JPEG.';
      } else if (errorMessage.includes('timeout') || errorMessage.includes('long')) {
        errorMessage = 'Processing took too long. Try a smaller image or disable some effects.';
      } else if (errorMessage.includes('Network') || errorMessage.includes('Cannot connect')) {
        errorMessage = 'Cannot connect to server. Please ensure the backend is running:\n\n1. Open terminal in the backend folder\n2. Run: python app.py\n3. Server should start on http://localhost:8000';
      }
      
      setError(errorMessage);
      console.error('Processing error:', err);
    } finally {
      setIsProcessing(false);
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

  const getCurrentSettings = () => ({
    style: selectedStyle,
    stainedGlass: stainedGlassEnabled,
    lineArt: lineArtEnabled ? lineArtSettings : { enabled: false },
  });

  const handleSelectHistoryItem = (item) => {
    setProcessedImage(item.image);
    setStats(item.stats);
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
    }
  };


  // Show educational page if requested
  if (showEducationalPage) {
    return <EducationalPage onBack={() => setShowEducationalPage(false)} />;
  }

  // Show gallery page if requested
  if (showGalleryPage) {
    return <GalleryPage onBack={() => setShowGalleryPage(false)} />;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Four Color Theorem</h1>
        <p className="subtitle">Automatic Image Coloring using Graph Theory</p>
        <div style={{ display: 'flex', gap: '10px', marginTop: '15px', flexWrap: 'wrap', justifyContent: 'center' }}>
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
              transition: 'all 0.3s ease'
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
              transition: 'all 0.3s ease'
            }}
          >
            View Example Gallery
          </button>
        </div>
      </header>

      <main className="App-main">
        {serverStatus === 'disconnected' && (
          <div className="server-warning" style={{
            background: '#fff3cd',
            border: '1px solid #ffc107',
            borderRadius: '8px',
            padding: '12px',
            margin: '20px 0',
            color: '#856404'
          }}>
            <strong>Server Not Connected</strong>
            <p style={{ margin: '8px 0 0 0', fontSize: '0.9rem' }}>
              Backend server is not running. Please start it with: <code style={{background: '#f0f0f0', padding: '2px 6px', borderRadius: '3px'}}>python app.py</code> in the backend folder.
            </p>
          </div>
        )}

        <div className="upload-section">
          <ImageUploader onImageSelect={handleImageSelect} />
          {selectedImage && (
            <div className="preview-section">
              <h3>Original Image</h3>
              <img 
                src={URL.createObjectURL(selectedImage)} 
                alt="Selected" 
                className="preview-image"
              />
            </div>
          )}
        </div>

        {selectedImage && (
          <>
            {enableSmartColorSuggestions && (
              <SmartColorSuggester
                imageFile={selectedImage}
                onSelectPalette={(palette) => {
                  setSelectedPalette(palette);
                  // Automatically process image with selected palette colors
                  if (palette && palette.colors) {
                    handleProcessWithColors(palette.colors);
                  }
                }}
                selectedPaletteId={selectedPalette?.id}
                useFiveColors={useFiveColors}
              />
            )}
            <div className="controls-section">
              <div className="setting-group" style={{ marginBottom: '16px', padding: '16px', background: 'white', border: '1px solid #e0e0e0', borderRadius: '8px' }}>
                <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', fontWeight: '500', color: '#333' }}>
                  <input
                    type="checkbox"
                    checked={enableSmartColorSuggestions}
                    onChange={(e) => setEnableSmartColorSuggestions(e.target.checked)}
                    style={{ width: '18px', height: '18px', cursor: 'pointer' }}
                  />
                  <span>Enable Smart Color Suggestions</span>
                </label>
                <p className="setting-description" style={{ fontSize: '0.85em', color: '#666', margin: '8px 0 0 28px', lineHeight: '1.4' }}>
                  Get AI-powered color palette suggestions based on your image content
                </p>
              </div>
              <StyleSelector 
                selectedStyle={selectedStyle} 
                onStyleChange={handleStyleChange} 
              />
              <SegmentationSettings
                useMLSegmentation={useMLSegmentation}
                onToggleML={setUseMLSegmentation}
                segmentationMethod={segmentationMethod}
                onMethodChange={setSegmentationMethod}
                targetRegions={targetRegions}
                onTargetRegionsChange={setTargetRegions}
              />
              <FiveColorToggle
                enabled={useFiveColors}
                onToggle={setUseFiveColors}
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
                disabled={isProcessing} 
              />
            </div>
          </>
        )}

        <ImageHistory onSelectHistoryItem={handleSelectHistoryItem} />

        {isProcessing && <ProgressIndicator />}

        {error && (
          <div className="error-message">
            <p>Error: {error}</p>
          </div>
        )}

        {processedImage && (
          <ResultViewer 
            image={processedImage} 
            stats={stats} 
          />
        )}
      </main>

      <footer className="App-footer">
        <p>Upload a coloring book style image to automatically color it with 4 or 5 colors</p>
      </footer>
    </div>
  );
}

export default App;

