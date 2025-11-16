import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import ProcessButton from './components/ProcessButton';
import ProgressIndicator from './components/ProgressIndicator';
import ResultViewer from './components/ResultViewer';
import StyleSelector from './components/StyleSelector';
import StainedGlassToggle from './components/StainedGlassToggle';
import { processImage } from './services/api';
import { applyStainedGlassEffect } from './effects/stainedGlassEffect';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState('vibrant');
  const [stainedGlassEnabled, setStainedGlassEnabled] = useState(false);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelect = (imageFile) => {
    setSelectedImage(imageFile);
    setProcessedImage(null);
    setStats(null);
    setError(null);
  };

  const handleProcess = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Always get colored image from backend (stained glass disabled on backend for speed)
      const result = await processImage(selectedImage, selectedStyle, false);
      
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
    } catch (err) {
      setError(err.message || 'Failed to process image');
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

  return (
    <div className="App">
      <header className="App-header">
        <h1>Four Color Theorem</h1>
        <p className="subtitle">Automatic Image Coloring using Graph Theory</p>
      </header>

      <main className="App-main">
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
          <div className="controls-section">
            <StyleSelector 
              selectedStyle={selectedStyle} 
              onStyleChange={handleStyleChange} 
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
        )}

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
        <p>Upload a coloring book style image to automatically color it with at most 4 colors</p>
      </footer>
    </div>
  );
}

export default App;

