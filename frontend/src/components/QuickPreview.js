import React, { useState } from 'react';
import './QuickPreview.css';
import { previewImage } from '../services/api';

function QuickPreview({ imageFile, onPreviewReady }) {
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [previewImageData, setPreviewImageData] = useState(null);
  const [previewError, setPreviewError] = useState(null);

  const handlePreview = async () => {
    if (!imageFile) {
      setPreviewError('Please select an image first');
      return;
    }

    setIsPreviewing(true);
    setPreviewError(null);
    setPreviewImageData(null);

    try {
      const result = await previewImage(imageFile);
      setPreviewImageData(result);
      if (onPreviewReady) {
        onPreviewReady(result);
      }
    } catch (err) {
      setPreviewError(err.message || 'Preview failed');
      console.error('Preview error:', err);
    } finally {
      setIsPreviewing(false);
    }
  };

  return (
    <div className="quick-preview">
      <div className="preview-header">
        <h3>Quick Preview</h3>
        <button
          className="preview-button"
          onClick={handlePreview}
          disabled={isPreviewing || !imageFile}
        >
          {isPreviewing ? 'Previewing...' : 'Generate Preview'}
        </button>
      </div>

      {previewError && (
        <div className="preview-error">
          <p>{previewError}</p>
        </div>
      )}

      {previewImageData && (
        <div className="preview-result">
          <p className="preview-note">
            Low-resolution preview ({previewImageData.preview_size?.[0]}x{previewImageData.preview_size?.[1]}) - 
            Processing time: {previewImageData.processing_time_ms}ms
          </p>
          <div className="preview-image-container">
            <img src={previewImageData.preview} alt="Preview" />
          </div>
        </div>
      )}
    </div>
  );
}

export default QuickPreview;

