import React, { useState } from 'react';
import ExportOptions from './ExportOptions';

function ResultViewer({ image, stats }) {
  const [imageError, setImageError] = useState(false);

  if (!image) {
    return (
      <div className="result-viewer">
        <h3>Colored Result</h3>
        <p>No image data available</p>
      </div>
    );
  }

  return (
    <div className="result-viewer">
      <h3>Colored Result</h3>
      <div className="result-image-container">
        {imageError ? (
          <div style={{ padding: '20px', textAlign: 'center', color: '#ff4444' }}>
            <p>Failed to load image</p>
            <p style={{ fontSize: '0.8rem', marginTop: '10px' }}>
              Image data length: {image?.length || 0} characters
            </p>
          </div>
        ) : (
          <img 
            src={image} 
            alt="Colored result" 
            className="result-image"
            onError={(e) => {
              console.error('Image load error:', e);
              console.error('Image source:', image?.substring(0, 100));
              setImageError(true);
            }}
            onLoad={() => {
              console.log('Image loaded successfully');
              setImageError(false);
            }}
          />
        )}
      </div>
      
      {stats && (
        <div className="result-stats">
          <h4>Statistics</h4>
          <ul>
            <li>Regions detected: {stats.regions}</li>
            <li>Colors used: {stats.colors_used}</li>
            <li>Graph nodes: {stats.graph_nodes}</li>
            <li>Graph edges: {stats.graph_edges}</li>
            {stats.image_resized && (
              <li className="info-note">
                Image was resized from {stats.original_size?.[0]}x{stats.original_size?.[1]} 
                to {stats.processed_size?.[0]}x{stats.processed_size?.[1]} for processing
              </li>
            )}
          </ul>
        </div>
      )}

      <ExportOptions image={image} />
    </div>
  );
}

export default ResultViewer;

