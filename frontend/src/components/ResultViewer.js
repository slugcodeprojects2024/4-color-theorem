import React from 'react';
import ExportOptions from './ExportOptions';

function ResultViewer({ image, stats }) {

  return (
    <div className="result-viewer">
      <h3>Colored Result</h3>
      <div className="result-image-container">
        <img src={image} alt="Colored result" className="result-image" />
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

