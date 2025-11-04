import React from 'react';

function ResultViewer({ image, stats }) {
  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = image;
    link.download = 'colored-image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

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
          </ul>
        </div>
      )}

      <button className="download-button" onClick={handleDownload}>
        Download Image
      </button>
    </div>
  );
}

export default ResultViewer;

