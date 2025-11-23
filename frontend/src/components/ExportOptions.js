import React, { useState } from 'react';
import './ExportOptions.css';

function ExportOptions({ image, filename: initialFilename = 'colored-image' }) {
  const [filename, setFilename] = useState(initialFilename);
  const [exportFormat, setExportFormat] = useState('png');
  const [quality, setQuality] = useState(0.95);
  const [resolution, setResolution] = useState('1x');

  const handleDownload = () => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      let scale = 1;
      
      // Apply resolution multiplier
      if (resolution === '2x') scale = 2;
      else if (resolution === '4x') scale = 4;
      
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      // Convert to requested format
      let mimeType = 'image/png';
      let fileExtension = 'png';
      
      if (exportFormat === 'jpg' || exportFormat === 'jpeg') {
        mimeType = 'image/jpeg';
        fileExtension = 'jpg';
      }
      
      const dataUrl = canvas.toDataURL(mimeType, quality);
      
      const link = document.createElement('a');
      link.href = dataUrl;
      link.download = `${filename}.${fileExtension}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
    
    img.src = image;
  };

  const handlePDFExport = async () => {
    // For PDF export, we'll use a simple approach
    // In production, you might want to use a library like jsPDF
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        // Create a new window with the image for printing
        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
          <html>
            <head>
              <title>${filename}</title>
              <style>
                body { margin: 0; padding: 20px; }
                img { max-width: 100%; height: auto; }
                @media print {
                  body { padding: 0; }
                  img { width: 100%; }
                }
              </style>
            </head>
            <body>
              <img src="${image}" alt="${filename}" />
            </body>
          </html>
        `);
        printWindow.document.close();
        printWindow.print();
      };
      
      img.src = image;
    } catch (error) {
      console.error('PDF export error:', error);
      alert('PDF export failed. Please use the download option and print manually.');
    }
  };

  return (
    <div className="export-options">
      <h4>Export Options</h4>
      
      <div className="export-controls">
        <div className="control-group">
          <label>Filename:</label>
          <input
            type="text"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            placeholder="colored-image"
          />
        </div>

        <div className="control-group">
          <label>Format:</label>
          <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
            <option value="png">PNG (Lossless)</option>
            <option value="jpg">JPG (Compressed)</option>
          </select>
        </div>

        {exportFormat === 'jpg' && (
          <div className="control-group">
            <label>Quality: {Math.round(quality * 100)}%</label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={quality}
              onChange={(e) => setQuality(parseFloat(e.target.value))}
            />
          </div>
        )}

        <div className="control-group">
          <label>Resolution:</label>
          <select value={resolution} onChange={(e) => setResolution(e.target.value)}>
            <option value="1x">1x (Original)</option>
            <option value="2x">2x (High Resolution)</option>
            <option value="4x">4x (Ultra High Resolution)</option>
          </select>
        </div>
      </div>

      <div className="export-buttons">
        <button className="export-button primary" onClick={handleDownload}>
          Download {exportFormat.toUpperCase()}
        </button>
        <button className="export-button secondary" onClick={handlePDFExport}>
          Print/PDF
        </button>
      </div>
    </div>
  );
}

export default ExportOptions;

