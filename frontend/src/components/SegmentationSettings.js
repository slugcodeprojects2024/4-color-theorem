import React from 'react';
import './SegmentationSettings.css';

function SegmentationSettings({ 
  useMLSegmentation, 
  onToggleML, 
  segmentationMethod, 
  onMethodChange,
  targetRegions,
  onTargetRegionsChange
}) {
  return (
    <div className="segmentation-settings">
      <div className="settings-header">
        <h4>Segmentation Settings</h4>
      </div>
      
      <div className="setting-group">
        <label className="toggle-label">
              <input
                type="checkbox"
                checked={useMLSegmentation}
                onChange={(e) => onToggleML(e.target.checked)}
              />
              <span>Use ML-Enhanced Segmentation</span>
            </label>
            <p className="setting-description">
              Uses SLIC superpixels for better region detection on photos. May produce fewer regions than traditional edge detection.
            </p>
      </div>

      {useMLSegmentation && (
        <>
          <div className="setting-group">
            <label htmlFor="segmentation-method">Segmentation Method:</label>
            <select
              id="segmentation-method"
              value={segmentationMethod}
              onChange={(e) => onMethodChange(e.target.value)}
              className="method-select"
            >
              <option value="auto">Auto (Recommended)</option>
              <option value="slic">SLIC Superpixels</option>
              <option value="edge">Edge Detection</option>
            </select>
            <p className="setting-description">
              {segmentationMethod === 'auto' && 'Automatically selects the best method based on image type'}
              {segmentationMethod === 'slic' && 'Best for photos and complex images with many colors'}
              {segmentationMethod === 'edge' && 'Best for line art and coloring book style images'}
            </p>
          </div>

          <div className="setting-group">
            <label htmlFor="target-regions">Target Regions: {targetRegions}</label>
            <input
              id="target-regions"
              type="range"
              min="20"
              max="200"
              step="10"
              value={targetRegions}
              onChange={(e) => onTargetRegionsChange(parseInt(e.target.value))}
              className="regions-slider"
            />
            <div className="slider-labels">
              <span>Fewer (20)</span>
              <span>More (200)</span>
            </div>
            <p className="setting-description">
              Approximate number of regions to detect. Lower values = faster processing, higher values = more detail
            </p>
          </div>
        </>
      )}
    </div>
  );
}

export default SegmentationSettings;

