import React from 'react';
import './LineArtConverter.css';

function LineArtConverter({ enabled, onToggle, settings, onSettingsChange }) {

  const handleThicknessChange = (e) => {
    onSettingsChange({
      ...settings,
      lineThickness: e.target.value
    });
  };

  const handleDetailChange = (e) => {
    onSettingsChange({
      ...settings,
      detailLevel: e.target.value
    });
  };

  const handleContrastChange = (e) => {
    onSettingsChange({
      ...settings,
      contrast: parseFloat(e.target.value)
    });
  };

  return (
    <div className="line-art-converter">
      <div className="line-art-toggle">
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onToggle(e.target.checked)}
          />
          <span>Convert Photo to Line Art</span>
        </label>
        <p className="toggle-description">
          Transform regular photos into coloring book pages
        </p>
      </div>

      {enabled && (
        <div className="line-art-settings">
          <div className="setting-group">
            <label>Line Thickness</label>
            <select
              value={settings.lineThickness}
              onChange={handleThicknessChange}
            >
              <option value="thin">Thin</option>
              <option value="medium">Medium</option>
              <option value="thick">Thick</option>
            </select>
          </div>

          <div className="setting-group">
            <label>Detail Level</label>
            <select
              value={settings.detailLevel}
              onChange={handleDetailChange}
            >
              <option value="simple">Simple</option>
              <option value="detailed">Detailed</option>
            </select>
          </div>

          <div className="setting-group">
            <label>
              Contrast: {settings.contrast.toFixed(1)}
            </label>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={settings.contrast}
              onChange={handleContrastChange}
            />
            <div className="range-labels">
              <span>Low</span>
              <span>High</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default LineArtConverter;

