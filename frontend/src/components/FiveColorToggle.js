import React from 'react';
import './FiveColorToggle.css';

function FiveColorToggle({ enabled, onToggle }) {
  return (
    <div className="five-color-toggle">
      <div className="setting-group">
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onToggle(e.target.checked)}
          />
          <span>Use 5-Color Mode</span>
        </label>
        <p className="setting-description">
          Enable 5-color mode for more complex images. The 4-color theorem guarantees 4 colors are sufficient for planar graphs, but 5-color mode can help with non-planar or very complex graphs.
        </p>
      </div>
    </div>
  );
}

export default FiveColorToggle;

