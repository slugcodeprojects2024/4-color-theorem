import React from 'react';

function StainedGlassToggle({ enabled, onToggle }) {
  return (
    <div className="stained-glass-toggle">
      <label className="toggle-label">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
        />
        <span>Enable Stained Glass Effect</span>
      </label>
      <p className="toggle-hint">Add a stained glass texture effect to the result</p>
    </div>
  );
}

export default StainedGlassToggle;

