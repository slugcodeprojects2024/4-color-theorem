import React, { useState, useCallback } from 'react';
import './CustomColorPicker.css';

const DEFAULT_COLORS = ['#DC143C', '#00BFFF', '#32CD32', '#FFD700'];

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return [r, g, b];
}

function rgbToHex(r, g, b) {
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

function CustomColorPicker({ enabled, onToggle, colors, onColorsChange }) {
  const [hexValues, setHexValues] = useState(() =>
    (colors || DEFAULT_COLORS.map(hexToRgb)).map(([r, g, b]) => rgbToHex(r, g, b))
  );

  const handleColorChange = useCallback((index, hex) => {
    const updated = [...hexValues];
    updated[index] = hex;
    setHexValues(updated);
    onColorsChange(updated.map(h => hexToRgb(h)));
  }, [hexValues, onColorsChange]);

  const handleHexInput = useCallback((index, raw) => {
    const updated = [...hexValues];
    updated[index] = raw;
    setHexValues(updated);

    // Only push to parent when it's a valid 7-char hex
    if (/^#[0-9a-fA-F]{6}$/.test(raw)) {
      onColorsChange(updated.map(h => {
        if (/^#[0-9a-fA-F]{6}$/.test(h)) return hexToRgb(h);
        return hexToRgb(DEFAULT_COLORS[0]);
      }));
    }
  }, [hexValues, onColorsChange]);

  const randomize = useCallback(() => {
    const rand = () => Math.floor(Math.random() * 200 + 30);
    const newHexes = Array.from({ length: 4 }, () => rgbToHex(rand(), rand(), rand()));
    setHexValues(newHexes);
    onColorsChange(newHexes.map(h => hexToRgb(h)));
  }, [onColorsChange]);

  return (
    <div className={`custom-color-picker ${enabled ? 'active' : ''}`}>
      <div className="ccp-toggle">
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onToggle(e.target.checked)}
          />
          <span>Use Custom Colors</span>
        </label>
        <p className="toggle-description">Pick your own four colors instead of a preset palette</p>
      </div>

      {enabled && (
        <div className="ccp-body">
          <div className="ccp-swatches">
            {hexValues.map((hex, i) => (
              <div className="ccp-swatch-group" key={i}>
                <label className="ccp-swatch-label">Color {i + 1}</label>
                <div className="ccp-swatch-row">
                  <div
                    className="ccp-swatch"
                    style={{ backgroundColor: /^#[0-9a-fA-F]{6}$/.test(hex) ? hex : '#ccc' }}
                  >
                    <input
                      type="color"
                      className="ccp-native-picker"
                      value={/^#[0-9a-fA-F]{6}$/.test(hex) ? hex : '#cccccc'}
                      onChange={(e) => handleColorChange(i, e.target.value)}
                    />
                  </div>
                  <input
                    type="text"
                    className="ccp-hex-input"
                    value={hex}
                    maxLength={7}
                    spellCheck={false}
                    onChange={(e) => handleHexInput(i, e.target.value)}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="ccp-actions">
            <button className="ccp-randomize" onClick={randomize} type="button">
              Randomize
            </button>
          </div>

          <div className="ccp-preview-strip">
            {hexValues.map((hex, i) => (
              <div
                key={i}
                className="ccp-preview-band"
                style={{ backgroundColor: /^#[0-9a-fA-F]{6}$/.test(hex) ? hex : '#ccc' }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default CustomColorPicker;