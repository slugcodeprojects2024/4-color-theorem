import React, { useState, useEffect } from 'react';
import { suggestPalettes } from '../services/api';
import './PaletteSuggester.css';

function PaletteSuggester({ imageFile, onSelectPalette, selectedPaletteId, apiBaseUrl = 'http://localhost:8000' }) {
  const [suggestions, setSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);

  useEffect(() => {
    setHasAnalyzed(false);
    setSuggestions([]);
  }, [imageFile]);

  const fetchSuggestions = async () => {
    if (!imageFile) return;
    setIsLoading(true);
    setError(null);

    try {
      const data = await suggestPalettes(imageFile, 6);
      if (data.success) {
        setSuggestions(data.suggestions);
        setHasAnalyzed(true);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectPalette = (palette) => {
    onSelectPalette({
      id: palette.name.toLowerCase().replace(/\s+/g, '-'),
      name: palette.name,
      colors: palette.colors,
      style: palette.style
    });
  };

  if (!imageFile) return null;

  return (
    <div className="palette-suggester">
      <div className="suggester-header">
        <h4>AI-Suggested Palettes</h4>
        {hasAnalyzed && <button className="refresh-btn" onClick={fetchSuggestions}>Refresh</button>}
      </div>

      {isLoading && <div className="suggester-loading"><div className="loading-spinner"></div>Analyzing...</div>}
      {error && <div className="suggester-error">Error: {error}</div>}

      {suggestions.length > 0 && (
        <div className="suggestions-grid">
          {suggestions.map((palette, index) => (
            <div
              key={palette.name}
              className={`palette-card ${selectedPaletteId === palette.name.toLowerCase().replace(/\s+/g, '-') ? 'selected' : ''}`}
              onClick={() => handleSelectPalette(palette)}
            >
              <div className="palette-preview">
                {palette.colors.map((color, i) => (
                  <div 
                    key={i} 
                    className="color-swatch" 
                    style={{ backgroundColor: Array.isArray(color) ? `rgb(${color.join(',')})` : color }} 
                  />
                ))}
              </div>
              <div className="palette-info">
                <span className="palette-name">{palette.name}</span>
                <span className="palette-score">{Math.round(palette.score * 100)}% match</span>
              </div>
              {index === 0 && <div className="best-match-badge">Best Match</div>}
            </div>
          ))}
        </div>
      )}

      {!hasAnalyzed && !isLoading && (
        <button className="analyze-btn" onClick={fetchSuggestions}>
          Analyze Image for Palette Suggestions
        </button>
      )}
    </div>
  );
}

export default PaletteSuggester;

