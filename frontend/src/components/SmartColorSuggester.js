/**
 * Smart Color Suggester - Hybrid AI Component
 * 
 * Layer 1: Instant results from server-side OpenCV
 * Layer 2: Enhanced results from browser AI (optional, on-demand)
 */

import React, { useState, useEffect } from 'react';
import { useBrowserAI } from '../hooks/useBrowserAI';
import './SmartColorSuggester.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function SmartColorSuggester({ 
  imageFile, 
  onSelectPalette, 
  selectedPaletteId,
  useFiveColors = false 
}) {
  // Server analysis state (Layer 1)
  const [serverAnalysis, setServerAnalysis] = useState(null);
  const [serverLoading, setServerLoading] = useState(false);
  
  // Browser AI state (Layer 2)
  const { analyzeImage, isLoading: aiLoading, loadProgress } = useBrowserAI();
  const [aiAnalysis, setAiAnalysis] = useState(null);
  const [aiEnabled, setAiEnabled] = useState(false);
  
  // Combined state
  const [suggestions, setSuggestions] = useState([]);
  const [showDetails, setShowDetails] = useState(false);
  const [error, setError] = useState(null);

  // Extend palette to 5 colors
  const extendToFive = React.useCallback((colors) => {
    if (colors.length >= 5) return colors.slice(0, 5);
    
    const extended = [...colors];
    while (extended.length < 5) {
      const base = extended[extended.length - 1];
      const factor = (base[0] + base[1] + base[2]) > 384 ? 0.7 : 1.3;
      extended.push([
        Math.min(255, Math.max(0, Math.round(base[0] * factor))),
        Math.min(255, Math.max(0, Math.round(base[1] * factor))),
        Math.min(255, Math.max(0, Math.round(base[2] * factor)))
      ]);
    }
    return extended;
  }, []);

  // Merge server and AI suggestions
  const mergeSuggestions = React.useCallback((server, ai) => {
    const merged = [];
    const seen = new Set();

    // AI results first (higher quality)
    if (ai?.suggested_palettes) {
      for (const palette of ai.suggested_palettes) {
        const key = palette.colors.map(c => c.join(',')).join('|');
        if (!seen.has(key)) {
          seen.add(key);
          merged.push({
            ...palette,
            source_layer: 'ai',
            priority: 1
          });
        }
      }
    }

    // Server results
    if (server?.suggested_palettes) {
      for (const palette of server.suggested_palettes) {
        const key = palette.colors.map(c => c.join(',')).join('|');
        if (!seen.has(key)) {
          seen.add(key);
          merged.push({
            ...palette,
            source_layer: 'server',
            priority: 2
          });
        }
      }
    }

    // Sort by score and priority
    merged.sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      return b.score - a.score;
    });

    // Extend to 5 colors if needed
    if (useFiveColors) {
      return merged.map(p => ({
        ...p,
        colors: extendToFive(p.colors)
      }));
    }

    return merged.slice(0, 8);
  }, [useFiveColors, extendToFive]);

  // Reset when image changes
  useEffect(() => {
    setServerAnalysis(null);
    setAiAnalysis(null);
    setSuggestions([]);
    setError(null);
    setAiEnabled(false);
  }, [imageFile]);

  // Run server analysis automatically when image is uploaded
  useEffect(() => {
    if (imageFile && !serverAnalysis && !serverLoading) {
      analyzeWithServer();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageFile]);

  // Merge suggestions when analyses update
  useEffect(() => {
    const merged = mergeSuggestions(serverAnalysis, aiAnalysis);
    setSuggestions(merged);
  }, [serverAnalysis, aiAnalysis, mergeSuggestions]);

  // Server-side analysis (Layer 1)
  const analyzeWithServer = async () => {
    if (!imageFile) return;
    
    setServerLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze-colors`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Server analysis failed');

      const data = await response.json();
      if (data.success) {
        setServerAnalysis(data.analysis);
      }
    } catch (err) {
      setError(`Server: ${err.message}`);
    } finally {
      setServerLoading(false);
    }
  };

  // Browser AI analysis (Layer 2)
  const analyzeWithAI = async () => {
    if (!imageFile) return;
    
    setAiEnabled(true);
    setError(null);

    try {
      const result = await analyzeImage(imageFile);
      setAiAnalysis(result);
    } catch (err) {
      setError(`AI: ${err.message}`);
    }
  };

  // Handle palette selection
  const handleSelectPalette = (palette) => {
    if (onSelectPalette) {
      onSelectPalette({
        id: palette.name.toLowerCase().replace(/\s+/g, '-'),
        name: palette.name,
        colors: palette.colors,
        is_smart: palette.is_smart,
        source: palette.source_layer
      });
    }
  };

  // Get display info
  const getDetectedInfo = () => {
    if (aiAnalysis) {
      return {
        subject: aiAnalysis.detected_subject || 'unknown',
        style: aiAnalysis.detected_style || 'unknown',
        confidence: aiAnalysis.confidence || 0,
        source: 'AI'
      };
    }
    if (serverAnalysis) {
      return {
        subject: serverAnalysis.estimated_subjects?.[0] || 'unknown',
        style: serverAnalysis.style || 'unknown',
        confidence: serverAnalysis.confidence || 0,
        source: 'Pattern'
      };
    }
    return null;
  };

  const detectedInfo = getDetectedInfo();

  if (!imageFile) return null;

  return (
    <div className="smart-color-suggester">
      <div className="smart-header">
        <h4>🎨 Smart Color Suggestions</h4>
        <div className="header-badges">
          {serverAnalysis && <span className="badge server">Pattern ✓</span>}
          {aiAnalysis && <span className="badge ai">AI ✓</span>}
        </div>
      </div>

      {/* Loading states */}
      {serverLoading && (
        <div className="loading-bar">
          <div className="loading-text">Analyzing patterns...</div>
        </div>
      )}

      {/* AI Enhancement Section */}
      {!aiAnalysis && serverAnalysis && (
        <div className="ai-enhance-section">
          {!aiEnabled ? (
            <button className="ai-enhance-btn" onClick={analyzeWithAI}>
              <span className="ai-icon">🤖</span>
              <span className="ai-text">
                <strong>Enhance with AI</strong>
                <small>Better detection using browser AI (~150MB download, cached)</small>
              </span>
            </button>
          ) : aiLoading ? (
            <div className="ai-loading">
              <div className="ai-progress-bar">
                <div className="ai-progress-fill" style={{ width: `${loadProgress}%` }}></div>
              </div>
              <span>{loadProgress < 100 ? `Loading AI model... ${loadProgress}%` : 'Analyzing...'}</span>
            </div>
          ) : null}
        </div>
      )}

      {/* Error display */}
      {error && <div className="error-message">⚠️ {error}</div>}

      {/* Detection results */}
      {detectedInfo && (
        <div className="detection-results">
          <div className="detection-row">
            <span className="detection-label">Detected:</span>
            <span className="detection-value">{detectedInfo.subject.replace(/_/g, ' ')}</span>
            <span className="detection-source">({detectedInfo.source})</span>
          </div>
          <div className="detection-row">
            <span className="detection-label">Style:</span>
            <span className="detection-value">{detectedInfo.style.replace(/_/g, ' ')}</span>
          </div>
          <div className="confidence-row">
            <div className="confidence-bar">
              <div 
                className="confidence-fill" 
                style={{ width: `${detectedInfo.confidence * 100}%` }}
              ></div>
            </div>
            <span className="confidence-text">
              {Math.round(detectedInfo.confidence * 100)}% confidence
            </span>
          </div>
        </div>
      )}

      {/* AI Classifications (detailed view) */}
      {aiAnalysis && (
        <div className="ai-details">
          <button 
            className="details-toggle"
            onClick={() => setShowDetails(!showDetails)}
          >
            {showDetails ? '▼' : '▶'} AI Classifications
          </button>
          {showDetails && (
            <div className="classifications-list">
              {aiAnalysis.classifications?.slice(0, 5).map((c, i) => (
                <div key={i} className="classification-item">
                  <span className="classification-label">
                    {c.label.replace('a coloring page of ', '')}
                  </span>
                  <span className="classification-score">
                    {Math.round(c.score * 100)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Palette suggestions */}
      {suggestions.length > 0 && (
        <div className="palette-suggestions">
          <h5>Suggested Palettes</h5>
          <div className="palette-grid">
            {suggestions.map((palette, index) => (
              <div
                key={`${palette.name}-${index}`}
                className={`palette-card ${
                  selectedPaletteId === palette.name.toLowerCase().replace(/\s+/g, '-') 
                    ? 'selected' 
                    : ''
                } ${palette.source_layer === 'ai' ? 'ai-palette' : ''}`}
                onClick={() => handleSelectPalette(palette)}
              >
                {/* Badges */}
                {palette.is_smart && (
                  <div className={`smart-badge ${palette.source_layer}`}>
                    {palette.source_layer === 'ai' ? '🤖 AI' : '⚡'}
                  </div>
                )}
                {index === 0 && <div className="best-badge">Best</div>}

                {/* Color preview */}
                <div className="palette-colors">
                  {palette.colors.map((color, i) => (
                    <div
                      key={i}
                      className="color-block"
                      style={{
                        backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`
                      }}
                    />
                  ))}
                </div>

                {/* Info */}
                <div className="palette-info">
                  <span className="palette-name">{palette.name}</span>
                  <span className="palette-desc">{palette.description}</span>
                  {palette.score && (
                    <span className="palette-score">
                      {Math.round(palette.score * 100)}% match
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default SmartColorSuggester;

