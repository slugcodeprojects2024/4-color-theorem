import React, { useState, useEffect } from 'react';
import './ImageHistory.css';

const STORAGE_KEY = 'fourColorImageHistory';
const MAX_HISTORY_ITEMS = 10;

function ImageHistory({ onSelectHistoryItem }) {
  const [history, setHistory] = useState([]);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = () => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        // Limit to MAX_HISTORY_ITEMS in case storage was corrupted
        const limited = parsed.slice(0, MAX_HISTORY_ITEMS);
        setHistory(limited);
      }
    } catch (e) {
      console.error('Failed to load history:', e);
      // If loading fails, clear corrupted data
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch (clearErr) {
        console.error('Failed to clear corrupted history:', clearErr);
      }
    }
  };

  const saveToHistory = async (imageData, settings, stats) => {
    // Compress image data by reducing quality or storing as thumbnail
    const compressImageData = (dataUrl) => {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          // Create thumbnail (max 200x200) to save space
          const maxSize = 200;
          let width = img.width;
          let height = img.height;
          
          if (width > height) {
            if (width > maxSize) {
              height = (height * maxSize) / width;
              width = maxSize;
            }
          } else {
            if (height > maxSize) {
              width = (width * maxSize) / height;
              height = maxSize;
            }
          }
          
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, width, height);
          
          // Use lower quality JPEG to save space
          const compressed = canvas.toDataURL('image/jpeg', 0.7);
          resolve(compressed);
        };
        img.onerror = () => resolve(dataUrl); // Fallback to original if compression fails
        img.src = dataUrl;
      });
    };

    try {
      // Save with compression
      const compressedImage = await compressImageData(imageData);
      const historyItem = {
        id: Date.now(),
        image: compressedImage,
        settings: settings,
        stats: stats,
        timestamp: new Date().toISOString(),
      };

      const updated = [historyItem, ...history].slice(0, MAX_HISTORY_ITEMS);
      setHistory(updated);
      
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      } catch (e) {
        console.error('Failed to save history:', e);
        // If storage is still full, reduce to fewer items
        if (e.name === 'QuotaExceededError' || e.name === 'NS_ERROR_DOM_QUOTA_REACHED') {
          // Try with even fewer items
          for (let count = 5; count >= 1; count--) {
            try {
              const reduced = updated.slice(0, count);
              localStorage.setItem(STORAGE_KEY, JSON.stringify(reduced));
              setHistory(reduced);
              console.warn(`History reduced to ${count} items due to storage limits`);
              break;
            } catch (err) {
              if (count === 1) {
                // Last resort: clear history
                localStorage.removeItem(STORAGE_KEY);
                setHistory([]);
                console.warn('History cleared due to storage limits');
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Failed to compress and save image to history:', error);
      // Silently fail - history is optional
    }
  };

  const clearHistory = () => {
    if (window.confirm('Clear all image history?')) {
      setHistory([]);
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch (e) {
        console.error('Failed to clear history:', e);
      }
    }
  };

  const removeHistoryItem = (id, e) => {
    e.stopPropagation();
    const updated = history.filter(item => item.id !== id);
    setHistory(updated);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch (e) {
      console.error('Failed to update history:', e);
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  // Expose saveToHistory function via ref or context
  // For now, we'll use a custom hook pattern
  useEffect(() => {
    window.__saveToImageHistory = saveToHistory;
    return () => {
      delete window.__saveToImageHistory;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history]);

  if (history.length === 0 && !isExpanded) {
    return null;
  }

  return (
    <div className="image-history">
      <div className="history-header" onClick={() => setIsExpanded(!isExpanded)}>
        <h3>Image History ({history.length})</h3>
        <span className="history-toggle">{isExpanded ? '−' : '+'}</span>
      </div>

      {isExpanded && (
        <div className="history-content">
          {history.length === 0 ? (
            <p className="history-empty">No history yet. Process an image to see it here.</p>
          ) : (
            <>
              <div className="history-grid">
                {history.map(item => (
                  <div
                    key={item.id}
                    className="history-item"
                    onClick={() => onSelectHistoryItem(item)}
                  >
                    <div className="history-item-image">
                      <img src={item.image} alt="History" />
                      <button
                        className="history-item-delete"
                        onClick={(e) => removeHistoryItem(item.id, e)}
                      >
                        ×
                      </button>
                    </div>
                    <div className="history-item-info">
                      <span className="history-item-style">{item.settings?.style || 'vibrant'}</span>
                      <span className="history-item-time">{formatTimestamp(item.timestamp)}</span>
                    </div>
                  </div>
                ))}
              </div>
              <button className="history-clear" onClick={clearHistory}>
                Clear History
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Export function to save to history from outside
export const saveImageToHistory = (imageData, settings, stats) => {
  if (window.__saveToImageHistory) {
    // Call the async function but don't block
    window.__saveToImageHistory(imageData, settings, stats).catch(err => {
      console.warn('Failed to save image to history:', err);
    });
  }
};

export default ImageHistory;

