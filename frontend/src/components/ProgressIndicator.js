import React, { useState, useEffect } from 'react';

function ProgressIndicator({ progress = null }) {
  const [animatedProgress, setAnimatedProgress] = useState(0);

  useEffect(() => {
    if (progress !== null) {
      setAnimatedProgress(progress);
    } else {
      // Animate progress bar when no specific progress is provided
      const interval = setInterval(() => {
        setAnimatedProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 5;
        });
      }, 500);
      return () => clearInterval(interval);
    }
  }, [progress]);

  const displayProgress = progress !== null ? progress : animatedProgress;

  return (
    <div className="progress-indicator">
      <div className="spinner"></div>
      <p>Processing image... This may take a moment.</p>
      <div className="progress-bar-container">
        <div 
          className="progress-bar" 
          style={{ width: `${Math.min(displayProgress, 95)}%` }}
        ></div>
      </div>
      {progress !== null && (
        <p className="progress-text">{Math.round(displayProgress)}%</p>
      )}
    </div>
  );
}

export default ProgressIndicator;

