import React, { useState, useEffect } from 'react';

function ProgressIndicator({ progress = null, stage = null }) {
  const [animatedProgress, setAnimatedProgress] = useState(0);

  useEffect(() => {
    if (progress !== null) {
      setAnimatedProgress(progress);
    } else {
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
      <p>{stage || 'Processing image...'}</p>
      <div className="progress-bar-container">
        <div
          className="progress-bar"
          style={{ width: `${Math.min(displayProgress, 100)}%` }}
        ></div>
      </div>
      <p className="progress-text">{Math.round(displayProgress)}%</p>
    </div>
  );
}

export default ProgressIndicator;