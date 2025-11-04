import React from 'react';

function ProgressIndicator() {
  return (
    <div className="progress-indicator">
      <div className="spinner"></div>
      <p>Processing image... This may take a moment.</p>
    </div>
  );
}

export default ProgressIndicator;

