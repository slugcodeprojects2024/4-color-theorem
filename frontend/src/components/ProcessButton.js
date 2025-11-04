import React from 'react';

function ProcessButton({ onProcess, disabled }) {
  return (
    <button 
      className="process-button"
      onClick={onProcess}
      disabled={disabled}
    >
      {disabled ? 'Processing...' : 'Color Image'}
    </button>
  );
}

export default ProcessButton;

