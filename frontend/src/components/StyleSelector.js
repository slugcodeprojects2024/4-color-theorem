import React from 'react';

const STYLES = [
  { value: 'vibrant', label: 'Vibrant', description: 'Bright, bold colors' },
  { value: 'pastel', label: 'Pastel', description: 'Soft, gentle colors' },
  { value: 'earth', label: 'Earth', description: 'Natural, earthy tones' },
  { value: 'monochrome', label: 'Monochrome', description: 'Grayscale shades' },
  { value: 'ocean', label: 'Ocean', description: 'Blues and aquas' },
  { value: 'sunset', label: 'Sunset', description: 'Warm oranges and pinks' },
  { value: 'forest', label: 'Forest', description: 'Greens and browns' },
  { value: 'neon', label: 'Neon', description: 'Bright fluorescent colors' }
];

function StyleSelector({ selectedStyle, onStyleChange }) {
  return (
    <div className="style-selector">
      <h3>Color Palette</h3>
      <div className="style-options">
        {STYLES.map(style => (
          <button
            key={style.value}
            className={`style-option ${selectedStyle === style.value ? 'active' : ''}`}
            onClick={() => onStyleChange(style.value)}
            title={style.description}
          >
            {style.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default StyleSelector;

