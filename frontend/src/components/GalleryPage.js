import React from 'react';
import './GalleryPage.css';

function GalleryPage({ onBack }) {
  // Helper function to encode image paths with spaces
  const getImagePath = (filename) => {
    const basePath = process.env.PUBLIC_URL || '';
    const encodedFilename = encodeURIComponent(filename);
    const path = `${basePath}/examples/${encodedFilename}`;
    console.log(`Image path - Original: "${filename}", Encoded: "${encodedFilename}", Full path: "${path}"`);
    return path;
  };

  const examples = [
    {
      title: 'Mandala Pattern',
      beforeSrc: getImagePath('mandala.webp'),
      afterSrc: getImagePath('colored-image-mandala.png'),
      beforeAlt: 'Before: original mandala pattern',
      afterAlt: 'After: automatically colored mandala pattern',
    },
    {
      title: 'Ocean Scene',
      beforeSrc: getImagePath('ocean-image-original.webp'),
      afterSrc: getImagePath('colored-image-ocean.png'),
      beforeAlt: 'Before: original ocean image',
      afterAlt: 'After: automatically colored ocean scene',
    },
    {
      title: 'Mushroom Illustration',
      beforeSrc: getImagePath('originalmushroom.jpg'),
      afterSrc: getImagePath('coloredmushroom.png'),
      beforeAlt: 'Before: original mushroom illustration',
      afterAlt: 'After: automatically colored mushroom',
    },
  ];

  return (
    <div className="gallery-page">
      <div className="gallery-container">
        <button className="back-button" onClick={onBack}>
          ← Back to App
        </button>

        <header className="gallery-header">
          <h1>Example Gallery</h1>
          <p className="subtitle">Before and After: Automatic Coloring Results</p>
        </header>

        <section className="gallery-section">
          <div className="gallery-intro">
            <p>
              Here are some examples of images that have been automatically colored using the Four Color Theorem.
              Each example shows the original image on the left and the colored result on the right.
            </p>
          </div>

          <div className="example-gallery">
            {examples.map((ex) => (
              <div className="example-item" key={ex.title}>
                <h3>{ex.title}</h3>
                <div className="split-image-container">
                  <div className="split-image-wrapper">
                    <div className="split-image-half before-half">
                      <div className="split-label">Before</div>
                      <img 
                        src={ex.beforeSrc} 
                        alt={ex.beforeAlt} 
                        loading="lazy"
                        onError={(e) => {
                          console.error('Failed to load image:', ex.beforeSrc);
                          console.error('Trying path:', e.target.src);
                          console.error('Expected filename:', ex.title === 'Ocean Scene' ? 'ocean image original.webp' : 'originalmushroom.jpg');
                        }}
                        onLoad={() => {
                          console.log('Successfully loaded:', ex.beforeSrc);
                        }}
                      />
                    </div>
                    <div className="split-divider"></div>
                    <div className="split-image-half after-half">
                      <div className="split-label">After</div>
                      <img 
                        src={ex.afterSrc} 
                        alt={ex.afterAlt} 
                        loading="lazy"
                        onError={(e) => {
                          console.error('Failed to load image:', ex.afterSrc);
                          console.error('Trying path:', e.target.src);
                          console.error('Expected filename:', ex.title === 'Ocean Scene' ? 'colored-image ocean.png' : 'coloredmushroom.png');
                        }}
                        onLoad={() => {
                          console.log('Successfully loaded:', ex.afterSrc);
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        <footer className="gallery-footer">
          <button className="back-button" onClick={onBack}>
            ← Back to App
          </button>
        </footer>
      </div>
    </div>
  );
}

export default GalleryPage;

