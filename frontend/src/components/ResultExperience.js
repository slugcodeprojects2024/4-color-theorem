import React, { useState, useEffect } from 'react';
import ResultViewer from './ResultViewer';
import ColoringAnimation from './ColoringAnimation';
import StainedGlassViewer from './StainedGlassViewer';
import './ResultExperience.css';

/**
 * Tabbed result area:
 *   Result        - the flat colored image (download/export as before)
 *   Live Coloring - watch the page color itself in, region by region
 *   Stained Glass - three.js WebGL glass shader with interactive light
 *   Line Art      - shown for the photo pipeline (intermediate output)
 */
function ResultExperience({
  image,
  stats,
  animation,
  lineart,
  originalUrl,
  defaultTab = 'result',
}) {
  const [tab, setTab] = useState(defaultTab);

  useEffect(() => {
    setTab(defaultTab);
  }, [defaultTab, image]);

  const tabs = [
    { id: 'result', label: 'Result' },
    ...(animation ? [{ id: 'animation', label: 'Live Coloring' }] : []),
    { id: 'glass', label: 'Stained Glass' },
    ...(lineart ? [{ id: 'lineart', label: 'Line Art' }] : []),
  ];

  return (
    <div className="result-experience">
      <div className="result-tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`result-tab ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'result' && <ResultViewer image={image} stats={stats} />}

      {tab === 'animation' && animation && (
        <ColoringAnimation
          animation={animation}
          originalUrl={lineart ? originalUrl : null}
          lineart={lineart}
        />
      )}

      {tab === 'glass' && <StainedGlassViewer image={image} />}

      {tab === 'lineart' && lineart && (
        <div className="lineart-panel">
          <h3>Generated Line Art</h3>
          <img src={lineart} alt="Line art" className="lineart-image" />
          <a href={lineart} download="line-art.png" className="lineart-download">
            Download line art
          </a>
        </div>
      )}
    </div>
  );
}

export default ResultExperience;
