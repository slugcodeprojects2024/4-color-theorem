import React, { useEffect, useRef, useState, useCallback } from 'react';
import './ColoringAnimation.css';

/**
 * Real-time coloring window.
 *
 * The backend ships a packed label map (region ids encoded into RGB),
 * a per-region color table, a paint order, and an anti-aliased line
 * overlay. This component decodes them once and then animates the page
 * coloring itself region-by-region on a <canvas>.
 *
 * For the photo pipeline it first crossfades the original photo into
 * the generated line art, then colors it in.
 */

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function ColoringAnimation({ animation, originalUrl, lineart }) {
  const canvasRef = useRef(null);
  const stateRef = useRef(null); // decoded data lives here, not in React state
  const rafRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState(1);
  const speedRef = useRef(1);
  const [phase, setPhase] = useState('idle'); // idle | lineart | coloring | done

  useEffect(() => { speedRef.current = speed; }, [speed]);

  // ---- Decode payload once --------------------------------------------
  useEffect(() => {
    if (!animation) return;
    let cancelled = false;

    (async () => {
      try {
        const { width, height } = animation;
        const labelImg = await loadImage(animation.label_map);
        const off = document.createElement('canvas');
        off.width = width;
        off.height = height;
        const octx = off.getContext('2d', { willReadFrequently: true });
        octx.drawImage(labelImg, 0, 0);
        const ld = octx.getImageData(0, 0, width, height).data;

        // Unpack RGB -> Uint32 label per pixel
        const n = width * height;
        const labels = new Uint32Array(n);
        for (let i = 0; i < n; i++) {
          const j = i * 4;
          labels[i] = (ld[j] << 16) | (ld[j + 1] << 8) | ld[j + 2];
        }

        // Bucket pixel indices per region (single pass)
        const pixelsByRegion = new Map();
        for (let i = 0; i < n; i++) {
          const r = labels[i];
          let arr = pixelsByRegion.get(r);
          if (!arr) { arr = []; pixelsByRegion.set(r, arr); }
          arr.push(i);
        }

        let lineOverlay = null;
        if (animation.line_overlay) {
          lineOverlay = await loadImage(animation.line_overlay);
        }
        let origImg = null;
        if (originalUrl) {
          try { origImg = await loadImage(originalUrl); } catch (e) { /* optional */ }
        }
        let lineartImg = null;
        if (lineart) {
          lineartImg = await loadImage(lineart);
        }

        if (cancelled) return;

        stateRef.current = {
          width, height, pixelsByRegion,
          order: animation.paint_order,
          colors: animation.region_colors,
          lineOverlay, origImg, lineartImg,
          frame: null, // ImageData being painted
          nextRegion: 0,
        };
        setReady(true);
      } catch (err) {
        console.error('Failed to decode animation payload:', err);
      }
    })();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [animation, originalUrl, lineart]);

  // ---- Drawing helpers -------------------------------------------------
  const drawLines = useCallback((ctx) => {
    const s = stateRef.current;
    if (s && s.lineOverlay) ctx.drawImage(s.lineOverlay, 0, 0);
  }, []);

  const resetCanvas = useCallback(() => {
    const s = stateRef.current;
    const canvas = canvasRef.current;
    if (!s || !canvas) return;
    canvas.width = s.width;
    canvas.height = s.height;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, s.width, s.height);
    // Start as a blank "page": white with the line work
    drawLines(ctx);
    // Fresh paint buffer (white)
    s.frame = ctx.createImageData(s.width, s.height);
    const fd = s.frame.data;
    for (let i = 0; i < fd.length; i += 4) {
      fd[i] = 255; fd[i + 1] = 255; fd[i + 2] = 255; fd[i + 3] = 255;
    }
    s.nextRegion = 0;
    setProgress(0);
  }, [drawLines]);

  // ---- Animation loop --------------------------------------------------
  const colorStep = useCallback(() => {
    const s = stateRef.current;
    const canvas = canvasRef.current;
    if (!s || !canvas) return;
    const ctx = canvas.getContext('2d');
    const total = s.order.length;

    // Regions per frame: finish in ~6s at 60fps at 1x speed
    const perFrame = Math.max(1, Math.ceil((total / 360) * speedRef.current));

    const fd = s.frame.data;
    let painted = 0;
    while (s.nextRegion < total && painted < perFrame) {
      const region = s.order[s.nextRegion];
      const color = s.colors[String(region)] || [200, 200, 200];
      const px = s.pixelsByRegion.get(region);
      if (px) {
        for (let k = 0; k < px.length; k++) {
          const j = px[k] * 4;
          fd[j] = color[0]; fd[j + 1] = color[1]; fd[j + 2] = color[2];
        }
      }
      s.nextRegion++;
      painted++;
    }

    ctx.putImageData(s.frame, 0, 0);
    drawLines(ctx);
    setProgress(Math.round((s.nextRegion / total) * 100));

    if (s.nextRegion < total) {
      rafRef.current = requestAnimationFrame(colorStep);
    } else {
      setPlaying(false);
      setPhase('done');
    }
  }, [drawLines]);

  const play = useCallback(async () => {
    const s = stateRef.current;
    const canvas = canvasRef.current;
    if (!s || !canvas || playing) return;
    setPlaying(true);
    resetCanvas();
    const ctx = canvas.getContext('2d');

    // Phase 1: photo -> line art crossfade (photo pipeline only)
    if (s.origImg && s.lineartImg) {
      setPhase('lineart');
      const steps = 45;
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, s.width, s.height);
        ctx.globalAlpha = 1 - t;
        ctx.drawImage(s.origImg, 0, 0, s.width, s.height);
        ctx.globalAlpha = t;
        ctx.drawImage(s.lineartImg, 0, 0, s.width, s.height);
        ctx.globalAlpha = 1;
        // eslint-disable-next-line no-await-in-loop
        await new Promise((r) => setTimeout(r, 22 / speedRef.current));
      }
      await new Promise((r) => setTimeout(r, 350));
      resetCanvas();
    }

    // Phase 2: color it in
    setPhase('coloring');
    rafRef.current = requestAnimationFrame(colorStep);
  }, [playing, resetCanvas, colorStep]);

  useEffect(() => {
    if (ready) {
      resetCanvas();
      play();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  if (!animation) return null;

  return (
    <div className="coloring-animation">
      <div className="coloring-animation-toolbar">
        <button
          className="anim-btn"
          onClick={play}
          disabled={!ready || playing}
        >
          {phase === 'done' ? 'Replay' : playing ? 'Coloring…' : 'Play'}
        </button>
        <label className="anim-speed">
          Speed
          <input
            type="range"
            min="0.25"
            max="4"
            step="0.25"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
          />
          <span>{speed.toFixed(2)}×</span>
        </label>
        <div className="anim-progress">
          <div className="anim-progress-bar" style={{ width: `${progress}%` }} />
        </div>
        <span className="anim-progress-label">
          {phase === 'lineart' ? 'Tracing line art…' : `${progress}%`}
        </span>
      </div>
      <div className="coloring-animation-stage">
        <canvas ref={canvasRef} className="coloring-animation-canvas" />
        {!ready && <div className="anim-loading">Preparing animation…</div>}
      </div>
    </div>
  );
}

export default ColoringAnimation;
