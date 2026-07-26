import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import './StainedGlassViewer.css';

/**
 * WebGL stained glass viewer (three.js).
 *
 * Renders the flat-colored result through a custom shader that simulates
 * real glass: per-pane surface normals, refraction wobble, beveled lead
 * came with specular rims, glass grain, and a light source that drifts
 * on its own and follows the pointer.
 *
 * This replaces the old canvas effect, which multiply-stamped lead lines
 * five times and turned any densely detailed image almost entirely black.
 */

const FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;   // flat colored result
  uniform sampler2D uEdges;     // edge/lead intensity map (r channel)
  uniform vec2  uResolution;
  uniform float uTime;
  uniform vec2  uLight;         // light position in uv space
  uniform float uIntensity;

  // --- hash noise -------------------------------------------------------
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }
  float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
               u.y);
  }
  float fbm(vec2 p) {
    float v = 0.0; float a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.1; a *= 0.5; }
    return v;
  }

  void main() {
    vec2 px = 1.0 / uResolution;

    // --- surface normal from glass grain + edge bevel -------------------
    float grain = fbm(vUv * 40.0) * 0.5 + fbm(vUv * 160.0 + 7.3) * 0.25;
    float gx = fbm((vUv + vec2(px.x, 0.0)) * 40.0) - fbm((vUv - vec2(px.x, 0.0)) * 40.0);
    float gy = fbm((vUv + vec2(0.0, px.y)) * 40.0) - fbm((vUv - vec2(0.0, px.y)) * 40.0);

    float eC = texture2D(uEdges, vUv).r;
    float eR = texture2D(uEdges, vUv + vec2(px.x * 2.0, 0.0)).r;
    float eL = texture2D(uEdges, vUv - vec2(px.x * 2.0, 0.0)).r;
    float eU = texture2D(uEdges, vUv + vec2(0.0, px.y * 2.0)).r;
    float eD = texture2D(uEdges, vUv - vec2(0.0, px.y * 2.0)).r;

    // Bevel normal: panes dip toward the lead lines
    vec2 bevel = vec2(eR - eL, eU - eD) * 2.5;
    vec3 N = normalize(vec3(gx * 3.0 * uIntensity + bevel.x,
                            gy * 3.0 * uIntensity + bevel.y, 1.0));

    // --- refraction: sample the color slightly offset by the normal -----
    vec2 refr = N.xy * 0.004 * uIntensity
              + vec2(sin(uTime * 0.4 + vUv.y * 8.0),
                     cos(uTime * 0.3 + vUv.x * 8.0)) * 0.0008 * uIntensity;
    vec3 base = texture2D(uTexture, clamp(vUv + refr, 0.0, 1.0)).rgb;

    // Saturate like real cathedral glass
    float lum = dot(base, vec3(0.299, 0.587, 0.114));
    base = mix(vec3(lum), base, 1.0 + 0.45 * uIntensity);

    // --- lighting --------------------------------------------------------
    vec3 L = normalize(vec3(uLight - vUv, 0.55));
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 48.0);

    // Light transmission: glass glows where the light is behind it
    float dLight = distance(vUv, uLight);
    float transmit = exp(-dLight * dLight * 3.0) * 0.55 * uIntensity;

    vec3 col = base * (0.55 + 0.55 * diff) + base * transmit;
    col += vec3(1.0, 0.98, 0.92) * spec * 0.35 * uIntensity;

    // Subtle internal streaks (hand-blown glass)
    float streak = fbm(vUv * vec2(3.0, 22.0) + uTime * 0.05);
    col *= 0.96 + streak * 0.08;

    // --- lead came ---------------------------------------------------------
    // Rounded profile: dark core, faint metallic sheen on the rim
    float lead = smoothstep(0.15, 0.75, eC);
    float rim = smoothstep(0.05, 0.35, eC) - lead;
    vec3 leadCol = vec3(0.09, 0.09, 0.115) + spec * 0.5 * vec3(0.7, 0.72, 0.8);
    col = mix(col, leadCol, lead);
    col += rim * 0.10 * vec3(0.8, 0.82, 0.9);

    // Gentle vignette
    float vig = 1.0 - 0.22 * uIntensity * smoothstep(0.4, 0.95, distance(vUv, vec2(0.5)));
    col *= vig;

    gl_FragColor = vec4(col, 1.0);
  }
`;

const VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

/** Build an edge-intensity texture from the image (Sobel on a canvas). */
function buildEdgeTexture(img, maxDim = 1024) {
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  const w = Math.max(2, Math.round(img.width * scale));
  const h = Math.max(2, Math.round(img.height * scale));
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const src = ctx.getImageData(0, 0, w, h).data;

  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const j = i * 4;
    lum[i] = (src[j] * 0.299 + src[j + 1] * 0.587 + src[j + 2] * 0.114);
  }

  const out = ctx.createImageData(w, h);
  const od = out.data;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = -lum[i - w - 1] + lum[i - w + 1]
               - 2 * lum[i - 1] + 2 * lum[i + 1]
               - lum[i + w - 1] + lum[i + w + 1];
      const gy = -lum[i - w - 1] - 2 * lum[i - w] - lum[i - w + 1]
               + lum[i + w - 1] + 2 * lum[i + w] + lum[i + w + 1];
      const m = Math.min(255, Math.sqrt(gx * gx + gy * gy));
      const j = i * 4;
      od[j] = m; od[j + 1] = m; od[j + 2] = m; od[j + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
  // Slight blur widens the lead lines smoothly (single pass, not 5 stamps)
  ctx.filter = 'blur(1px)';
  ctx.drawImage(c, 0, 0);
  ctx.filter = 'none';

  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  return tex;
}

function StainedGlassViewer({ image, intensity = 0.85 }) {
  const mountRef = useRef(null);
  const stateRef = useRef({});
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!image || !mountRef.current) return undefined;
    const mount = mountRef.current;
    let disposed = false;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (disposed) return;
      try {
        const aspect = img.width / img.height;
        const maxW = Math.min(mount.clientWidth || 640, 900);
        const w = maxW;
        const h = Math.round(w / aspect);

        const renderer = new THREE.WebGLRenderer({
          antialias: true,
          preserveDrawingBuffer: true, // enables snapshot download
        });
        renderer.setSize(w, h);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        mount.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        const camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0.1, 10);
        camera.position.z = 1;

        const texture = new THREE.Texture(img);
        texture.needsUpdate = true;
        texture.minFilter = THREE.LinearFilter;
        const edges = buildEdgeTexture(img);

        const uniforms = {
          uTexture: { value: texture },
          uEdges: { value: edges },
          uResolution: { value: new THREE.Vector2(w, h) },
          uTime: { value: 0 },
          uLight: { value: new THREE.Vector2(0.3, 0.75) },
          uIntensity: { value: intensity },
        };

        const material = new THREE.ShaderMaterial({
          uniforms, vertexShader: VERT, fragmentShader: FRAG,
        });
        const quad = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
        scene.add(quad);

        const pointer = { x: 0.3, y: 0.75, active: false };
        const onMove = (e) => {
          const rect = renderer.domElement.getBoundingClientRect();
          pointer.x = (e.clientX - rect.left) / rect.width;
          pointer.y = 1 - (e.clientY - rect.top) / rect.height;
          pointer.active = true;
        };
        const onLeave = () => { pointer.active = false; };
        renderer.domElement.addEventListener('pointermove', onMove);
        renderer.domElement.addEventListener('pointerleave', onLeave);

        const clock = new THREE.Clock();
        let raf;
        const animate = () => {
          raf = requestAnimationFrame(animate);
          const t = clock.getElapsedTime();
          uniforms.uTime.value = t;
          // Drift light on its own; ease toward the pointer when hovered
          const drift = new THREE.Vector2(
            0.5 + 0.35 * Math.sin(t * 0.25),
            0.6 + 0.25 * Math.cos(t * 0.18)
          );
          const target = pointer.active
            ? new THREE.Vector2(pointer.x, pointer.y) : drift;
          uniforms.uLight.value.lerp(target, 0.06);
          renderer.render(scene, camera);
        };
        animate();

        stateRef.current = { renderer, raf, onMove, onLeave, material, texture, edges };
      } catch (e) {
        console.error('StainedGlassViewer init failed:', e);
        setError('WebGL is not available in this browser.');
      }
    };
    img.onerror = () => setError('Failed to load image.');
    img.src = image;

    return () => {
      disposed = true;
      const s = stateRef.current;
      if (s.raf) cancelAnimationFrame(s.raf);
      if (s.renderer) {
        s.renderer.domElement.removeEventListener('pointermove', s.onMove);
        s.renderer.domElement.removeEventListener('pointerleave', s.onLeave);
        if (s.renderer.domElement.parentNode === mount) {
          mount.removeChild(s.renderer.domElement);
        }
        s.renderer.dispose();
      }
      if (s.material) s.material.dispose();
      if (s.texture) s.texture.dispose();
      if (s.edges) s.edges.dispose();
      stateRef.current = {};
    };
  }, [image, intensity]);

  const handleDownload = () => {
    const s = stateRef.current;
    if (!s.renderer) return;
    const link = document.createElement('a');
    link.download = 'stained-glass.png';
    link.href = s.renderer.domElement.toDataURL('image/png');
    link.click();
  };

  if (error) {
    return (
      <div className="stained-glass-viewer">
        <p className="sg-error">{error} Showing flat result instead.</p>
        <img src={image} alt="Colored result" style={{ maxWidth: '100%' }} />
      </div>
    );
  }

  return (
    <div className="stained-glass-viewer">
      <div className="sg-canvas-mount" ref={mountRef} />
      <div className="sg-toolbar">
        <span className="sg-hint">Move your cursor over the glass to shine light through it</span>
        <button className="sg-download" onClick={handleDownload}>
          Download snapshot
        </button>
      </div>
    </div>
  );
}

export default StainedGlassViewer;
