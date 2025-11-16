/**
 * WebGL-based stained glass effect using GPU-accelerated shaders.
 * Provides realistic glass texture, lighting, and refraction effects.
 */

export class StainedGlassEffect {
  constructor() {
    this.canvas = null;
    this.gl = null;
    this.program = null;
    this.texture = null;
    this.initialized = false;
  }

  /**
   * Initialize WebGL context and shaders
   */
  init(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    
    if (!this.gl) {
      console.warn('WebGL not supported, falling back to canvas 2D');
      return false;
    }

    // Create shader program
    const vertexShader = this.createShader(this.gl.VERTEX_SHADER, this.getVertexShaderSource());
    const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, this.getFragmentShaderSource());
    
    if (!vertexShader || !fragmentShader) {
      return false;
    }

    this.program = this.createProgram(vertexShader, fragmentShader);
    if (!this.program) {
      return false;
    }

    this.initialized = true;
    return true;
  }

  /**
   * Apply stained glass effect to an image
   */
  applyEffect(image, intensity = 0.9) {
    if (!this.initialized) {
      if (!this.canvas) {
        this.canvas = this.createCanvas();
      }
      if (!this.init(this.canvas)) {
        // Fallback to canvas 2D if WebGL not available
        console.log('WebGL initialization failed, using Canvas 2D');
        return this.applyEffect2D(image, intensity);
      }
    }

    const gl = this.gl;
    const canvas = this.canvas;
    
    try {
      // Set canvas size
      canvas.width = image.width;
      canvas.height = image.height;
      gl.viewport(0, 0, canvas.width, canvas.height);

      // Clear canvas
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Create texture from image
      const texture = this.createTexture(gl, image);
      
      // Setup geometry (full-screen quad)
      const positionBuffer = this.createQuad(gl);
      
      // Use shader program
      gl.useProgram(this.program);
      
      // Set attributes and uniforms
      const positionLocation = gl.getAttribLocation(this.program, 'a_position');
      const resolutionLocation = gl.getUniformLocation(this.program, 'u_resolution');
      const textureLocation = gl.getUniformLocation(this.program, 'u_texture');
      const intensityLocation = gl.getUniformLocation(this.program, 'u_intensity');
      const timeLocation = gl.getUniformLocation(this.program, 'u_time');
      
      if (positionLocation === -1 || resolutionLocation === null || textureLocation === null) {
        throw new Error('Shader uniform/attribute location failed');
      }
      
      // Setup attributes
      gl.enableVertexAttribArray(positionLocation);
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
      
      // Set uniforms
      gl.uniform2f(resolutionLocation, canvas.width, canvas.height);
      gl.uniform1i(textureLocation, 0);
      gl.uniform1f(intensityLocation, intensity);
      gl.uniform1f(timeLocation, Date.now() * 0.001);
      
      // Bind texture
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      
      // Enable blending for transparency
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      
      // Draw
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      
      // Check for errors
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
        console.warn('WebGL error:', error);
        throw new Error(`WebGL error: ${error}`);
      }
      
      // Cleanup
      gl.deleteTexture(texture);
      gl.deleteBuffer(positionBuffer);
      
      return canvas;
    } catch (error) {
      console.error('WebGL effect error:', error);
      // Fallback to Canvas 2D
      return this.applyEffect2D(image, intensity);
    }
  }

  /**
   * Advanced frosted glass effect using Canvas 2D with multiple techniques
   */
  applyEffect2D(image, intensity = 0.9) {
    console.log('Applying advanced frosted glass effect...', { width: image.width, height: image.height, intensity });
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext('2d');
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Step 1: Create base with multi-pass blur for better quality
    const blurredCanvas = document.createElement('canvas');
    blurredCanvas.width = width;
    blurredCanvas.height = height;
    const blurredCtx = blurredCanvas.getContext('2d');
    blurredCtx.drawImage(image, 0, 0);
    
    // Multi-pass blur (better quality than single pass)
    const blurAmount = 1.5 + intensity * 4.0;
    for (let i = 0; i < 2; i++) {
      blurredCtx.filter = `blur(${blurAmount * 0.6}px)`;
      blurredCtx.drawImage(blurredCanvas, 0, 0);
    }
    blurredCtx.filter = 'none';
    
    // Step 2: Draw blurred image to main canvas
    ctx.drawImage(blurredCanvas, 0, 0);
    
    // Step 3: Create advanced frosted texture with Perlin-like noise
    const textureSize = 128;
    const textureCanvas = document.createElement('canvas');
    textureCanvas.width = textureSize;
    textureCanvas.height = textureSize;
    const textureCtx = textureCanvas.getContext('2d');
    
    // Create multi-octave noise pattern (simulates etched glass)
    const noiseData = textureCtx.createImageData(textureSize, textureSize);
    const noise = noiseData.data;
    
    for (let y = 0; y < textureSize; y++) {
      for (let x = 0; x < textureSize; x++) {
        let value = 0.0;
        let amplitude = 0.5;
        let frequency = 1.0;
        
        // Multi-octave noise
        for (let octave = 0; octave < 4; octave++) {
          const nx = (x / textureSize) * frequency;
          const ny = (y / textureSize) * frequency;
          value += amplitude * this.simplexNoise(nx, ny);
          amplitude *= 0.5;
          frequency *= 2.0;
        }
        
        value = (value + 1.0) * 0.5; // Normalize to 0-1
        const gray = Math.floor(value * 50 + 200); // Light gray-white
        const idx = (y * textureSize + x) * 4;
        noise[idx] = gray;
        noise[idx + 1] = gray;
        noise[idx + 2] = gray;
        noise[idx + 3] = Math.floor(200 * intensity);
      }
    }
    
    textureCtx.putImageData(noiseData, 0, 0);
    
    // Blur texture slightly
    textureCtx.filter = `blur(${1 + intensity}px)`;
    textureCtx.drawImage(textureCanvas, 0, 0);
    textureCtx.filter = 'none';
    
    // Overlay texture
    ctx.globalCompositeOperation = 'overlay';
    ctx.globalAlpha = 0.3 * intensity;
    const pattern = ctx.createPattern(textureCanvas, 'repeat');
    if (pattern) {
      ctx.fillStyle = pattern;
      ctx.fillRect(0, 0, width, height);
    }
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1.0;
    
    // Step 4: Add subtle light diffusion (soft glow)
    const glowCanvas = document.createElement('canvas');
    glowCanvas.width = width;
    glowCanvas.height = height;
    const glowCtx = glowCanvas.getContext('2d');
    glowCtx.drawImage(canvas, 0, 0);
    
    // Create soft glow
    glowCtx.filter = `blur(${6 + intensity * 4}px) brightness(${1.15 + intensity * 0.2})`;
    glowCtx.drawImage(glowCanvas, 0, 0);
    glowCtx.filter = 'none';
    
    // Blend glow
    ctx.globalCompositeOperation = 'soft-light';
    ctx.globalAlpha = 0.25 * intensity;
    ctx.drawImage(glowCanvas, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1.0;
    
    // Step 5: Add subtle directional lighting
    const lightingGradient = ctx.createLinearGradient(0, 0, 0, height);
    lightingGradient.addColorStop(0, `rgba(255, 255, 255, ${0.15 * intensity})`);
    lightingGradient.addColorStop(0.4, `rgba(255, 255, 255, ${0.05 * intensity})`);
    lightingGradient.addColorStop(0.8, 'rgba(0, 0, 0, 0)');
    lightingGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = lightingGradient;
    ctx.globalCompositeOperation = 'overlay';
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = 'source-over';
    
    // Step 6: Final color adjustments
    const workCanvas = document.createElement('canvas');
    workCanvas.width = width;
    workCanvas.height = height;
    const workCtx = workCanvas.getContext('2d');
    workCtx.drawImage(canvas, 0, 0);
    
    // Subtle color adjustments (frosted glass characteristics)
    ctx.filter = `brightness(${1.06 + intensity * 0.1}) saturate(${0.9 - intensity * 0.08}) contrast(${0.97})`;
    ctx.drawImage(workCanvas, 0, 0);
    ctx.filter = 'none';
    
    console.log('Advanced frosted glass effect complete');
    return canvas;
  }
  
  // Simple Simplex noise implementation for texture generation
  simplexNoise(x, y) {
    // Simplified 2D noise function
    const n = Math.floor(x) + Math.floor(y) * 57;
    const n2 = (n << 13) ^ n;
    return (1.0 - ((n2 * (n2 * n2 * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0);
  }

  createCanvas() {
    const canvas = document.createElement('canvas');
    return canvas;
  }

  createShader(type, source) {
    const shader = this.gl.createShader(type);
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error('Shader compilation error:', this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }
    
    return shader;
  }

  createProgram(vertexShader, fragmentShader) {
    const program = this.gl.createProgram();
    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);
    this.gl.linkProgram(program);
    
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      console.error('Program linking error:', this.gl.getProgramInfoLog(program));
      this.gl.deleteProgram(program);
      return null;
    }
    
    return program;
  }

  createTexture(gl, image) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return texture;
  }

  createQuad(gl) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -1, -1,
         1, -1,
        -1,  1,
        -1,  1,
         1, -1,
         1,  1,
      ]),
      gl.STATIC_DRAW
    );
    return buffer;
  }

  getVertexShaderSource() {
    return `
      attribute vec2 a_position;
      varying vec2 v_texCoord;
      
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = (a_position + 1.0) * 0.5;
        v_texCoord.y = 1.0 - v_texCoord.y; // Flip Y coordinate
      }
    `;
  }

  getFragmentShaderSource() {
    return `
      precision mediump float;
      
      uniform vec2 u_resolution;
      uniform sampler2D u_texture;
      uniform float u_intensity;
      uniform float u_time;
      
      varying vec2 v_texCoord;
      
      // Advanced Perlin-like noise for realistic glass texture
      vec3 mod289(vec3 x) {
        return x - floor(x * (1.0 / 289.0)) * 289.0;
      }
      
      vec4 mod289(vec4 x) {
        return x - floor(x * (1.0 / 289.0)) * 289.0;
      }
      
      vec4 permute(vec4 x) {
        return mod289(((x*34.0)+1.0)*x);
      }
      
      vec4 taylorInvSqrt(vec4 r) {
        return 1.79284291400159 - 0.85373472095314 * r;
      }
      
      // Perlin noise function
      float snoise(vec3 v) {
        const vec2 C = vec2(1.0/6.0, 1.0/3.0);
        const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
        
        vec3 i = floor(v + dot(v, C.yyy));
        vec3 x0 = v - i + dot(i, C.xxx);
        
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min(g.xyz, l.zxy);
        vec3 i2 = max(g.xyz, l.zxy);
        
        vec3 x1 = x0 - i1 + C.xxx;
        vec3 x2 = x0 - i2 + C.yyy;
        vec3 x3 = x0 - D.yyy;
        
        i = mod289(i);
        vec4 p = permute(permute(permute(
          i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4(0.0, i1.x, i2.x, 1.0));
        
        float n_ = 0.142857142857;
        vec3 ns = n_ * D.wyz - D.xzx;
        
        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
        
        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_);
        
        vec4 x = x_ *ns.x + ns.yyyy;
        vec4 y = y_ *ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);
        
        vec4 b0 = vec4(x.xy, y.xy);
        vec4 b1 = vec4(x.zw, y.zw);
        
        vec4 s0 = floor(b0)*2.0 + 1.0;
        vec4 s1 = floor(b1)*2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));
        
        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
        
        vec3 p0 = vec3(a0.xy, h.x);
        vec3 p1 = vec3(a0.zw, h.y);
        vec3 p2 = vec3(a1.xy, h.z);
        vec3 p3 = vec3(a1.zw, h.w);
        
        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;
        
        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
      }
      
      // Multi-octave Perlin noise for organic glass texture
      float glassNoise(vec2 uv) {
        vec3 coord = vec3(uv * 8.0, u_time * 0.1);
        float value = 0.0;
        float amplitude = 0.5;
        float frequency = 1.0;
        
        for (int i = 0; i < 4; i++) {
          value += amplitude * snoise(coord * frequency);
          amplitude *= 0.5;
          frequency *= 2.0;
        }
        
        return value * 0.5 + 0.5;
      }
      
      // Calculate surface normal from noise (for refraction)
      vec2 getNormal(vec2 uv) {
        float eps = 0.01;
        float hL = glassNoise(uv - vec2(eps, 0.0));
        float hR = glassNoise(uv + vec2(eps, 0.0));
        float hD = glassNoise(uv - vec2(0.0, eps));
        float hU = glassNoise(uv + vec2(0.0, eps));
        return normalize(vec2(hL - hR, hD - hU));
      }
      
      void main() {
        vec2 uv = v_texCoord;
        
        // Simplified frosted glass: blur + texture + subtle effects
        
        // Multi-tap Gaussian blur (core frosted glass effect)
        float blurRadius = (1.0 + u_intensity * 2.5) / u_resolution.x;
        vec4 blurredColor = vec4(0.0);
        float totalWeight = 0.0;
        
        // 5x5 Gaussian kernel for smooth blur
        for (int x = -2; x <= 2; x++) {
          for (int y = -2; y <= 2; y++) {
            vec2 offset = vec2(float(x), float(y)) * blurRadius;
            float dist = length(vec2(x, y));
            float weight = exp(-dist * dist / 2.0);
            blurredColor += texture2D(u_texture, uv + offset) * weight;
            totalWeight += weight;
          }
        }
        blurredColor /= totalWeight;
        
        // Mix original with blurred (frosted glass diffuses light)
        vec4 color = mix(texture2D(u_texture, uv), blurredColor, 0.3 + u_intensity * 0.5);
        
        // Add subtle frosted texture (etched glass surface)
        float frostPattern = glassNoise(uv * 20.0);
        vec3 surfaceTexture = vec3(0.96 + frostPattern * 0.04);
        color.rgb *= surfaceTexture;
        
        // Subtle light diffusion (soft glow)
        float glow = glassNoise(uv * 4.0) * 0.05 * u_intensity;
        color.rgb += vec3(glow);
        
        // Reduce saturation slightly (frosted glass mutes colors)
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        color.rgb = mix(color.rgb, vec3(gray), 0.08 * u_intensity);
        
        // Brightness boost (light transmission through glass)
        color.rgb *= 1.05 + u_intensity * 0.1;
        
        // Soften contrast (frosted glass has softer edges)
        color.rgb = pow(color.rgb, vec3(0.97));
        
        gl_FragColor = color;
      }
    `;
  }
}

/**
 * Apply stained glass effect to an image element or URL
 */
export function applyStainedGlassEffect(imageSource, intensity = 0.9) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      try {
        console.log('Image loaded, applying stained glass effect...', {
          width: img.width,
          height: img.height,
          intensity
        });
        
        const effect = new StainedGlassEffect();
        const canvas = effect.createCanvas();
        
        // Try WebGL first
        if (effect.init(canvas)) {
          console.log('Using WebGL shader effect');
          try {
            const resultCanvas = effect.applyEffect(img, intensity);
            const dataUrl = resultCanvas.toDataURL('image/png');
            console.log('WebGL effect applied successfully');
            resolve(dataUrl);
            return;
          } catch (webglError) {
            console.warn('WebGL effect failed, falling back to Canvas 2D:', webglError);
          }
        } else {
          console.log('WebGL not available, using Canvas 2D fallback');
        }
        
        // Fallback to Canvas 2D (more robust and noticeable)
        const resultCanvas = effect.applyEffect2D(img, intensity);
        const dataUrl = resultCanvas.toDataURL('image/png');
        console.log('Canvas 2D effect applied successfully');
        resolve(dataUrl);
      } catch (error) {
        console.error('Error applying stained glass effect:', error);
        // Even if effect fails, return original image
        resolve(imageSource);
      }
    };
    
    img.onerror = (error) => {
      console.error('Failed to load image for stained glass effect:', error);
      // Return original image if loading fails
      resolve(imageSource);
    };
    
    // Handle data URLs and regular URLs
    if (imageSource.startsWith('data:')) {
      img.src = imageSource;
    } else {
      img.src = imageSource;
    }
  });
}

