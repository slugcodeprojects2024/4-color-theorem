/**
 * Optimized Stained Glass Effect
 * Creates authentic stained glass appearance with prominent lead lines
 */

export async function applyStainedGlassEffect(imageSource, intensity = 0.8) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      try {
        console.log('Applying optimized stained glass effect...', {
          width: img.width,
          height: img.height,
          intensity
        });
        
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
          throw new Error('Could not get 2D context');
        }
        
        // Step 1: Draw the colored image
        ctx.drawImage(img, 0, 0);
        
        // Step 2: Extract and enhance edges for lead lines
        const leadLines = createLeadLines(canvas, canvas.width, canvas.height, intensity);
        
        // Step 3: Apply glass texture
        applyGlassTexture(ctx, canvas.width, canvas.height, intensity);
        
        // Step 4: Overlay lead lines prominently
        ctx.globalCompositeOperation = 'multiply';
        ctx.drawImage(leadLines, 0, 0);
        ctx.globalCompositeOperation = 'source-over';
        
        // Step 5: Final glass effects
        applyFinalGlassEffects(ctx, canvas.width, canvas.height, intensity);
        
        const dataUrl = canvas.toDataURL('image/png');
        console.log('Stained glass effect complete, data URL length:', dataUrl.length);
        resolve(dataUrl);
      } catch (error) {
        console.error('Stained glass effect error:', error);
        console.error('Error stack:', error.stack);
        resolve(imageSource);
      }
    };
    
    img.onerror = () => {
      console.error('Failed to load image');
      resolve(imageSource);
    };
    
    img.src = imageSource;
  });
}

function createLeadLines(canvas, width, height, intensity) {
  // Get image data for edge detection from the canvas
  const tempCtx = canvas.getContext('2d');
  if (!tempCtx) {
    throw new Error('Could not get 2D context for lead lines');
  }
  const imageData = tempCtx.getImageData(0, 0, width, height);
  const data = imageData.data;
  
  // Create canvas for lead lines
  const leadCanvas = document.createElement('canvas');
  leadCanvas.width = width;
  leadCanvas.height = height;
  const leadCtx = leadCanvas.getContext('2d');
  
  // Detect edges using Sobel operator
  // For performance, sample every Nth pixel for large images
  const sampleRate = width * height > 1000000 ? 2 : 1;
  const edges = new Uint8Array(width * height);
  
  console.log('Detecting edges for lead lines...', { width, height, sampleRate });
  
  for (let y = 1; y < height - 1; y += sampleRate) {
    for (let x = 1; x < width - 1; x += sampleRate) {
      // Get surrounding pixels
      const tl = ((y-1) * width + (x-1)) * 4;
      const tm = ((y-1) * width + x) * 4;
      const tr = ((y-1) * width + (x+1)) * 4;
      const ml = (y * width + (x-1)) * 4;
      const mr = (y * width + (x+1)) * 4;
      const bl = ((y+1) * width + (x-1)) * 4;
      const bm = ((y+1) * width + x) * 4;
      const br = ((y+1) * width + (x+1)) * 4;
      
      // Calculate gradients (Sobel)
      const gx = (
        -1 * getBrightness(data, tl) + 1 * getBrightness(data, tr) +
        -2 * getBrightness(data, ml) + 2 * getBrightness(data, mr) +
        -1 * getBrightness(data, bl) + 1 * getBrightness(data, br)
      );
      
      const gy = (
        -1 * getBrightness(data, tl) + 1 * getBrightness(data, bl) +
        -2 * getBrightness(data, tm) + 2 * getBrightness(data, bm) +
        -1 * getBrightness(data, tr) + 1 * getBrightness(data, br)
      );
      
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      const edgeValue = Math.min(magnitude, 255);
      edges[y * width + x] = edgeValue;
      
      // Fill in sampled pixels if using sampleRate > 1
      if (sampleRate > 1) {
        for (let dy = 0; dy < sampleRate && (y + dy) < height; dy++) {
          for (let dx = 0; dx < sampleRate && (x + dx) < width; dx++) {
            if (y + dy < height - 1 && x + dx < width - 1) {
              edges[(y + dy) * width + (x + dx)] = edgeValue;
            }
          }
        }
      }
    }
  }
  
  console.log('Edge detection complete');
  
  // Create thick, dark lead lines from edges
  const leadImageData = leadCtx.createImageData(width, height);
  const leadData = leadImageData.data;
  
  // First pass: Create base lead lines
  for (let i = 0; i < edges.length; i++) {
    const idx = i * 4;
    if (edges[i] > 30) { // Threshold for edge detection
      const alpha = Math.min(edges[i] * 2 * intensity, 255);
      leadData[idx] = 15;      // Very dark gray
      leadData[idx + 1] = 15;
      leadData[idx + 2] = 20;  // Slight blue tint
      leadData[idx + 3] = alpha;
    }
  }
  
  leadCtx.putImageData(leadImageData, 0, 0);
  
  // Make lead lines thicker
  leadCtx.filter = 'blur(1px)';
  leadCtx.drawImage(leadCanvas, 0, 0);
  leadCtx.filter = 'none';
  
  // Enhance lead lines with multiple passes
  leadCtx.globalCompositeOperation = 'multiply';
  leadCtx.globalAlpha = 0.7;
  leadCtx.drawImage(leadCanvas, 0, 0);
  leadCtx.drawImage(leadCanvas, 1, 0);
  leadCtx.drawImage(leadCanvas, 0, 1);
  leadCtx.drawImage(leadCanvas, -1, 0);
  leadCtx.drawImage(leadCanvas, 0, -1);
  leadCtx.globalCompositeOperation = 'source-over';
  leadCtx.globalAlpha = 1.0;
  
  // Add metallic shine to lead
  const shineGradient = leadCtx.createLinearGradient(0, 0, width, height);
  shineGradient.addColorStop(0, 'rgba(40, 40, 50, 0.3)');
  shineGradient.addColorStop(0.5, 'rgba(30, 30, 40, 0.1)');
  shineGradient.addColorStop(1, 'rgba(20, 20, 30, 0.3)');
  
  leadCtx.fillStyle = shineGradient;
  leadCtx.globalCompositeOperation = 'overlay';
  leadCtx.fillRect(0, 0, width, height);
  leadCtx.globalCompositeOperation = 'source-over';
  
  return leadCanvas;
}

function getBrightness(data, idx) {
  return (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
}

function applyGlassTexture(ctx, width, height, intensity) {
  // Save current state
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = width;
  tempCanvas.height = height;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(ctx.canvas, 0, 0);
  
  // Create glass texture overlay
  const textureCanvas = document.createElement('canvas');
  textureCanvas.width = width;
  textureCanvas.height = height;
  const textureCtx = textureCanvas.getContext('2d');
  
  // Create noise pattern for glass texture
  const imageData = textureCtx.createImageData(width, height);
  const data = imageData.data;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      
      // Multi-scale noise for realistic glass
      const noise1 = Math.sin(x * 0.01) * Math.cos(y * 0.01) * 30;
      const noise2 = Math.sin(x * 0.03 + y * 0.02) * 20;
      const noise3 = Math.random() * 10 - 5;
      
      const value = 128 + noise1 + noise2 + noise3;
      
      data[idx] = value;
      data[idx + 1] = value;
      data[idx + 2] = value + 10; // Slight blue tint
      data[idx + 3] = 40 * intensity;
    }
  }
  
  textureCtx.putImageData(imageData, 0, 0);
  
  // Apply texture
  ctx.globalCompositeOperation = 'overlay';
  ctx.globalAlpha = 0.5;
  ctx.drawImage(textureCanvas, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1.0;
  
  // Add color richness
  ctx.globalCompositeOperation = 'overlay';
  ctx.globalAlpha = intensity * 0.2;
  ctx.drawImage(tempCanvas, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1.0;
}

function applyFinalGlassEffects(ctx, width, height, intensity) {
  // Add lighting effects
  const lightGradient = ctx.createLinearGradient(0, 0, width * 0.7, height * 0.7);
  lightGradient.addColorStop(0, `rgba(255, 250, 200, ${0.4 * intensity})`);
  lightGradient.addColorStop(0.5, `rgba(255, 255, 255, ${0.2 * intensity})`);
  lightGradient.addColorStop(1, `rgba(200, 220, 255, ${0.3 * intensity})`);
  
  ctx.fillStyle = lightGradient;
  ctx.globalCompositeOperation = 'soft-light';
  ctx.fillRect(0, 0, width, height);
  
  // Add specular highlights
  for (let i = 0; i < 3; i++) {
    const x = Math.random() * width;
    const y = Math.random() * height;
    const radius = 30 + Math.random() * 70;
    
    const highlight = ctx.createRadialGradient(x, y, 0, x, y, radius);
    highlight.addColorStop(0, `rgba(255, 255, 255, ${0.4 * intensity})`);
    highlight.addColorStop(0.5, `rgba(255, 250, 230, ${0.2 * intensity})`);
    highlight.addColorStop(1, 'rgba(255, 255, 255, 0)');
    
    ctx.fillStyle = highlight;
    ctx.globalCompositeOperation = 'screen';
    ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
  }
  
  // Add vignette
  const vignette = ctx.createRadialGradient(
    width / 2, height / 2, Math.min(width, height) * 0.3,
    width / 2, height / 2, Math.max(width, height) * 0.8
  );
  vignette.addColorStop(0, 'rgba(0, 0, 0, 0)');
  vignette.addColorStop(1, `rgba(0, 0, 0, ${0.4 * intensity})`);
  
  ctx.fillStyle = vignette;
  ctx.globalCompositeOperation = 'multiply';
  ctx.fillRect(0, 0, width, height);
  
  // Final adjustments
  ctx.globalCompositeOperation = 'source-over';
  const finalCanvas = document.createElement('canvas');
  finalCanvas.width = width;
  finalCanvas.height = height;
  const finalCtx = finalCanvas.getContext('2d');
  finalCtx.drawImage(ctx.canvas, 0, 0);
  
  // Boost contrast and saturation
  ctx.filter = `contrast(${1.1 + intensity * 0.2}) saturate(${1.2 + intensity * 0.3}) brightness(${1.05})`;
  ctx.drawImage(finalCanvas, 0, 0);
  ctx.filter = 'none';
}

// Export simplified version for testing
export function applySimpleStainedGlass(imageSource, intensity = 0.8) {
  return applyStainedGlassEffect(imageSource, intensity);
}
