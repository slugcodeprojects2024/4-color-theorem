/**
 * AI Worker - Runs Transformers.js models in a separate thread
 * This keeps the main UI responsive during AI processing
 */

/* eslint-disable no-restricted-globals */
import { pipeline, env } from '@huggingface/transformers';

// Configure for browser
env.allowLocalModels = false;
env.useBrowserCache = true;

let classifier = null;

// Coloring book specific labels for CLIP classification
const COLORING_LABELS = [
  // Subjects
  "a coloring page of flowers",
  "a coloring page of animals",
  "a coloring page of a lion",
  "a coloring page of a bear",
  "a coloring page of a butterfly",
  "a coloring page of birds",
  "a coloring page of fish or ocean life",
  "a coloring page of a dragon",
  "a coloring page of a unicorn",
  "a coloring page of a castle",
  "a coloring page of nature and trees",
  "a coloring page of mountains",
  "a coloring page of the sun and sky",
  "a coloring page of mushrooms",
  "a coloring page of food or fruits",
  "a coloring page of vehicles",
  "a coloring page of buildings or houses",
  "a coloring page of people or characters",
  "a coloring page of abstract patterns",
  "a mandala coloring page",
  "a geometric pattern coloring page",
  "a zentangle pattern",
  // Styles
  "a simple coloring page for children",
  "a detailed coloring page for adults",
  "a kawaii cute style coloring page",
];

// Subject to color mapping
const SUBJECT_COLORS = {
  "flowers": {
    primary: [[255, 105, 180], [255, 182, 193], [255, 20, 147], [238, 130, 238]],
    secondary: [[34, 139, 34], [50, 205, 50], [255, 215, 0]],
    name: "Floral Garden"
  },
  "lion": {
    primary: [[255, 165, 0], [255, 140, 0], [218, 165, 32], [255, 215, 0]],
    secondary: [[255, 218, 185], [139, 90, 43]],
    name: "Safari Gold"
  },
  "bear": {
    primary: [[139, 90, 43], [160, 82, 45], [101, 67, 33], [205, 133, 63]],
    secondary: [[34, 139, 34], [135, 206, 235]],
    name: "Forest Brown"
  },
  "butterfly": {
    primary: [[138, 43, 226], [255, 105, 180], [255, 20, 147], [148, 0, 211]],
    secondary: [[0, 255, 127], [255, 215, 0]],
    name: "Butterfly Wings"
  },
  "dragon": {
    primary: [[178, 34, 34], [255, 69, 0], [255, 140, 0], [139, 0, 0]],
    secondary: [[255, 215, 0], [0, 100, 0]],
    name: "Dragon Fire"
  },
  "unicorn": {
    primary: [[255, 182, 193], [221, 160, 221], [230, 230, 250], [255, 105, 180]],
    secondary: [[255, 215, 0], [135, 206, 235]],
    name: "Magical Unicorn"
  },
  "ocean": {
    primary: [[0, 105, 148], [72, 202, 228], [64, 224, 208], [0, 128, 128]],
    secondary: [[255, 127, 80], [255, 215, 0]],
    name: "Ocean Deep"
  },
  "nature": {
    primary: [[34, 139, 34], [85, 107, 47], [0, 128, 0], [107, 142, 35]],
    secondary: [[135, 206, 235], [255, 215, 0], [139, 90, 43]],
    name: "Nature Walk"
  },
  "mandala": {
    primary: [[128, 0, 128], [0, 128, 128], [178, 34, 34], [255, 215, 0]],
    secondary: [[138, 43, 226], [0, 206, 209]],
    name: "Mandala Harmony"
  },
  "geometric": {
    primary: [[220, 20, 60], [0, 0, 139], [255, 215, 0], [34, 139, 34]],
    secondary: [[255, 255, 255], [0, 0, 0]],
    name: "Geometric Bold"
  },
  "kawaii": {
    primary: [[255, 182, 193], [255, 218, 185], [176, 224, 230], [255, 160, 122]],
    secondary: [[255, 105, 180], [221, 160, 221]],
    name: "Kawaii Cute"
  },
  "castle": {
    primary: [[128, 128, 128], [169, 169, 169], [105, 105, 105], [192, 192, 192]],
    secondary: [[70, 130, 180], [255, 215, 0], [139, 69, 19]],
    name: "Castle Stone"
  },
  "mushrooms": {
    primary: [[255, 0, 0], [220, 20, 60], [255, 99, 71], [255, 255, 255]],
    secondary: [[139, 90, 43], [34, 139, 34]],
    name: "Mushroom Magic"
  },
  "default": {
    primary: [[135, 206, 235], [34, 139, 34], [255, 215, 0], [255, 105, 180]],
    secondary: [[139, 90, 43], [128, 0, 128]],
    name: "Universal Palette"
  }
};

/**
 * Initialize the CLIP classifier
 */
async function initClassifier(onProgress) {
  if (!classifier) {
    classifier = await pipeline(
      'zero-shot-image-classification',
      'Xenova/clip-vit-base-patch32',
      { 
        progress_callback: onProgress,
        quantized: true  // Use quantized model for faster loading
      }
    );
  }
  return classifier;
}

/**
 * Classify image against coloring book labels
 */
async function classifyImage(imageData, onProgress) {
  const model = await initClassifier(onProgress);
  
  // Run classification
  const results = await model(imageData, COLORING_LABELS);
  
  // Sort by score
  results.sort((a, b) => b.score - a.score);
  
  return results;
}

/**
 * Extract subject from classification label
 */
function extractSubject(label) {
  const subjectMap = {
    "flowers": "flowers",
    "lion": "lion",
    "bear": "bear",
    "butterfly": "butterfly",
    "dragon": "dragon",
    "unicorn": "unicorn",
    "fish": "ocean",
    "ocean": "ocean",
    "nature": "nature",
    "trees": "nature",
    "mountains": "nature",
    "mandala": "mandala",
    "geometric": "geometric",
    "zentangle": "geometric",
    "kawaii": "kawaii",
    "cute": "kawaii",
    "castle": "castle",
    "mushroom": "mushrooms",
    "simple": "simple",
    "children": "simple",
    "detailed": "detailed",
    "adults": "detailed",
  };
  
  const lowerLabel = label.toLowerCase();
  for (const [key, subject] of Object.entries(subjectMap)) {
    if (lowerLabel.includes(key)) {
      return subject;
    }
  }
  return "default";
}

/**
 * Generate palettes based on detected subjects
 */
function generatePalettes(classificationResults) {
  const palettes = [];
  const usedSubjects = new Set();
  
  // Get top 3 classifications
  const topResults = classificationResults.slice(0, 5);
  
  for (const result of topResults) {
    const subject = extractSubject(result.label);
    
    if (usedSubjects.has(subject)) continue;
    usedSubjects.add(subject);
    
    const colors = SUBJECT_COLORS[subject] || SUBJECT_COLORS.default;
    
    // Create main palette
    palettes.push({
      name: colors.name,
      colors: colors.primary.slice(0, 4),
      description: `AI detected: ${result.label.replace("a coloring page of ", "")}`,
      score: result.score,
      source: "ai_classification",
      is_smart: true,
      detected_label: result.label
    });
    
    // Create variation with secondary colors
    if (colors.secondary.length >= 2 && palettes.length < 4) {
      const mixedColors = [
        colors.primary[0],
        colors.primary[1],
        colors.secondary[0],
        colors.secondary[1] || colors.primary[2]
      ];
      
      palettes.push({
        name: `${colors.name} Variation`,
        colors: mixedColors,
        description: `Alternative palette for ${subject}`,
        score: result.score * 0.9,
        source: "ai_classification",
        is_smart: true
      });
    }
  }
  
  return palettes;
}

/**
 * Main message handler
 */
self.onmessage = async function(e) {
  const { type, imageData, id } = e.data;
  
  try {
    if (type === 'analyze') {
      // Send progress updates
      const onProgress = (progress) => {
        self.postMessage({
          type: 'progress',
          id,
          progress: progress
        });
      };
      
      // Classify image
      self.postMessage({ type: 'status', id, status: 'Analyzing image...' });
      const results = await classifyImage(imageData, onProgress);
      
      // Generate palettes
      self.postMessage({ type: 'status', id, status: 'Generating palettes...' });
      const palettes = generatePalettes(results);
      
      // Extract detected info
      const topResult = results[0];
      const detectedSubject = extractSubject(topResult.label);
      
      // Determine style
      let style = "unknown";
      for (const result of results.slice(0, 3)) {
        if (result.label.includes("simple") || result.label.includes("children")) {
          style = "children_simple";
          break;
        } else if (result.label.includes("detailed") || result.label.includes("adults")) {
          style = "adult_detailed";
          break;
        } else if (result.label.includes("mandala")) {
          style = "mandala";
          break;
        } else if (result.label.includes("kawaii") || result.label.includes("cute")) {
          style = "kawaii";
          break;
        } else if (result.label.includes("geometric") || result.label.includes("zentangle")) {
          style = "geometric";
          break;
        }
      }
      
      self.postMessage({
        type: 'result',
        id,
        result: {
          layer: "browser_ai",
          model: "clip-vit-base-patch32",
          classifications: results.slice(0, 5).map(r => ({
            label: r.label,
            score: r.score
          })),
          detected_subject: detectedSubject,
          detected_style: style,
          confidence: topResult.score,
          suggested_palettes: palettes
        }
      });
      
    } else if (type === 'check') {
      // Check if model is loaded
      self.postMessage({
        type: 'status',
        id,
        loaded: classifier !== null
      });
    }
    
  } catch (error) {
    self.postMessage({
      type: 'error',
      id,
      error: error.message
    });
  }
};

