import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180000,
});

// =====================================================================
// Build a FormData from the common option set
// =====================================================================

function buildFormData(imageFile, options = {}) {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('style', options.style || 'vibrant');
  formData.append(
    'stained_glass',
    options.stainedGlassEnabled ? 'true' : 'false'
  );

  if (options.lineArtSettings && options.lineArtSettings.enabled) {
    formData.append('convert_to_lineart', 'true');
    formData.append(
      'line_thickness',
      options.lineArtSettings.lineThickness || 'medium'
    );
    formData.append(
      'detail_level',
      options.lineArtSettings.detailLevel || 'detailed'
    );
    formData.append(
      'contrast',
      String(options.lineArtSettings.contrast || 1.0)
    );
  } else {
    formData.append('convert_to_lineart', 'false');
  }

  formData.append('use_five_colors', 'false');

  if (
    options.customColors &&
    Array.isArray(options.customColors) &&
    options.customColors.length > 0
  ) {
    formData.append('custom_colors', JSON.stringify(options.customColors));
  }

  return formData;
}

// =====================================================================
// Original synchronous process (kept for backward compat)
// =====================================================================

export const processImage = async (
  imageFile,
  style = 'vibrant',
  stainedGlassEnabled = false,
  lineArtSettings = null,
  mlSettings = null,
  customColors = null
) => {
  const formData = buildFormData(imageFile, {
    style,
    stainedGlassEnabled,
    lineArtSettings,
    customColors,
  });

  try {
    const response = await api.post('/api/process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    if (response.data.success) {
      if (!response.data.image) {
        throw new Error('Server response missing image data');
      }
      return response.data;
    } else {
      throw new Error(response.data.error || 'Processing failed');
    }
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Server error');
    } else if (error.request) {
      throw new Error(
        'Network error - please check if the server is running'
      );
    } else {
      throw new Error(error.message || 'An error occurred');
    }
  }
};

// =====================================================================
// Streaming process with real-time progress
// =====================================================================

/**
 * Process an image with real-time progress updates via SSE.
 *
 * @param {File}     imageFile  – the image to process
 * @param {object}   options    – { style, stainedGlassEnabled, lineArtSettings, customColors }
 * @param {function} onProgress – called with (stageName: string, percent: number)
 * @returns {Promise<object>}  – same shape as processImage response
 */
export const processImageWithProgress = async (
  imageFile,
  options = {},
  onProgress
) => {
  const formData = buildFormData(imageFile, options);

  const response = await fetch(`${API_BASE_URL}/api/process-stream`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by double newlines
    let boundary;
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const message = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (message.startsWith('data: ')) {
        try {
          const data = JSON.parse(message.slice(6));

          if (data.type === 'progress' && onProgress) {
            onProgress(data.stage, data.progress);
          } else if (data.type === 'result') {
            result = data.data;
          } else if (data.type === 'error') {
            throw new Error(data.message || 'Processing failed');
          }
        } catch (parseErr) {
          // Re-throw application errors, ignore JSON parse errors
          if (parseErr.message && !parseErr.message.includes('JSON')) {
            throw parseErr;
          }
        }
      }
      // Lines starting with ":" are SSE comments / keepalives — ignore
    }
  }

  if (!result) {
    throw new Error('Stream ended without a result');
  }

  return result;
};

// =====================================================================
// Recolor (fast palette swap)
// =====================================================================

/**
 * Instantly re-render a previously processed image with a different palette.
 * Uses cached data on the server — no heavy processing.
 *
 * @param {string}        sessionId    – from a previous process response
 * @param {string}        style        – palette name (e.g. 'vibrant')
 * @param {Array|null}    customColors – [[r,g,b], ...] or null
 * @returns {Promise<object>}          – { success, image, stats, session_id }
 */
export const recolorImage = async (sessionId, style, customColors = null) => {
  const formData = new FormData();
  formData.append('session_id', sessionId);
  formData.append('style', style);
  formData.append('use_five_colors', 'false');

  if (
    customColors &&
    Array.isArray(customColors) &&
    customColors.length > 0
  ) {
    formData.append('custom_colors', JSON.stringify(customColors));
  }

  try {
    const response = await api.post('/api/recolor', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    if (response.data.success) {
      return response.data;
    } else {
      throw new Error(response.data.error || 'Recolor failed');
    }
  } catch (error) {
    if (error.response) {
      // 404 means the session expired
      if (error.response.status === 404) {
        throw new Error('SESSION_EXPIRED');
      }
      throw new Error(error.response.data.detail || 'Server error');
    } else if (error.request) {
      throw new Error(
        'Network error - please check if the server is running'
      );
    } else {
      throw new Error(error.message || 'An error occurred');
    }
  }
};

// =====================================================================
// Other endpoints (unchanged)
// =====================================================================

export const previewLineArt = async (imageFile, lineArtSettings) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append(
    'line_thickness',
    lineArtSettings.lineThickness || 'medium'
  );
  formData.append(
    'detail_level',
    lineArtSettings.detailLevel || 'detailed'
  );
  formData.append('contrast', String(lineArtSettings.contrast || 1.0));

  try {
    const response = await api.post('/api/preview-lineart', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    if (response.data.success) {
      return response.data;
    } else {
      throw new Error(response.data.error || 'Preview failed');
    }
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Server error');
    } else if (error.request) {
      throw new Error(
        'Network error - please check if the server is running'
      );
    } else {
      throw new Error(error.message || 'An error occurred');
    }
  }
};

export const previewImage = async (imageFile, options = {}) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('style', options.style || 'vibrant');

  try {
    const response = await api.post('/api/preview', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Server error');
    }
    throw error;
  }
};

export const checkServerStatus = async () => {
  try {
    const response = await api.get('/');
    return response.data;
  } catch (error) {
    throw new Error('Server is not available');
  }
};

export default api;