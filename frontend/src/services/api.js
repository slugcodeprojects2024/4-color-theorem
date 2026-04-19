import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180000, // 3 minutes for large image processing
});

export const processImage = async (
  imageFile,
  style = 'vibrant',
  stainedGlassEnabled = false,
  lineArtSettings = null,
  mlSettings = null,  // kept for backwards compat, ignored
  useFiveColors = false,
  customColors = null
) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('style', style);
  formData.append('stained_glass', stainedGlassEnabled ? 'true' : 'false');

  if (lineArtSettings && lineArtSettings.enabled) {
    formData.append('convert_to_lineart', 'true');
    formData.append('line_thickness', lineArtSettings.lineThickness || 'medium');
    formData.append('detail_level', lineArtSettings.detailLevel || 'detailed');
    formData.append('contrast', String(lineArtSettings.contrast || 1.0));
  } else {
    formData.append('convert_to_lineart', 'false');
  }

  formData.append('use_five_colors', useFiveColors ? 'true' : 'false');

  if (customColors && Array.isArray(customColors) && customColors.length > 0) {
    formData.append('custom_colors', JSON.stringify(customColors));
  }

  try {
    const response = await api.post('/api/process', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    console.log('API Response:', {
      status: response.status,
      hasData: !!response.data,
      success: response.data?.success,
      hasImage: !!response.data?.image,
      imageLength: response.data?.image?.length,
      hasStats: !!response.data?.stats
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
      throw new Error('Network error - please check if the server is running');
    } else {
      throw new Error(error.message || 'An error occurred');
    }
  }
};

export const previewLineArt = async (imageFile, lineArtSettings) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('line_thickness', lineArtSettings.lineThickness || 'medium');
  formData.append('detail_level', lineArtSettings.detailLevel || 'detailed');
  formData.append('contrast', String(lineArtSettings.contrast || 1.0));

  try {
    const response = await api.post('/api/preview-lineart', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
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
      throw new Error('Network error - please check if the server is running');
    } else {
      throw new Error(error.message || 'An error occurred');
    }
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