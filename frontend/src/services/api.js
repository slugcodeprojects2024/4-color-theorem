import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for image processing
});

export const processImage = async (imageFile, style = 'vibrant', stainedGlassEnabled = false) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('style', style);
  
  if (stainedGlassEnabled) {
    formData.append('stained_glass', 'true');
  }

  try {
    const response = await api.post('/api/process', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (response.data.success) {
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

export const checkServerStatus = async () => {
  try {
    const response = await api.get('/');
    return response.data;
  } catch (error) {
    throw new Error('Server is not available');
  }
};

export default api;

