import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

/**
 * Extract a human-readable error message from an Axios error or backend response.
 */
export function extractApiErrorMessage(error) {
  if (!error) return 'An unknown error occurred.';
  if (error.response) {
    const data = error.response.data;
    if (data && typeof data === 'object') {
      if (data.error) return data.error;
      if (data.message) return data.message;
    }
    if (error.response.status === 413) {
      return 'Uploaded image exceeds the maximum allowed size (10 MB).';
    }
    if (error.response.status === 503) {
      return 'Trained model is not loaded yet. Please train the model first via scripts/train_model.py.';
    }
    return `Server returned HTTP ${error.response.status}.`;
  }
  if (error.request) {
    return 'Cannot reach the Flask backend API (http://127.0.0.1:5000). Ensure the backend server is running.';
  }
  return error.message || 'Unexpected error communicating with backend.';
}

/**
 * Resolve relative `/api/...` image URLs returned by the backend.
 */
export function resolveBackendAssetUrl(relativeUrl) {
  if (!relativeUrl) return '';
  if (relativeUrl.startsWith('http://') || relativeUrl.startsWith('https://') || relativeUrl.startsWith('blob:')) {
    return relativeUrl;
  }
  return `${API_BASE_URL}${relativeUrl}`;
}

/**
 * GET /api/health
 */
export async function fetchHealthStatus() {
  const response = await apiClient.get('/api/health');
  return response.data;
}

/**
 * GET /api/model-info
 */
export async function fetchModelInfo() {
  const response = await apiClient.get('/api/model-info');
  return response.data;
}

/**
 * POST /api/predict
 */
export async function predictLesionImage(file, includeGradcam = true) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('include_gradcam', includeGradcam ? 'true' : 'false');

  const response = await apiClient.post('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * POST /api/gradcam
 */
export async function generateGradcamForImage(file) {
  const formData = new FormData();
  formData.append('image', file);

  const response = await apiClient.post('/api/gradcam', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * POST /api/report
 * Downloads a generated PDF report for a prediction result.
 */
export async function downloadPredictionReport(predictionData) {
  const response = await apiClient.post('/api/report', predictionData, {
    headers: { 'Content-Type': 'application/json' },
    responseType: 'blob',
  });

  const blob = new Blob([response.data], { type: 'application/pdf' });
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  const safeId = predictionData?.prediction_id || 'result';
  link.href = downloadUrl;
  link.setAttribute('download', `skin_lesion_report_${safeId}.pdf`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(downloadUrl);
}
