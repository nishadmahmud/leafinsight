import axios from 'axios';

const API_BASE_URL = 'https://api.eurus.studio/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealth = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

export const getModels = async () => {
  const response = await apiClient.get('/models');
  return response.data;
};

export const detectImage = async (file, modelName, confThresh = 0.25, iouThresh = 0.45) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('model_name', modelName);
  formData.append('conf_thresh', confThresh);
  formData.append('iou_thresh', iouThresh);

  const response = await apiClient.post('/detect', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const compareModels = async (file, modelsList, confThresh = 0.25, iouThresh = 0.45) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('models', modelsList.join(','));
  formData.append('conf_thresh', confThresh);
  formData.append('iou_thresh', iouThresh);

  const response = await apiClient.post('/compare', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const explainModel = async (file, modelName, method) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('model_name', modelName);
  formData.append('method', method); // 'rise' or 'occlusion'

  const response = await apiClient.post('/explain', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const benchmarkModels = async (file, modelsList) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('models', modelsList.join(','));

  const response = await apiClient.post('/benchmark', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};
