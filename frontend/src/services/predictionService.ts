/**
 * Service for making prediction API calls
 */

import api from './api';
import type {
  CarData,
  PredictionResponse,
  ShapGenerationResponse,
  ShapData,
  ModelsResponse,
  HealthResponse,
} from '../types';

/**
 * Get server health status
 */
export const getHealthStatus = async (): Promise<HealthResponse> => {
  try {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  } catch (error: any) {
    throw new Error(
      error.response?.data?.error || 'Failed to get health status'
    );
  }
};

/**
 * Get available models
 */
export const getAvailableModels = async (): Promise<ModelsResponse> => {
  try {
    const response = await api.get<ModelsResponse>('/models');
    return response.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || 'Failed to fetch models');
  }
};

/**
 * Get predictions from all available models
 */
export const predictPrice = async (
  carData: CarData
): Promise<PredictionResponse> => {
  try {
    const response = await api.post<PredictionResponse>('/predict', carData);

    if (!response.data.success) {
      throw new Error(response.data.error || 'Prediction failed');
    }

    return response.data;
  } catch (error: any) {
    if (error.response) {
      throw new Error(error.response.data.error || 'Server error');
    } else if (error.request) {
      throw new Error('No response from server. Please check your connection.');
    } else {
      throw new Error(error.message || 'Unexpected error occurred');
    }
  }
};

/**
 * Start SHAP generation (async)
 */
export const generateShap = async (
  carData: CarData,
  modelName: string = 'xgboost'
): Promise<ShapGenerationResponse> => {
  try {
    const response = await api.post<ShapGenerationResponse>(
      `/shap/generate?model=${modelName}`,
      carData
    );

    if (!response.data.success) {
      throw new Error(response.data.error || 'SHAP generation failed');
    }

    return response.data;
  } catch (error: any) {
    console.error('SHAP generation error:', error);
    // Don't throw - SHAP is optional
    return {
      success: false,
      error: error.message || 'SHAP generation failed',
    };
  }
};

/**
 * Get SHAP generation status
 */
export const getShapStatus = async (requestId: string): Promise<ShapData> => {
  try {
    const response = await api.get<ShapData>(`/shap/status/${requestId}`);
    return response.data;
  } catch (error: any) {
    console.error('SHAP status error:', error);
    return {
      status: 'error',
      error: error.message || 'Failed to get SHAP status',
    };
  }
};

/**
 * Cleanup SHAP results from cache
 */
export const cleanupShap = async (requestId: string): Promise<void> => {
  try {
    await api.delete(`/shap/cleanup/${requestId}`);
  } catch (error: any) {
    console.error('SHAP cleanup error:', error);
  }
};
