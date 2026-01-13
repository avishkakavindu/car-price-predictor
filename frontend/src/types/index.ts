/**
 * TypeScript type definitions for the Car Price Prediction system
 */

// Car input data structure
export interface CarData {
  Brand: string;
  Model: string;
  YOM: number;
  'Engine (cc)': number;
  Gear: string;
  'Fuel Type': string;
  'Millage(KM)': number;
  Town: string;
  Date: string;
  Leasing: string;
  Condition: string;
  'AIR CONDITION': string;
  'POWER STEERING': string;
  'POWER MIRROR': string;
  'POWER WINDOW': string;
}

// Individual model prediction
export interface ModelPrediction {
  model_name: string;
  prediction_log: number;
  prediction_lakhs: number;
  status: 'success' | 'error';
  error?: string;
}

// Ensemble prediction
export interface EnsemblePrediction {
  method: string;
  prediction_lakhs: number;
  std_dev: number;
  min: number;
  max: number;
  num_models: number;
  confidence_interval_95?: number;
  confidence_lower?: number;
  confidence_upper?: number;
}

// Prediction API response
export interface PredictionResponse {
  success: boolean;
  predictions: ModelPrediction[];
  ensemble: EnsemblePrediction;
  error?: string;
}

// SHAP feature importance
export interface FeatureImportance {
  feature: string;
  importance: number;
}

// Individual SHAP value for a feature
export interface ShapValue {
  feature: string;
  shap_value: number;
  feature_value: number;
}

// SHAP explanation data
export interface ShapData {
  status: 'processing' | 'completed' | 'error';
  progress?: number;
  bar_chart?: string;  // Base64 encoded image
  waterfall?: string;  // Base64 encoded image
  feature_importance?: FeatureImportance[];
  base_value?: number;
  shap_values?: ShapValue[];  // Individual SHAP values for this prediction
  error?: string;
}

// SHAP generation response
export interface ShapGenerationResponse {
  success: boolean;
  request_id?: string;
  status?: string;
  message?: string;
  error?: string;
}

// Model info
export interface ModelInfo {
  model_name: string;
  algorithm: string;
  is_loaded: boolean;
  model_path?: string;
  features?: number;
  hyperparameters?: {
    n_estimators?: number;
    learning_rate?: number;
    max_depth?: number;
  };
}

// Models API response
export interface ModelsResponse {
  success: boolean;
  models: ModelInfo[];
  count: number;
  error?: string;
}

// Health check response
export interface HealthResponse {
  status: string;
  models_loaded: number;
  available_models: string[];
}

// Form validation errors
export interface ValidationErrors {
  [key: string]: string;
}
