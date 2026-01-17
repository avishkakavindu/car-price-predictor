/**
 * Main application component
 * Integrates prediction form, results display, and SHAP explanations
 */

import React, { useState, useEffect } from 'react';
import PredictionForm from './components/PredictionForm/PredictionForm';
import PredictionResults from './components/PredictionResults/PredictionResults';
import ShapExplanation from './components/ShapExplanation/ShapExplanation';
import ErrorMessage from './components/common/ErrorMessage';
import {
  getAvailableModels,
  generateShap,
  getShapStatus,
} from './services/predictionService';
import type { CarData, PredictionResponse, ShapData } from './types';
import './App.css';

const App: React.FC = () => {
  const [predictionData, setPredictionData] =
    useState<PredictionResponse | null>(null);
  const [shapData, setShapData] = useState<ShapData | null>(null);
  const [inputData, setInputData] = useState<CarData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedShapModel, setSelectedShapModel] = useState<string>('xgboost');
  const [isLoadingShap, setIsLoadingShap] = useState<boolean>(false);

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        console.log('Fetching available models...');
        const response = await getAvailableModels();
        console.log('Models response:', response);
        if (response.success && response.models) {
          const modelNames = response.models.map((m) =>
            m.model_name.toLowerCase()
          );
          console.log('Available models:', modelNames);
          setAvailableModels(modelNames);
          if (
            modelNames.length > 0 &&
            !modelNames.includes(selectedShapModel)
          ) {
            setSelectedShapModel(modelNames[0]);
          }
        }
      } catch (err) {
        console.error('Failed to fetch models:', err);
        // Set default models as fallback
        setAvailableModels(['xgboost', 'lightgbm', 'catboost']);
      }
    };
    fetchModels();
  }, []);

  const handlePredictionSuccess = (
    data: PredictionResponse,
    input: CarData
  ) => {
    setPredictionData(data);
    setInputData(input);
    setShapData(null); // Reset SHAP data when new prediction is made
    setError(null);

    // Fallback: get available models from prediction response if not already set
    if (availableModels.length === 0 && data.predictions) {
      const modelNames = data.predictions
        .filter((p) => p.status === 'success')
        .map((p) => p.model_name.toLowerCase());
      if (modelNames.length > 0) {
        setAvailableModels(modelNames);
        if (!modelNames.includes(selectedShapModel)) {
          setSelectedShapModel(modelNames[0]);
        }
      }
    }
  };

  const handleShapData = (data: ShapData) => {
    setShapData(data);
    setIsLoadingShap(false);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
  };

  const handleReset = () => {
    setPredictionData(null);
    setShapData(null);
    setInputData(null);
    setError(null);
    setIsLoadingShap(false);
  };

  // Handle SHAP model selection change
  const handleShapModelChange = async (modelName: string) => {
    if (!inputData || isLoadingShap) return;

    setSelectedShapModel(modelName);
    setShapData(null);
    setIsLoadingShap(true);

    try {
      const shapResult = await generateShap(inputData, modelName);
      if (shapResult.success && shapResult.request_id) {
        pollShapStatus(shapResult.request_id);
      } else {
        setIsLoadingShap(false);
      }
    } catch (err) {
      console.error('SHAP generation error:', err);
      setIsLoadingShap(false);
    }
  };

  // Poll for SHAP status
  const pollShapStatus = async (requestId: string) => {
    const maxAttempts = 60; // 60 attempts * 2 seconds = 120 seconds max (CatBoost may take longer)
    let attempts = 0;

    const poll = setInterval(async () => {
      attempts++;

      try {
        const status = await getShapStatus(requestId);

        if (status.status === 'completed') {
          handleShapData(status);
          clearInterval(poll);
        } else if (status.status === 'error' || attempts >= maxAttempts) {
          console.error('SHAP generation failed or timed out');
          setIsLoadingShap(false);
          clearInterval(poll);
        }
      } catch (error) {
        console.error('Error polling SHAP:', error);
        if (attempts >= maxAttempts) {
          setIsLoadingShap(false);
          clearInterval(poll);
        }
      }
    }, 2000);
  };

  return (
    <div className='app'>
      <header className='app-header'>
        <div className='header-content'>
          <div className='header-left'>
            <h1>Car Price Prediction</h1>
            <p className='subtitle'>
              ML-powered price estimation with explainable AI
            </p>
          </div>
          <div className='header-right'>
            <h4 className='team-title'>Team Members</h4>
            <div className='team-list'>
              <div className='team-member'>
                <span className='member-name'>Perera A.A.R.D.</span>
                <span className='member-id'>258812E</span>
              </div>
              <div className='team-member'>
                <span className='member-name'>Perera W.A.N.M.</span>
                <span className='member-id'>258815P</span>
              </div>
              <div className='team-member'>
                <span className='member-name'>Dambawinna W.R.P.W.M.A.K.B.</span>
                <span className='member-id'>258765K</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className='app-main'>
        <div className='container'>
          {error && (
            <ErrorMessage message={error} onClose={() => setError(null)} />
          )}

          {!predictionData && (
            <PredictionForm
              onPredictionSuccess={handlePredictionSuccess}
              onShapData={handleShapData}
              onError={handleError}
            />
          )}

          {predictionData && inputData && (
            <>
              <PredictionResults
                predictions={predictionData.predictions}
                ensemble={predictionData.ensemble}
                input={inputData}
                onReset={handleReset}
              />

              {shapData && shapData.status === 'completed' && (
                <ShapExplanation
                  shapData={shapData}
                  availableModels={availableModels}
                  selectedModel={selectedShapModel}
                  onModelChange={handleShapModelChange}
                  isLoading={isLoadingShap}
                />
              )}

              {(isLoadingShap ||
                (shapData && shapData.status === 'processing')) && (
                <div className='shap-loading'>
                  <div className='loading-spinner'></div>
                  <p>
                    Generating SHAP explanations for{' '}
                    {selectedShapModel.toUpperCase()}...
                  </p>
                  <p className='shap-loading-hint'>
                    {selectedShapModel === 'catboost'
                      ? 'CatBoost uses KernelExplainer which may take longer...'
                      : 'This may take a few seconds...'}
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </main>

      <footer className='app-footer'>
        <p>
          Powered by XGBoost, LightGBM, CatBoost, SHAP &amp; React | MSc in AI -
          Machine Learning Assignment
        </p>
      </footer>
    </div>
  );
};

export default App;
