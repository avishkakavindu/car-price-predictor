/**
 * Main application component
 * Integrates prediction form, results display, and SHAP explanations
 */

import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm/PredictionForm';
import PredictionResults from './components/PredictionResults/PredictionResults';
import ShapExplanation from './components/ShapExplanation/ShapExplanation';
import ErrorMessage from './components/common/ErrorMessage';
import type { CarData, PredictionResponse, ShapData } from './types';
import './App.css';

const App: React.FC = () => {
  const [predictionData, setPredictionData] = useState<PredictionResponse | null>(null);
  const [shapData, setShapData] = useState<ShapData | null>(null);
  const [inputData, setInputData] = useState<CarData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredictionSuccess = (
    data: PredictionResponse,
    input: CarData
  ) => {
    setPredictionData(data);
    setInputData(input);
    setShapData(null); // Reset SHAP data when new prediction is made
    setError(null);
  };

  const handleShapData = (data: ShapData) => {
    setShapData(data);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
  };

  const handleReset = () => {
    setPredictionData(null);
    setShapData(null);
    setInputData(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>Car Price Prediction</h1>
          <p className="subtitle">
            ML-powered price estimation with explainable AI
          </p>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
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
                <ShapExplanation shapData={shapData} />
              )}

              {shapData && shapData.status === 'processing' && (
                <div className="shap-loading">
                  <div className="loading-spinner"></div>
                  <p>Generating SHAP explanations...</p>
                </div>
              )}
            </>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>
          Powered by XGBoost, SHAP, and React | MSc in AI - Machine Learning Assignment
        </p>
      </footer>
    </div>
  );
};

export default App;
