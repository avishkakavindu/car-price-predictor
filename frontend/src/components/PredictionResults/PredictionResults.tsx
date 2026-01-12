/**
 * Prediction results display component
 */

import React from 'react';
import type { ModelPrediction, EnsemblePrediction, CarData } from '../../types';
import { formatPrice, formatNumber } from '../../utils/validation';
import './PredictionResults.css';

interface PredictionResultsProps {
  predictions: ModelPrediction[];
  ensemble: EnsemblePrediction;
  input: CarData;
  onReset: () => void;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({
  predictions,
  ensemble,
  input,
  onReset
}) => {
  return (
    <div className="results-container">
      <div className="results-header">
        <h2>Prediction Results</h2>
        <button onClick={onReset} className="reset-button">
          New Prediction
        </button>
      </div>

      {/* Input Summary */}
      <div className="input-summary">
        <h3>Your Car:</h3>
        <div className="summary-grid">
          <div className="summary-item">
            <span className="summary-label">Brand:</span>
            <span className="summary-value">{input.Brand}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Model:</span>
            <span className="summary-value">{input.Model}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Year:</span>
            <span className="summary-value">{input.YOM}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Engine:</span>
            <span className="summary-value">{input['Engine (cc)']} cc</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Mileage:</span>
            <span className="summary-value">{formatNumber(input['Millage(KM)'])} km</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Transmission:</span>
            <span className="summary-value">{input.Gear}</span>
          </div>
        </div>
      </div>

      {/* Ensemble Prediction (Highlighted) */}
      <div className="ensemble-card">
        <div className="ensemble-header">
          <h3>Ensemble Prediction</h3>
          <span className="ensemble-badge">Average of {ensemble.num_models} Model{ensemble.num_models > 1 ? 's' : ''}</span>
        </div>
        <div className="ensemble-price">
          {formatPrice(ensemble.prediction_lakhs)}
        </div>
        <div className="ensemble-details">
          <div className="detail-item">
            <span className="detail-label">Confidence Range:</span>
            <span className="detail-value">
              {formatPrice(ensemble.min)} - {formatPrice(ensemble.max)}
            </span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Standard Deviation:</span>
            <span className="detail-value">±{ensemble.std_dev.toFixed(2)} Lakhs</span>
          </div>
        </div>
      </div>

      {/* Individual Model Predictions */}
      {predictions.length > 1 && (
        <div className="individual-predictions">
          <h3>Individual Model Predictions</h3>
          <div className="models-grid">
            {predictions.map((pred, index) => (
              <div key={index} className="model-card">
                <div className="model-header">
                  <h4>{pred.model_name}</h4>
                  <span className={`status-badge ${pred.status}`}>
                    {pred.status}
                  </span>
                </div>
                {pred.status === 'success' ? (
                  <div className="model-price">
                    {formatPrice(pred.prediction_lakhs)}
                  </div>
                ) : (
                  <div className="model-error">
                    {pred.error || 'Prediction failed'}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Price Range Visualization */}
      <div className="price-range-section">
        <h3>Price Range Analysis</h3>
        <div className="range-visualization">
          <div className="range-labels">
            <span className="range-min">{formatPrice(ensemble.min)}</span>
            <span className="range-max">{formatPrice(ensemble.max)}</span>
          </div>
          <div className="range-bar">
            <div
              className="range-fill"
              style={{
                left: '0%',
                width: '100%'
              }}
            ></div>
            <div
              className="range-marker"
              style={{
                left: ensemble.num_models > 1
                  ? `${((ensemble.prediction_lakhs - ensemble.min) / (ensemble.max - ensemble.min)) * 100}%`
                  : '50%'
              }}
            ></div>
          </div>
          <div className="range-description">
            <p>
              The ensemble prediction ({formatPrice(ensemble.prediction_lakhs)})
              {ensemble.num_models > 1 && ` falls within a range of ${formatPrice(ensemble.min)} to ${formatPrice(ensemble.max)}`}
              {ensemble.std_dev > 0 && ` with a variation of ±${ensemble.std_dev.toFixed(2)} Lakhs`}.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResults;
