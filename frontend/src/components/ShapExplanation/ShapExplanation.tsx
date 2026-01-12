/**
 * SHAP explanation visualization component
 */

import React from 'react';
import type { ShapData } from '../../types';
import './ShapExplanation.css';

interface ShapExplanationProps {
  shapData: ShapData;
}

const ShapExplanation: React.FC<ShapExplanationProps> = ({ shapData }) => {
  if (!shapData || shapData.status !== 'completed') {
    return null;
  }

  return (
    <div className="shap-container">
      <h2>Model Explanation (SHAP Analysis)</h2>
      <p className="shap-description">
        SHAP (SHapley Additive exPlanations) shows how each feature contributed to the prediction.
      </p>

      {/* Feature Importance Bar Chart */}
      {shapData.bar_chart && (
        <div className="chart-section">
          <h3>Feature Importance</h3>
          <p className="chart-description">
            The top features that most influence car prices across all predictions.
          </p>
          <div className="chart-wrapper">
            <img
              src={shapData.bar_chart}
              alt="SHAP Feature Importance"
              className="chart-image"
            />
          </div>
        </div>
      )}

      {/* Top Features List */}
      {shapData.feature_importance && shapData.feature_importance.length > 0 && (
        <div className="features-list-section">
          <h3>Top 10 Most Important Features</h3>
          <div className="features-table">
            <div className="table-header">
              <span className="header-rank">Rank</span>
              <span className="header-feature">Feature</span>
              <span className="header-importance">Importance</span>
              <span className="header-bar">Impact</span>
            </div>
            {shapData.feature_importance.slice(0, 10).map((feature, index) => {
              const maxImportance = shapData.feature_importance![0].importance;
              const percentage = (feature.importance / maxImportance) * 100;

              return (
                <div key={index} className="table-row">
                  <span className="cell-rank">{index + 1}</span>
                  <span className="cell-feature">{feature.feature}</span>
                  <span className="cell-importance">{feature.importance.toFixed(4)}</span>
                  <div className="cell-bar">
                    <div
                      className="importance-bar"
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Waterfall Plot */}
      {shapData.waterfall && (
        <div className="chart-section">
          <h3>Prediction Breakdown</h3>
          <p className="chart-description">
            How each feature pushed the price up (red) or down (blue) for your specific car.
          </p>
          <div className="chart-wrapper">
            <img
              src={shapData.waterfall}
              alt="SHAP Waterfall Plot"
              className="chart-image"
            />
          </div>
          {shapData.base_value && (
            <div className="base-value-info">
              <strong>Base Value:</strong> {shapData.base_value.toFixed(3)} (average prediction in log scale)
            </div>
          )}
        </div>
      )}

      {/* Interpretation Guide */}
      <div className="interpretation-guide">
        <h3>How to Read These Charts</h3>
        <div className="guide-grid">
          <div className="guide-item">
            <div className="guide-icon feature-importance-icon">📊</div>
            <div className="guide-content">
              <h4>Feature Importance</h4>
              <p>
                Shows which features matter most overall. Higher bars mean that feature
                has a bigger impact on price predictions across all cars.
              </p>
            </div>
          </div>

          <div className="guide-item">
            <div className="guide-icon waterfall-icon">🌊</div>
            <div className="guide-content">
              <h4>Waterfall Plot</h4>
              <p>
                Shows step-by-step how your car's features affected the final price,
                starting from the average price (base value).
              </p>
            </div>
          </div>

          <div className="guide-item">
            <div className="guide-icon positive-icon">🔴</div>
            <div className="guide-content">
              <h4>Red / Positive</h4>
              <p>Features that increased the predicted price above the base value.</p>
            </div>
          </div>

          <div className="guide-item">
            <div className="guide-icon negative-icon">🔵</div>
            <div className="guide-content">
              <h4>Blue / Negative</h4>
              <p>Features that decreased the predicted price below the base value.</p>
            </div>
          </div>
        </div>

        <div className="interpretation-example">
          <h4>Example Interpretation:</h4>
          <ul>
            <li>
              <strong>Low Mileage:</strong> If your car has low mileage, it will push
              the price UP (red/positive contribution)
            </li>
            <li>
              <strong>Automatic Transmission:</strong> Automatic cars typically have
              higher prices (positive contribution)
            </li>
            <li>
              <strong>Large Engine:</strong> Higher engine capacity usually increases
              the price
            </li>
            <li>
              <strong>Popular Brand (Toyota):</strong> Well-known brands command premium
              prices
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ShapExplanation;
