/**
 * SHAP explanation visualization component
 */

import React from 'react';
import type { ShapData } from '../../types';
import './ShapExplanation.css';

interface ShapExplanationProps {
  shapData: ShapData;
  availableModels?: string[];
  selectedModel?: string;
  onModelChange?: (modelName: string) => void;
  isLoading?: boolean;
}

const ShapExplanation: React.FC<ShapExplanationProps> = ({
  shapData,
  availableModels = [],
  selectedModel = 'xgboost',
  onModelChange,
  isLoading = false
}) => {
  if (!shapData || shapData.status !== 'completed') {
    return null;
  }

  const getModelDisplayName = (model: string) => {
    const names: { [key: string]: string } = {
      'xgboost': 'XGBoost',
      'lightgbm': 'LightGBM',
      'adaboost': 'AdaBoost'
    };
    return names[model.toLowerCase()] || model.toUpperCase();
  };

  return (
    <div className="shap-container">
      <div className="shap-header">
        <h2>Model Explanation (SHAP Analysis)</h2>

        {/* Model Selector */}
        {availableModels.length > 1 && onModelChange && (
          <div className="model-selector">
            <label htmlFor="shap-model-select">Explain with:</label>
            <select
              id="shap-model-select"
              value={selectedModel}
              onChange={(e) => onModelChange(e.target.value)}
              disabled={isLoading}
              className="model-select"
            >
              {availableModels.map((model) => (
                <option key={model} value={model}>
                  {getModelDisplayName(model)}
                </option>
              ))}
            </select>
            {isLoading && <span className="loading-indicator">Loading...</span>}
          </div>
        )}
      </div>

      {/* Current Model Badge */}
      <div className="current-model-badge">
        Showing SHAP explanation for: <strong>{getModelDisplayName(shapData.model_name || selectedModel)}</strong>
      </div>

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
          {shapData.base_value !== undefined && (
            <div className="base-value-info">
              <strong>Base Value (Log Scale):</strong> {shapData.base_value.toFixed(3)}
              <br />
              <strong>Converted to Actual Price:</strong> Rs. {(Math.exp(shapData.base_value) - 1).toFixed(2)} Lakhs
              <br />
              <span style={{ fontSize: '0.9em', color: '#666' }}>
                (This is the average prediction before considering specific car features)
              </span>
            </div>
          )}
        </div>
      )}

      {/* SHAP Values Breakdown */}
      {shapData.shap_values && shapData.base_value !== undefined && (
        <div className="shap-breakdown">
          <h3>📊 SHAP Values Breakdown & Calculation</h3>
          <p style={{ fontSize: '0.9em', color: '#666', marginBottom: '16px' }}>
            See how each feature contributed to the final prediction (sorted by impact):
          </p>

          <div className="calculation-example" style={{
            backgroundColor: '#f8f9fa',
            padding: '16px',
            borderRadius: '8px',
            marginBottom: '16px',
            border: '2px solid #e9ecef'
          }}>
            <h4 style={{ marginTop: 0 }}>💡 Step-by-Step Price Calculation:</h4>
            <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
              <div style={{ marginBottom: '8px' }}>
                <strong>1. Start with Base Value:</strong> {shapData.base_value.toFixed(4)} (log scale)
                = Rs. {(Math.exp(shapData.base_value) - 1).toFixed(2)} Lakhs
              </div>

              {shapData.shap_values.slice(0, 5).map((item, idx) => {
                const cumulativeLog = shapData.base_value! +
                  shapData.shap_values!.slice(0, idx + 1).reduce((sum, v) => sum + v.shap_value, 0);
                const cumulativePrice = Math.exp(cumulativeLog) - 1;
                const sign = item.shap_value >= 0 ? '+' : '';
                const percentChange = ((Math.exp(item.shap_value) - 1) * 100);

                return (
                  <div key={idx} style={{
                    marginBottom: '8px',
                    paddingLeft: '20px',
                    borderLeft: item.shap_value >= 0 ? '3px solid #dc3545' : '3px solid #007bff'
                  }}>
                    <strong>{idx + 2}. {item.shap_value >= 0 ? 'Add' : 'Subtract'} {item.feature}:</strong><br />
                    &nbsp;&nbsp;{sign}{item.shap_value.toFixed(4)} (log)
                    ≈ {percentChange >= 0 ? '+' : ''}{percentChange.toFixed(1)}% impact
                    <br />
                    &nbsp;&nbsp;Running total: {cumulativeLog.toFixed(4)} (log)
                    = Rs. {cumulativePrice.toFixed(2)} Lakhs
                  </div>
                );
              })}

              {shapData.shap_values.length > 5 && (
                <div style={{ marginBottom: '8px', paddingLeft: '20px', fontStyle: 'italic', color: '#666' }}>
                  ... + {shapData.shap_values.length - 5} more features ...
                </div>
              )}

              <div style={{
                marginTop: '12px',
                paddingTop: '12px',
                borderTop: '2px solid #495057',
                fontWeight: 'bold'
              }}>
                <strong>Final Prediction:</strong> {
                  (shapData.base_value +
                   shapData.shap_values.reduce((sum, v) => sum + v.shap_value, 0)).toFixed(4)
                } (log scale)
                <br />
                = Rs. {(Math.exp(
                  shapData.base_value +
                  shapData.shap_values.reduce((sum, v) => sum + v.shap_value, 0)
                ) - 1).toFixed(2)} Lakhs
              </div>
            </div>
          </div>

          <div className="shap-values-table" style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '0.9em'
            }}>
              <thead>
                <tr style={{ backgroundColor: '#e9ecef' }}>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>
                    Feature
                  </th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>
                    SHAP Value<br />(Log Scale)
                  </th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>
                    Impact<br />(Percentage)
                  </th>
                  <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>
                    Direction
                  </th>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>
                    Feature Value
                  </th>
                </tr>
              </thead>
              <tbody>
                {shapData.shap_values.map((item, index) => {
                  const percentImpact = ((Math.exp(item.shap_value) - 1) * 100);
                  const isPositive = item.shap_value >= 0;

                  return (
                    <tr key={index} style={{
                      backgroundColor: index % 2 === 0 ? '#fff' : '#f8f9fa'
                    }}>
                      <td style={{ padding: '10px', borderBottom: '1px solid #dee2e6' }}>
                        <strong>{item.feature}</strong>
                      </td>
                      <td style={{
                        padding: '10px',
                        textAlign: 'center',
                        fontFamily: 'monospace',
                        borderBottom: '1px solid #dee2e6',
                        color: isPositive ? '#dc3545' : '#007bff'
                      }}>
                        {item.shap_value >= 0 ? '+' : ''}{item.shap_value.toFixed(4)}
                      </td>
                      <td style={{
                        padding: '10px',
                        textAlign: 'center',
                        fontWeight: 'bold',
                        borderBottom: '1px solid #dee2e6',
                        color: isPositive ? '#dc3545' : '#007bff'
                      }}>
                        {percentImpact >= 0 ? '+' : ''}{percentImpact.toFixed(2)}%
                      </td>
                      <td style={{
                        padding: '10px',
                        textAlign: 'center',
                        borderBottom: '1px solid #dee2e6'
                      }}>
                        <span style={{
                          padding: '4px 12px',
                          borderRadius: '12px',
                          backgroundColor: isPositive ? '#dc354520' : '#007bff20',
                          color: isPositive ? '#dc3545' : '#007bff',
                          fontWeight: 'bold',
                          fontSize: '0.85em'
                        }}>
                          {isPositive ? '▲ Increase' : '▼ Decrease'}
                        </span>
                      </td>
                      <td style={{
                        padding: '10px',
                        fontFamily: 'monospace',
                        borderBottom: '1px solid #dee2e6'
                      }}>
                        {item.feature_value.toFixed(3)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
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
              <p style={{ fontSize: '0.85em', marginTop: '8px', color: '#666' }}>
                <strong>Note:</strong> Values are in log scale. To interpret: positive values (red)
                exponentially increase price, negative values (blue) exponentially decrease price.
                The final prediction at the bottom is also in log scale.
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

        <div className="interpretation-example" style={{ marginTop: '16px', backgroundColor: '#f0f8ff', padding: '12px', borderRadius: '8px' }}>
          <h4>📐 Understanding Log Scale Values:</h4>
          <p style={{ fontSize: '0.9em', marginBottom: '8px' }}>
            The waterfall plot shows values in <strong>log scale</strong>. Here's what that means:
          </p>
          <ul style={{ fontSize: '0.9em' }}>
            <li>
              <strong>Base value of 3.89 (log)</strong> = Rs. {(Math.exp(3.89) - 1).toFixed(2)} Lakhs (actual)
            </li>
            <li>
              A <strong>+0.5 increase</strong> in log scale ≈ <strong>65% price increase</strong> (e.g., from 30 to 49 Lakhs)
            </li>
            <li>
              A <strong>-0.5 decrease</strong> in log scale ≈ <strong>39% price decrease</strong> (e.g., from 30 to 18 Lakhs)
            </li>
            <li>
              <strong>Larger absolute values</strong> = stronger impact on price (exponential effect)
            </li>
          </ul>
          <p style={{ fontSize: '0.85em', marginTop: '8px', color: '#555' }}>
            💡 <strong>Tip:</strong> Focus on the <em>direction</em> (red vs blue) and <em>relative size</em>
            of bars rather than exact numbers. Your actual predicted price is shown at the top of the page!
          </p>
        </div>
      </div>
    </div>
  );
};

export default ShapExplanation;
