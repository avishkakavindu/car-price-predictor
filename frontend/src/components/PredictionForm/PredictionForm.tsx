/**
 * Main prediction form component with all 14 car input fields
 */

import React, { useState } from 'react';
import type { CarData, ValidationErrors, PredictionResponse, ShapData } from '../../types';
import {
  CAR_BRANDS,
  GEAR_TYPES,
  FUEL_TYPES,
  TOWNS,
  LEASING_OPTIONS,
  CONDITIONS,
  AVAILABLE_OPTIONS
} from '../../utils/constants';
import { validateCarData, hasValidationErrors } from '../../utils/validation';
import { predictPrice, generateShap } from '../../services/predictionService';
import './PredictionForm.css';

interface PredictionFormProps {
  onPredictionSuccess: (data: PredictionResponse, input: CarData) => void;
  onShapData: (data: ShapData) => void;
  onError: (error: string) => void;
}

const PredictionForm: React.FC<PredictionFormProps> = ({
  onPredictionSuccess,
  onShapData,
  onError
}) => {
  const [formData, setFormData] = useState<CarData>({
    Brand: '',
    Model: '',
    YOM: new Date().getFullYear() - 5,
    'Engine (cc)': 1500,
    Gear: 'Automatic',
    'Fuel Type': 'Petrol',
    'Millage(KM)': 50000,
    Town: 'Colombo',
    Date: new Date().toISOString().split('T')[0],
    Leasing: 'No Leasing',
    Condition: 'USED',
    'AIR CONDITION': 'Available',
    'POWER STEERING': 'Available',
    'POWER MIRROR': 'Available',
    'POWER WINDOW': 'Available'
  });

  const [errors, setErrors] = useState<ValidationErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;

    const updatedData = {
      ...formData,
      [name]: ['YOM', 'Engine (cc)', 'Millage(KM)'].includes(name)
        ? Number(value)
        : value
    };

    setFormData(updatedData);
    setErrors(prev => ({ ...prev, [name]: '' }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate
    const validationErrors = validateCarData(formData);
    if (hasValidationErrors(validationErrors)) {
      setErrors(validationErrors);
      onError('Please fix the validation errors before submitting');
      return;
    }

    setIsSubmitting(true);

    try {
      // Get predictions
      const predictionResult = await predictPrice(formData);
      onPredictionSuccess(predictionResult, formData);

      // Start SHAP generation (async)
      const shapResult = await generateShap(formData, 'xgboost');
      if (shapResult.success && shapResult.request_id) {
        // Poll for SHAP results
        pollShapStatus(shapResult.request_id);
      }
    } catch (error: any) {
      onError(error.message || 'Failed to get prediction');
    } finally {
      setIsSubmitting(false);
    }
  };

  const pollShapStatus = async (requestId: string) => {
    const maxAttempts = 30; // 30 attempts * 2 seconds = 60 seconds max
    let attempts = 0;

    const poll = setInterval(async () => {
      attempts++;

      try {
        const response = await fetch(`/api/shap/status/${requestId}`);
        const status = await response.json();

        if (status.status === 'completed') {
          onShapData(status);
          clearInterval(poll);
        } else if (status.status === 'error' || attempts >= maxAttempts) {
          console.error('SHAP generation failed or timed out');
          clearInterval(poll);
        }
      } catch (error) {
        console.error('Error polling SHAP:', error);
        if (attempts >= maxAttempts) {
          clearInterval(poll);
        }
      }
    }, 2000); // Poll every 2 seconds
  };

  return (
    <div className="form-card">
      <h2>Enter Car Details</h2>
      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="form-grid">
          {/* Row 1: Brand and Model */}
          <div className="form-group">
            <label htmlFor="Brand">Brand *</label>
            <select
              id="Brand"
              name="Brand"
              value={formData.Brand}
              onChange={handleChange}
              className={errors.Brand ? 'error' : ''}
              required
            >
              <option value="">Select Brand</option>
              {CAR_BRANDS.map(brand => (
                <option key={brand} value={brand}>{brand}</option>
              ))}
            </select>
            {errors.Brand && <span className="error-text">{errors.Brand}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="Model">Model *</label>
            <input
              type="text"
              id="Model"
              name="Model"
              value={formData.Model}
              onChange={handleChange}
              placeholder="e.g., Corolla"
              className={errors.Model ? 'error' : ''}
              required
            />
            {errors.Model && <span className="error-text">{errors.Model}</span>}
          </div>

          {/* Row 2: Year and Engine */}
          <div className="form-group">
            <label htmlFor="YOM">Year of Manufacture *</label>
            <input
              type="number"
              id="YOM"
              name="YOM"
              value={formData.YOM}
              onChange={handleChange}
              min="1990"
              max={new Date().getFullYear()}
              className={errors.YOM ? 'error' : ''}
              required
            />
            {errors.YOM && <span className="error-text">{errors.YOM}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="Engine (cc)">Engine Capacity (cc) *</label>
            <input
              type="number"
              id="Engine (cc)"
              name="Engine (cc)"
              value={formData['Engine (cc)']}
              onChange={handleChange}
              placeholder="e.g., 1500"
              min="500"
              max="5000"
              className={errors['Engine (cc)'] ? 'error' : ''}
              required
            />
            {errors['Engine (cc)'] && <span className="error-text">{errors['Engine (cc)']}</span>}
          </div>

          {/* Row 3: Gear and Fuel */}
          <div className="form-group">
            <label htmlFor="Gear">Transmission *</label>
            <select
              id="Gear"
              name="Gear"
              value={formData.Gear}
              onChange={handleChange}
              required
            >
              {GEAR_TYPES.map(gear => (
                <option key={gear} value={gear}>{gear}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="Fuel Type">Fuel Type *</label>
            <select
              id="Fuel Type"
              name="Fuel Type"
              value={formData['Fuel Type']}
              onChange={handleChange}
              required
            >
              {FUEL_TYPES.map(fuel => (
                <option key={fuel} value={fuel}>{fuel}</option>
              ))}
            </select>
          </div>

          {/* Row 4: Mileage and Town */}
          <div className="form-group">
            <label htmlFor="Millage(KM)">Mileage (KM) *</label>
            <input
              type="number"
              id="Millage(KM)"
              name="Millage(KM)"
              value={formData['Millage(KM)']}
              onChange={handleChange}
              placeholder="e.g., 50000"
              min="0"
              max="500000"
              className={errors['Millage(KM)'] ? 'error' : ''}
              required
            />
            {errors['Millage(KM)'] && <span className="error-text">{errors['Millage(KM)']}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="Town">Town *</label>
            <select
              id="Town"
              name="Town"
              value={formData.Town}
              onChange={handleChange}
              required
            >
              {TOWNS.map(town => (
                <option key={town} value={town}>{town}</option>
              ))}
            </select>
          </div>

          {/* Row 5: Condition and Leasing */}
          <div className="form-group">
            <label htmlFor="Condition">Condition *</label>
            <select
              id="Condition"
              name="Condition"
              value={formData.Condition}
              onChange={handleChange}
              required
            >
              {CONDITIONS.map(condition => (
                <option key={condition} value={condition}>{condition}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="Leasing">Leasing *</label>
            <select
              id="Leasing"
              name="Leasing"
              value={formData.Leasing}
              onChange={handleChange}
              required
            >
              {LEASING_OPTIONS.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Features Section */}
        <div className="features-section">
          <h3>Car Features</h3>
          <div className="features-grid">
            <div className="form-group-compact">
              <label htmlFor="AIR CONDITION">Air Conditioning</label>
              <select
                id="AIR CONDITION"
                name="AIR CONDITION"
                value={formData['AIR CONDITION']}
                onChange={handleChange}
              >
                {AVAILABLE_OPTIONS.map(option => (
                  <option key={option} value={option}>
                    {option === 'Available' ? 'Yes' : 'No'}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group-compact">
              <label htmlFor="POWER STEERING">Power Steering</label>
              <select
                id="POWER STEERING"
                name="POWER STEERING"
                value={formData['POWER STEERING']}
                onChange={handleChange}
              >
                {AVAILABLE_OPTIONS.map(option => (
                  <option key={option} value={option}>
                    {option === 'Available' ? 'Yes' : 'No'}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group-compact">
              <label htmlFor="POWER MIRROR">Power Mirror</label>
              <select
                id="POWER MIRROR"
                name="POWER MIRROR"
                value={formData['POWER MIRROR']}
                onChange={handleChange}
              >
                {AVAILABLE_OPTIONS.map(option => (
                  <option key={option} value={option}>
                    {option === 'Available' ? 'Yes' : 'No'}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group-compact">
              <label htmlFor="POWER WINDOW">Power Window</label>
              <select
                id="POWER WINDOW"
                name="POWER WINDOW"
                value={formData['POWER WINDOW']}
                onChange={handleChange}
              >
                {AVAILABLE_OPTIONS.map(option => (
                  <option key={option} value={option}>
                    {option === 'Available' ? 'Yes' : 'No'}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="submit-button"
          disabled={isSubmitting}
        >
          {isSubmitting ? 'Predicting...' : 'Get Price Prediction'}
        </button>
      </form>
    </div>
  );
};

export default PredictionForm;
