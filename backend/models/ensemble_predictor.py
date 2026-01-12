"""
Ensemble Predictor for aggregating predictions from multiple models
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .model_factory import ModelFactory


class EnsemblePredictor:
    """
    Handles ensemble predictions from multiple models

    This class:
        1. Gets predictions from all available models
        2. Aggregates them using statistical methods (average, weighted, etc.)
        3. Computes confidence metrics (std, min, max)
        4. Returns both individual and ensemble predictions
    """

    def __init__(self, model_factory: ModelFactory):
        """
        Initialize ensemble predictor

        Args:
            model_factory: ModelFactory instance with loaded models
        """
        self.model_factory = model_factory

    def predict_ensemble(self, input_data: pd.DataFrame,
                        method: str = 'average') -> Dict[str, Any]:
        """
        Make predictions using all available models and compute ensemble

        Args:
            input_data: DataFrame with raw input features (14 columns)
            method: Ensemble method ('average', 'median', 'weighted')

        Returns:
            Dict containing:
                - individual_predictions: List of predictions from each model
                - ensemble: Dict with ensemble statistics
                    - method: Ensemble method used
                    - prediction_lakhs: Ensemble prediction
                    - std_dev: Standard deviation across models
                    - min: Minimum prediction
                    - max: Maximum prediction
                    - num_models: Number of models used

        Raises:
            ValueError: If no models are available or all predictions fail
        """
        models = self.model_factory.get_all_models()

        if not models:
            raise ValueError("No models available for prediction")

        individual_predictions = []
        prediction_values = []
        failed_models = []

        # Get predictions from all models
        for model_name, model in models.items():
            try:
                result = model.predict(input_data)

                if result.get('status') == 'success':
                    individual_predictions.append(result)
                    prediction_values.append(result['prediction_lakhs'])
                else:
                    failed_models.append({
                        'model_name': model_name,
                        'error': result.get('error', 'Unknown error'),
                        'status': 'error'
                    })
                    print(f"Warning: {model_name} prediction failed: {result.get('error')}")

            except Exception as e:
                failed_models.append({
                    'model_name': model_name,
                    'error': str(e),
                    'status': 'error'
                })
                print(f"Error in {model_name}: {str(e)}")

        if not prediction_values:
            raise ValueError(
                f"No successful predictions from any model. "
                f"Failed models: {[m['model_name'] for m in failed_models]}"
            )

        # Compute ensemble statistics based on method
        if method == 'average':
            ensemble_prediction = float(np.mean(prediction_values))
        elif method == 'median':
            ensemble_prediction = float(np.median(prediction_values))
        elif method == 'weighted':
            # For future: implement weighted average based on model performance
            # For now, fall back to simple average
            ensemble_prediction = float(np.mean(prediction_values))
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        # Compute confidence metrics
        ensemble_std = float(np.std(prediction_values))
        ensemble_min = float(np.min(prediction_values))
        ensemble_max = float(np.max(prediction_values))

        # Confidence interval (95% assuming normal distribution)
        confidence_95 = 1.96 * ensemble_std

        result = {
            'individual_predictions': individual_predictions,
            'ensemble': {
                'method': method,
                'prediction_lakhs': ensemble_prediction,
                'std_dev': ensemble_std,
                'min': ensemble_min,
                'max': ensemble_max,
                'num_models': len(prediction_values),
                'confidence_interval_95': confidence_95,
                'confidence_lower': max(0, ensemble_prediction - confidence_95),
                'confidence_upper': ensemble_prediction + confidence_95
            }
        }

        # Add failed models info if any
        if failed_models:
            result['failed_models'] = failed_models

        return result

    def predict_single_model(self, input_data: pd.DataFrame,
                            model_name: str) -> Dict[str, Any]:
        """
        Make prediction using a single specific model

        Args:
            input_data: DataFrame with raw input features
            model_name: Name of the model to use

        Returns:
            Dict with prediction result

        Raises:
            ValueError: If model not found
        """
        model = self.model_factory.get_model(model_name)

        if not model:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {self.model_factory.get_available_models()}"
            )

        return model.predict(input_data)

    def get_ensemble_weights(self) -> Dict[str, float]:
        """
        Get weights for weighted ensemble (for future implementation)

        Weights could be based on:
            - Historical performance (MAE, RMSE)
            - Cross-validation scores
            - Domain expertise

        Returns:
            Dict mapping model names to weights
        """
        # For now, return equal weights
        models = self.model_factory.get_all_models()
        num_models = len(models)

        if num_models == 0:
            return {}

        weight = 1.0 / num_models
        return {name: weight for name in models.keys()}

    def get_prediction_summary(self, predictions: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of predictions

        Args:
            predictions: Result from predict_ensemble()

        Returns:
            str: Formatted summary
        """
        ensemble = predictions['ensemble']
        individual = predictions['individual_predictions']

        summary = []
        summary.append("=" * 60)
        summary.append("PREDICTION SUMMARY")
        summary.append("=" * 60)

        # Ensemble prediction
        summary.append(f"\nEnsemble Prediction ({ensemble['method']}):")
        summary.append(f"  Price: Rs. {ensemble['prediction_lakhs']:.2f} Lakhs")
        summary.append(f"  Confidence Interval (95%): Rs. {ensemble['confidence_lower']:.2f} - {ensemble['confidence_upper']:.2f} Lakhs")
        summary.append(f"  Standard Deviation: ±{ensemble['std_dev']:.2f} Lakhs")
        summary.append(f"  Range: Rs. {ensemble['min']:.2f} - {ensemble['max']:.2f} Lakhs")
        summary.append(f"  Models Used: {ensemble['num_models']}")

        # Individual predictions
        summary.append("\nIndividual Model Predictions:")
        for pred in individual:
            summary.append(f"  {pred['model_name']}: Rs. {pred['prediction_lakhs']:.2f} Lakhs")

        # Failed models (if any)
        if 'failed_models' in predictions and predictions['failed_models']:
            summary.append("\nFailed Models:")
            for failed in predictions['failed_models']:
                summary.append(f"  {failed['model_name']}: {failed['error']}")

        summary.append("=" * 60)

        return "\n".join(summary)

    def __repr__(self) -> str:
        """String representation"""
        return f"EnsemblePredictor(models={self.model_factory.get_model_count()})"
