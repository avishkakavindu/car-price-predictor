"""
Flask API routes for car price prediction service
"""
from flask import Blueprint, request, jsonify
import pandas as pd
from typing import Dict, Any

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# These will be injected by app.py during initialization
model_factory = None
shap_service = None
ensemble_predictor = None


def init_routes(factory, shap_svc, ensemble_pred):
    """
    Initialize routes with dependencies (dependency injection)

    Args:
        factory: ModelFactory instance
        shap_svc: ShapService instance
        ensemble_pred: EnsemblePredictor instance
    """
    global model_factory, shap_service, ensemble_predictor
    model_factory = factory
    shap_service = shap_svc
    ensemble_predictor = ensemble_pred


@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint

    Returns model loading status and availability

    Response:
        {
            "status": "healthy",
            "models_loaded": 1,
            "available_models": ["xgboost"]
        }
    """
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(model_factory.get_all_models()),
        'available_models': model_factory.get_available_models()
    })


@api_bp.route('/models', methods=['GET'])
def get_models():
    """
    Get list of available models with metadata

    Response:
        {
            "success": true,
            "models": [
                {
                    "model_name": "XGBoost",
                    "algorithm": "XGBoost Regressor",
                    "is_loaded": true,
                    "features": 27
                }
            ],
            "count": 1
        }
    """
    try:
        models_info = model_factory.get_models_info()
        return jsonify({
            'success': True,
            'models': models_info,
            'count': len(models_info)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions with all available models and return ensemble

    Request Body (JSON):
        {
            "Brand": "TOYOTA",
            "Model": "Corolla",
            "YOM": 2018,
            "Engine (cc)": 1500,
            "Gear": "Automatic",
            "Fuel Type": "Petrol",
            "Millage(KM)": 120000,
            "Town": "Colombo",
            "Date": "2025-01-10",
            "Leasing": "No Leasing",
            "Condition": "USED",
            "AIR CONDITION": "Available",
            "POWER STEERING": "Available",
            "POWER MIRROR": "Available",
            "POWER WINDOW": "Available"
        }

    Response:
        {
            "success": true,
            "predictions": [
                {
                    "model_name": "XGBoost",
                    "prediction_lakhs": 45.5,
                    "status": "success"
                }
            ],
            "ensemble": {
                "method": "average",
                "prediction_lakhs": 45.5,
                "std_dev": 0.0,
                "min": 45.5,
                "max": 45.5,
                "num_models": 1,
                "confidence_interval_95": 0.0
            }
        }
    """
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Required fields
        required_fields = [
            'Brand', 'Model', 'YOM', 'Engine (cc)', 'Gear', 'Fuel Type',
            'Millage(KM)', 'Town', 'Date', 'Leasing', 'Condition',
            'AIR CONDITION', 'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW'
        ]

        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Get ensemble prediction
        result = ensemble_predictor.predict_ensemble(input_df)

        return jsonify({
            'success': True,
            'predictions': result['individual_predictions'],
            'ensemble': result['ensemble']
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500


@api_bp.route('/predict/<model_name>', methods=['POST'])
def predict_single(model_name: str):
    """
    Make prediction with a single specific model

    Path Parameters:
        model_name: Name of the model (e.g., 'xgboost')

    Request Body:
        Same as /predict endpoint

    Response:
        {
            "success": true,
            "model_name": "XGBoost",
            "prediction_lakhs": 45.5,
            "status": "success"
        }
    """
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Get prediction from specific model
        result = ensemble_predictor.predict_single_model(input_df, model_name)

        if result.get('status') == 'error':
            return jsonify({
                'success': False,
                'error': result.get('error', 'Prediction failed')
            }), 400

        return jsonify({
            'success': True,
            **result
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500


@api_bp.route('/shap/generate', methods=['POST'])
def generate_shap():
    """
    Start async SHAP generation for a prediction

    Query Parameters:
        model: Model name (optional, defaults to 'xgboost')

    Request Body:
        Same as /predict endpoint

    Response:
        {
            "success": true,
            "request_id": "uuid-string",
            "status": "processing",
            "message": "SHAP generation started. Poll /shap/status/{request_id} for results."
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Get model name from query param
        model_name = request.args.get('model', 'xgboost')
        model_obj = model_factory.get_model(model_name)

        if not model_obj:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not found. Available models: {model_factory.get_available_models()}'
            }), 404

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Start async SHAP generation
        request_id = shap_service.generate_shap_async(model_obj.model, input_df)

        return jsonify({
            'success': True,
            'request_id': request_id,
            'status': 'processing',
            'message': f'SHAP generation started. Poll /api/shap/status/{request_id} for results.'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'SHAP generation failed: {str(e)}'}), 500


@api_bp.route('/shap/status/<request_id>', methods=['GET'])
def get_shap_status(request_id: str):
    """
    Get status of SHAP generation

    Path Parameters:
        request_id: UUID from /shap/generate response

    Response (processing):
        {
            "success": true,
            "status": "processing",
            "progress": 60
        }

    Response (completed):
        {
            "success": true,
            "status": "completed",
            "progress": 100,
            "bar_chart": "data:image/png;base64,...",
            "waterfall": "data:image/png;base64,...",
            "feature_importance": [
                {"feature": "Mileage_log", "importance": 0.2230},
                ...
            ],
            "base_value": 4.029
        }

    Response (error):
        {
            "success": false,
            "status": "error",
            "error": "Error message"
        }
    """
    try:
        status = shap_service.get_shap_status(request_id)

        if status is None:
            return jsonify({
                'success': False,
                'error': 'Request ID not found or expired'
            }), 404

        return jsonify({
            'success': True,
            **status
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/shap/cleanup/<request_id>', methods=['DELETE'])
def cleanup_shap(request_id: str):
    """
    Clean up SHAP results from cache

    Path Parameters:
        request_id: UUID to clean up

    Response:
        {
            "success": true,
            "message": "Cache cleaned up"
        }
    """
    try:
        shap_service.cleanup_cache(request_id)
        return jsonify({
            'success': True,
            'message': 'Cache cleaned up'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@api_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({'success': False, 'error': 'Method not allowed'}), 405


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500
