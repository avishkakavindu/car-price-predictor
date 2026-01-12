"""
Flask Application Entry Point for Car Price Prediction API
"""

from flask import Flask
from flask_cors import CORS
import os
import sys

# Add backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_factory import ModelFactory
from models.ensemble_predictor import EnsemblePredictor
from services.shap_service import ShapService
from api.routes import api_bp, init_routes


def create_app():
    """
    Application factory for creating Flask app

    This function:
        1. Initializes Flask app
        2. Configures CORS
        3. Loads ML models via ModelFactory
        4. Initializes services (SHAP, Ensemble)
        5. Registers API routes
        6. Sets up error handlers

    Returns:
        Flask app instance
    """
    print("\n" + "=" * 70)
    print("INITIALIZING CAR PRICE PREDICTION API")
    print("=" * 70 + "\n")

    # Create Flask app
    app = Flask(__name__)

    # Configuration
    app.config["JSON_SORT_KEYS"] = False  # Preserve JSON key order
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max request size

    # CORS configuration - allow all origins for development
    # In production, restrict to specific origins
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": "*",  # Change to specific domains in production
                "methods": ["GET", "POST", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type"],
            }
        },
    )

    # Get base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Initialize Model Factory
    model_factory = ModelFactory(base_path=base_path)

    if model_factory.get_model_count() == 0:
        print("\n[WARNING]  WARNING: No models loaded!")
        print("Please ensure model files exist in backend/data/models/")
    else:
        print(
            f"\n[OK] Model Factory initialized with {model_factory.get_model_count()} models"
        )

    # Initialize SHAP Service
    print("Initializing SHAP Service...")
    shap_service = ShapService(max_workers=2)
    print("SHAP Service initialized (2 concurrent workers)")

    # Initialize Ensemble Predictor
    print("Initializing Ensemble Predictor...")
    ensemble_predictor = EnsemblePredictor(model_factory)
    print(
        f"[OK] Ensemble Predictor initialized with {model_factory.get_model_count()} models"
    )

    # Initialize routes with dependencies
    print("Registering API Routes...")
    init_routes(model_factory, shap_service, ensemble_predictor)
    app.register_blueprint(api_bp)
    print("[OK] API routes registered at /api/*")

    # Register error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return {"success": False, "error": "Page not found"}, 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return {"success": False, "error": "Internal server error"}, 500

    # Root endpoint
    @app.route("/")
    def index():
        return {
            "message": "Car Price Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/api/health",
                "models": "/api/models",
                "predict": "POST /api/predict",
                "predict_single": "POST /api/predict/<model_name>",
                "shap_generate": "POST /api/shap/generate?model=<model_name>",
                "shap_status": "GET /api/shap/status/<request_id>",
            },
            "models_loaded": model_factory.get_model_count(),
            "available_models": model_factory.get_available_models(),
        }

    print("[OK] APPLICATION INITIALIZED SUCCESSFULLY")
    print(f"Models loaded: {model_factory.get_model_count()}")
    print(f"Available models: {', '.join(model_factory.get_available_models())}")

    return app


if __name__ == "__main__":
    app = create_app()

    print("\n")
    print("Starting Flask Development Server...")
    print("Server: http://localhost:5000")
    print("API Endpoints: http://localhost:5000/api/")
    print("Health Check: http://localhost:5000/api/health")
    print("\nPress CTRL+C to stop the server")

    # Run development server
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        threaded=True,  # Enable threading for concurrent requests
    )
