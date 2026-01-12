"""
Configuration management for Flask application
"""
import os


class Config:
    """Base configuration"""

    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Model paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models')
    DATASET_DIR = os.path.join(BASE_DIR, 'data', 'datasets')

    # SHAP configuration
    SHAP_MAX_WORKERS = 2
    SHAP_CACHE_TTL = 3600  # 1 hour in seconds

    # API configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    JSON_SORT_KEYS = False

    # CORS configuration
    CORS_ORIGINS = "*"  # Change to specific domains in production

    @staticmethod
    def get_model_path(model_name: str) -> str:
        """
        Get full path for a model file

        Args:
            model_name: Model identifier (e.g., 'xgbr', 'rf', 'lgbm')

        Returns:
            str: Full path to model file
        """
        return os.path.join(Config.MODEL_DIR, f"car_price_predictor_{model_name}.pkl")


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    # Override CORS for production
    CORS_ORIGINS = [
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ]


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
