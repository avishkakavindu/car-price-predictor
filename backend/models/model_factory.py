"""
Model Factory for creating and managing model instances (Factory Pattern)
"""

from typing import Dict, Optional, List
from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .adaboost_model import AdaBoostModel


class ModelFactory:
    """
    Factory for creating and managing model instances

    This class implements the Factory Pattern to:
        1. Centralize model creation
        2. Make it easy to add new models
        3. Handle model loading errors gracefully
        4. Provide a consistent interface for all models

    To add a new model:
        1. Create a new model class that extends BaseModel
        2. Add it to MODEL_REGISTRY
        3. Import it at the top of this file
        That's it! The system will automatically load and use it.
    """

    # Model registry: maps model names to their classes and file paths
    # This is the only place you need to modify to add a new model
    MODEL_REGISTRY = {
        "xgboost": {
            "class": XGBoostModel,
            "path": "data/models/car_price_predictor_xgbr.pkl",
            "description": "XGBoost gradient boosting model",
        },
        "lightgbm": {
            "class": LightGBMModel,
            "path": "data/models/car_price_predictor_lgbm.pkl",
            "description": "LightGBM gradient boosting model",
        },
        "adaboost": {
            "class": AdaBoostModel,
            "path": "data/models/car_price_predictor_adaboost.pkl",
            "description": "AdaBoost ensemble model",
        },
    }

    def __init__(self, base_path: str = ""):
        """
        Initialize factory with base path for model files

        Args:
            base_path: Base directory path for model files (default: current directory)
        """
        self.base_path = base_path
        self._models: Dict[str, BaseModel] = {}
        self._load_all_models()

    def _load_all_models(self):
        """
        Load all available models from registry

        This method:
            1. Iterates through MODEL_REGISTRY
            2. Attempts to load each model
            3. Stores successfully loaded models
            4. Continues even if some models fail to load

        This ensures the system is robust and continues working
        even if some models are missing or fail to load.
        """
        print("Loading models...")

        for model_name, model_info in self.MODEL_REGISTRY.items():
            try:
                # Construct full path
                if self.base_path:
                    model_path = f"{self.base_path}/{model_info['path']}"
                else:
                    model_path = model_info["path"]

                # Create model instance
                model_class = model_info["class"]
                model_instance = model_class(model_path)

                # Attempt to load
                if model_instance.load_model():
                    self._models[model_name] = model_instance
                    print(f"[OK] Loaded: {model_name} - {model_info['description']}")
                else:
                    print(f"[FAIL] Failed to load: {model_name}")

            except Exception as e:
                print(f"[ERROR] Error loading {model_name}: {str(e)}")

        print(
            f"Successfully loaded {len(self._models)} out of {len(self.MODEL_REGISTRY)} models"
        )

    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """
        Get a specific model by name

        Args:
            model_name: Name of the model (e.g., 'xgboost', 'random_forest')

        Returns:
            BaseModel instance if found, None otherwise
        """
        return self._models.get(model_name.lower())

    def get_all_models(self) -> Dict[str, BaseModel]:
        """
        Get all loaded models

        Returns:
            Dictionary mapping model names to BaseModel instances
        """
        return self._models.copy()

    def get_available_models(self) -> List[str]:
        """
        Get list of available model names

        Returns:
            List of model names (e.g., ['xgboost', 'random_forest'])
        """
        return list(self._models.keys())

    def get_models_info(self) -> List[Dict[str, any]]:
        """
        Get information about all loaded models

        Returns:
            List of dictionaries containing model metadata
        """
        return [model.get_model_info() for model in self._models.values()]

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available

        Args:
            model_name: Name of the model

        Returns:
            bool: True if model is loaded and available, False otherwise
        """
        model = self.get_model(model_name)
        return model is not None and model.is_model_loaded()

    def get_model_count(self) -> int:
        """
        Get count of loaded models

        Returns:
            int: Number of successfully loaded models
        """
        return len(self._models)

    def reload_model(self, model_name: str) -> bool:
        """
        Reload a specific model

        Useful for hot-reloading models after training updates

        Args:
            model_name: Name of the model to reload

        Returns:
            bool: True if reload successful, False otherwise
        """
        if model_name not in self.MODEL_REGISTRY:
            print(f"Model {model_name} not found in registry")
            return False

        try:
            model_info = self.MODEL_REGISTRY[model_name]

            # Construct path
            if self.base_path:
                model_path = f"{self.base_path}/{model_info['path']}"
            else:
                model_path = model_info["path"]

            # Create and load new instance
            model_class = model_info["class"]
            model_instance = model_class(model_path)

            if model_instance.load_model():
                self._models[model_name] = model_instance
                print(f"✓ Reloaded: {model_name}")
                return True
            else:
                print(f"✗ Failed to reload: {model_name}")
                return False

        except Exception as e:
            print(f"✗ Error reloading {model_name}: {str(e)}")
            return False

    def __repr__(self) -> str:
        """String representation"""
        return f"ModelFactory(models={list(self._models.keys())}, count={len(self._models)})"
