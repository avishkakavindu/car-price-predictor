"""
Abstract base class for ML models (Strategy Pattern)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all ML models using Strategy Pattern"""

    def __init__(self, model_name: str, model_path: str):
        """
        Initialize base model

        Args:
            model_name: Human-readable name of the model (e.g., "XGBoost", "Random Forest")
            model_path: Path to the saved model file (.pkl)
        """
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the model from disk

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction on input data

        Args:
            input_data: DataFrame with raw input features (14 columns)

        Returns:
            Dict containing:
                - model_name (str): Name of the model
                - prediction_log (float): Prediction in log scale
                - prediction_lakhs (float): Prediction in lakhs (actual price)
                - status (str): 'success' or 'error'
                - error (str, optional): Error message if status is 'error'
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names used by the model after transformation

        Returns:
            List of feature names (e.g., ['Engine (cc)', 'Mileage_log', 'Gear', ...])
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and information

        Returns:
            Dict containing:
                - model_name (str): Name of the model
                - algorithm (str): Algorithm description
                - is_loaded (bool): Whether model is loaded
                - model_path (str): Path to model file
                - features (int): Number of features
        """
        pass

    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded and ready for predictions

        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.is_loaded

    def __repr__(self) -> str:
        """String representation of the model"""
        return f"{self.__class__.__name__}(name='{self.model_name}', loaded={self.is_loaded})"
