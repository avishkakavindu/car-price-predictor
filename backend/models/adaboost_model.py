"""
AdaBoost model implementation following Strategy Pattern
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .base_model import BaseModel

# Import preprocessing functions so they're available when unpickling
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services import preprocessing

# Make functions available in __main__ namespace for pickle
sys.modules["__main__"].create_car_age = preprocessing.create_car_age
sys.modules["__main__"].create_log_mileage = preprocessing.create_log_mileage
sys.modules["__main__"].group_brands = preprocessing.group_brands
sys.modules["__main__"].group_fuel = preprocessing.group_fuel
sys.modules["__main__"].encode_binary_features = preprocessing.encode_binary_features
sys.modules["__main__"].select_features = preprocessing.select_features


class AdaBoostModel(BaseModel):
    """AdaBoost implementation of BaseModel"""

    def __init__(self, model_path: str):
        """
        Initialize AdaBoost model

        Args:
            model_path: Path to the saved pipeline (.pkl file)
        """
        super().__init__(model_name="AdaBoost", model_path=model_path)

    def load_model(self) -> bool:
        """
        Load AdaBoost pipeline from pickle file

        The pipeline contains:
            - preprocessing steps (feature engineering)
            - column transformer (scaling, encoding)
            - AdaBoost model

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            print(f"[OK] {self.model_name} loaded successfully from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"[ERROR] {self.model_name} file not found: {self.model_path}")
            self.is_loaded = False
            return False
        except Exception as e:
            print(f"[ERROR] Error loading {self.model_name}: {str(e)}")
            self.is_loaded = False
            return False

    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction using AdaBoost model

        The pipeline handles all preprocessing:
            1. Creates Car_Age from YOM
            2. Creates Mileage_log from Millage(KM)
            3. Groups brands into top 15 + Other
            4. Groups fuel types
            5. Encodes binary features
            6. Scales numerical features
            7. One-hot encodes categorical features
            8. Makes prediction

        Args:
            input_data: DataFrame with raw 14 features

        Returns:
            Dict with prediction results or error
        """
        if not self.is_loaded:
            return {
                "model_name": self.model_name,
                "error": "Model is not loaded",
                "status": "error",
            }

        try:
            # Model expects raw input - pipeline handles preprocessing
            price_log = self.model.predict(input_data)[0]

            # Convert from log scale to actual price in lakhs
            price_lakhs = np.expm1(price_log)

            return {
                "model_name": self.model_name,
                "prediction_log": float(price_log),
                "prediction_lakhs": float(price_lakhs),
                "status": "success",
            }

        except Exception as e:
            return {
                "model_name": self.model_name,
                "error": f"Prediction failed: {str(e)}",
                "status": "error",
            }

    def get_feature_names(self) -> List[str]:
        """
        Get feature names from the pipeline after transformation

        Returns:
            List of transformed feature names
        """
        if not self.is_loaded:
            return []

        try:
            # Extract column transformer from pipeline
            ct = self.model.named_steps["column_transform"]

            feature_names = []

            # 1. Add numerical features (scaled)
            feature_names.extend(["Engine (cc)", "Mileage_log"])

            # 2. Add binary features (passthrough - already 0/1)
            binary_features = [
                "Gear",
                "Leasing",
                "Condition",
                "AIR CONDITION",
                "POWER STEERING",
                "POWER MIRROR",
                "POWER WINDOW",
            ]
            feature_names.extend(binary_features)

            # 3. Add categorical features (one-hot encoded)
            cat_transformer = ct.named_transformers_["cat"]
            cat_features = cat_transformer.get_feature_names_out(
                ["Brand_Grouped", "Fuel_Grouped"]
            )
            feature_names.extend(cat_features)

            return feature_names

        except Exception as e:
            print(f"Error getting feature names: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration

        Returns:
            Dict with model information
        """
        info = {
            "model_name": self.model_name,
            "algorithm": "AdaBoost Regressor",
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "features": 0,
        }

        if self.is_loaded:
            try:
                info["features"] = len(self.get_feature_names())

                # Get AdaBoost-specific parameters if available
                ada_model = self.model.named_steps.get("model")
                if ada_model:
                    info["hyperparameters"] = {
                        "n_estimators": getattr(ada_model, "n_estimators", None),
                        "learning_rate": getattr(ada_model, "learning_rate", None),
                    }
            except Exception as e:
                print(f"Error getting model info: {e}")

        return info

    def get_pipeline_steps(self) -> List[str]:
        """
        Get the names of all steps in the pipeline

        Returns:
            List of pipeline step names
        """
        if not self.is_loaded:
            return []

        try:
            return [name for name, _ in self.model.steps]
        except Exception:
            return []
