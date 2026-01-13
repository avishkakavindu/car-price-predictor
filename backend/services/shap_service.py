"""
SHAP Service for generating model explanations asynchronously
"""
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any, Optional
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading


class ShapService:
    """
    Service for generating SHAP explanations asynchronously

    SHAP (SHapley Additive exPlanations) provides:
        - Feature importance (which features matter most)
        - Direction of impact (does feature increase/decrease prediction)
        - Local explanations (why this specific prediction)
        - Global explanations (general model behavior)

    This service runs SHAP generation in background threads to avoid
    blocking the API, since SHAP computation can take several seconds.
    """

    def __init__(self, max_workers: int = 2):
        """
        Initialize SHAP service

        Args:
            max_workers: Maximum number of concurrent SHAP generation threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def generate_shap_async(self,
                           model,
                           input_data: pd.DataFrame,
                           request_id: Optional[str] = None) -> str:
        """
        Start async SHAP generation

        Args:
            model: The pipeline model (includes preprocessing)
            input_data: Raw input DataFrame (14 features)
            request_id: Optional UUID for tracking (generated if not provided)

        Returns:
            request_id: UUID for tracking the SHAP generation
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Initialize cache entry
        with self._lock:
            self._cache[request_id] = {
                'status': 'processing',
                'progress': 0
            }

        # Submit to thread pool
        self.executor.submit(self._generate_shap, model, input_data, request_id)

        return request_id

    def _generate_shap(self, model, input_data: pd.DataFrame, request_id: str):
        """
        Internal method to generate SHAP explanations

        This runs in a background thread

        Args:
            model: The pipeline model
            input_data: Raw input DataFrame
            request_id: UUID for this request
        """
        try:
            # Update progress
            self._update_cache(request_id, {'status': 'processing', 'progress': 10})

            # Get the actual XGBoost model (unwrap from pipeline)
            xgb_model = model.named_steps['model']

            # Transform input data through preprocessing steps
            X_processed = model.named_steps['preprocessing'].transform(input_data)
            X_processed = model.named_steps['column_transform'].transform(X_processed)

            self._update_cache(request_id, {'status': 'processing', 'progress': 30})

            # Get feature names
            feature_names = self._get_feature_names(model)
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

            # Create SHAP explainer (TreeExplainer for XGBoost)
            print(f"Creating SHAP explainer for request {request_id}...")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_processed_df)

            self._update_cache(request_id, {'status': 'processing', 'progress': 60})

            # Generate visualizations
            print(f"Generating SHAP visualizations for request {request_id}...")
            bar_chart = self._generate_bar_chart(shap_values, X_processed_df, feature_names)
            waterfall = self._generate_waterfall(
                shap_values[0],
                explainer.expected_value,
                X_processed_df.iloc[0],
                feature_names
            )

            self._update_cache(request_id, {'status': 'processing', 'progress': 90})

            # Compute feature importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = [
                {
                    'feature': feature_names[i],
                    'importance': float(mean_abs_shap[i])
                }
                for i in range(len(feature_names))
            ]
            # Sort by importance (descending)
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)

            # Get individual SHAP values for this specific prediction
            shap_values_for_instance = [
                {
                    'feature': feature_names[i],
                    'shap_value': float(shap_values[0][i]),
                    'feature_value': float(X_processed_df.iloc[0, i])
                }
                for i in range(len(feature_names))
            ]
            # Sort by absolute SHAP value (most impactful first)
            shap_values_for_instance.sort(key=lambda x: abs(x['shap_value']), reverse=True)

            # Update cache with results
            result = {
                'status': 'completed',
                'progress': 100,
                'bar_chart': bar_chart,
                'waterfall': waterfall,
                'feature_importance': feature_importance[:10],  # Top 10
                'base_value': float(explainer.expected_value),
                'shap_values': shap_values_for_instance  # Individual SHAP values for this prediction
            }

            self._update_cache(request_id, result)
            print(f"SHAP generation completed for request {request_id}")

        except Exception as e:
            error_msg = str(e)
            print(f"Error generating SHAP for request {request_id}: {error_msg}")
            self._update_cache(request_id, {
                'status': 'error',
                'error': error_msg
            })

    def _get_feature_names(self, model) -> list:
        """
        Extract feature names from pipeline

        Args:
            model: The pipeline model

        Returns:
            List of feature names after transformation
        """
        feature_names = []

        # Numerical features (scaled)
        feature_names.extend(['Engine (cc)', 'Mileage_log'])

        # Binary features (passthrough)
        binary_features = [
            'Gear', 'Leasing', 'Condition',
            'AIR CONDITION', 'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW'
        ]
        feature_names.extend(binary_features)

        # Categorical features (one-hot encoded)
        ct = model.named_steps['column_transform']
        cat_transformer = ct.named_transformers_['cat']
        cat_features = cat_transformer.get_feature_names_out(['Brand_Grouped', 'Fuel_Grouped'])
        feature_names.extend(cat_features)

        return feature_names

    def _generate_bar_chart(self, shap_values, X, feature_names) -> str:
        """
        Generate SHAP feature importance bar chart as base64 image

        Args:
            shap_values: SHAP values array
            X: Processed input data
            feature_names: List of feature names

        Returns:
            str: Base64 encoded PNG image with data URI prefix
        """
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Mean |SHAP value| (average impact on model output)', fontsize=11)
        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def _generate_waterfall(self, shap_values, base_value, instance_data, feature_names) -> str:
        """
        Generate SHAP waterfall plot as base64 image

        Waterfall plot shows how each feature contributed to pushing
        the prediction from the base value (average) to the final value

        Args:
            shap_values: SHAP values for single instance
            base_value: Expected value (average prediction)
            instance_data: Feature values for this instance
            feature_names: List of feature names

        Returns:
            str: Base64 encoded PNG image with data URI prefix
        """
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=instance_data.values,
            feature_names=feature_names
        )

        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title('SHAP Waterfall Plot - Prediction Breakdown', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def get_shap_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of SHAP generation

        Args:
            request_id: UUID of the SHAP request

        Returns:
            Dict with status and results (if completed), None if not found
        """
        with self._lock:
            return self._cache.get(request_id)

    def _update_cache(self, request_id: str, data: Dict[str, Any]):
        """
        Update cache with thread safety

        Args:
            request_id: UUID of the request
            data: Data to update
        """
        with self._lock:
            if request_id in self._cache:
                self._cache[request_id].update(data)
            else:
                self._cache[request_id] = data

    def cleanup_cache(self, request_id: str):
        """
        Remove completed SHAP from cache

        Call this after the client has retrieved the results
        to free up memory

        Args:
            request_id: UUID of the request to clean up
        """
        with self._lock:
            if request_id in self._cache:
                del self._cache[request_id]

    def get_cache_size(self) -> int:
        """
        Get number of cached SHAP results

        Returns:
            int: Number of items in cache
        """
        with self._lock:
            return len(self._cache)

    def cleanup_old_cache(self, max_age_seconds: int = 3600):
        """
        Clean up cache entries older than max_age_seconds

        For future implementation with timestamps

        Args:
            max_age_seconds: Maximum age before cleanup (default: 1 hour)
        """
        # TODO: Implement timestamp tracking and cleanup
        pass

    def shutdown(self):
        """Shutdown the thread pool executor"""
        self.executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except Exception:
            pass
