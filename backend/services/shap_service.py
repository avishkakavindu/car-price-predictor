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
import json
from typing import Dict, Any, Optional
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
import traceback


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

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Convert value to float, handling strings and arrays."""
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0] if len(value) > 0 else default

        if isinstance(value, str):
            value = value.strip('[]')

        try:
            return float(value)
        except (ValueError, TypeError):
            print(f"[SHAP DEBUG] _safe_float fallback for value={value} (type={type(value)})")
            return default

    def _sanitize_xgb_model(self, model):
        """Ensure booster attributes (like base_score) are valid floats."""
        try:
            booster = model.get_booster()
            config = json.loads(booster.save_config())
            learner_params = config.get('learner', {}).get('learner_model_param', {})
            base_score = learner_params.get('base_score')
            if base_score is not None:
                clean_score = base_score.strip('[]')
                try:
                    clean_float = float(clean_score)
                    learner_params['base_score'] = str(clean_float)
                    booster.load_config(json.dumps(config))
                    print(f"[SHAP DEBUG] Sanitized base_score from {base_score} to {clean_float}")
                except ValueError:
                    learner_params['base_score'] = '0.0'
                    booster.load_config(json.dumps(config))
                    print(f"[SHAP DEBUG] Invalid base_score '{base_score}', forced to 0.0")
        except Exception as exc:
            print(f"[SHAP DEBUG] Failed to sanitize XGBoost model attributes: {exc}")
        return model

    def _debug_value(self, label: str, value: Any, sample_size: int = 3):
        """Print debug info about intermediate values."""
        try:
            info = {
                'type': type(value).__name__
            }

            if isinstance(value, np.ndarray):
                info['shape'] = value.shape
                info['dtype'] = str(value.dtype)
                info['sample'] = value.flatten()[:sample_size].tolist()
            elif isinstance(value, pd.DataFrame):
                info['shape'] = value.shape
                info['dtypes'] = {c: str(dt) for c, dt in value.dtypes.items()}
                info['sample'] = value.head(sample_size).to_dict(orient='records')
            elif isinstance(value, pd.Series):
                info['shape'] = value.shape
                info['dtype'] = str(value.dtype)
                info['sample'] = value.head(sample_size).tolist()
            elif isinstance(value, list):
                info['length'] = len(value)
                info['sample'] = value[:sample_size]
            elif isinstance(value, dict):
                keys = list(value.keys())[:sample_size]
                info['keys'] = keys
                info['sample'] = {k: value[k] for k in keys}
            else:
                info['value'] = value

            print(f"[SHAP DEBUG] {label}: {info}")
        except Exception as debug_exc:
            print(f"[SHAP DEBUG] Failed to log {label}: {debug_exc}")

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
            self._debug_value('After preprocessing.transform', X_processed)
            X_processed = model.named_steps['column_transform'].transform(X_processed)
            self._debug_value('After column_transform.transform', X_processed)

            self._update_cache(request_id, {'status': 'processing', 'progress': 30})

            # Get feature names
            feature_names = self._get_feature_names(model)
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
            self._debug_value('X_processed_df (raw)', X_processed_df)
            
            # Ensure all columns are numeric (handle any string representations)
            for col in X_processed_df.columns:
                try:
                    # Try to convert string representations to float
                    X_processed_df[col] = X_processed_df[col].astype(str).str.strip('[]').astype(float)
                except (ValueError, TypeError):
                    # If conversion fails, try to convert directly
                    try:
                        X_processed_df[col] = pd.to_numeric(X_processed_df[col], errors='coerce')
                    except Exception:
                        pass  # Keep original values if all conversions fail
            self._debug_value('X_processed_df (numeric enforced)', X_processed_df)

            # Convert to numpy array for SHAP computations
            X_shap_array = X_processed_df.to_numpy(dtype=np.float64, copy=True)
            self._debug_value('X_shap_array', X_shap_array)

            # Create SHAP explainer (TreeExplainer for XGBoost)
            print(f"Creating SHAP explainer for request {request_id}...")
            sanitized_model = self._sanitize_xgb_model(xgb_model)
            explainer = shap.TreeExplainer(sanitized_model)
            shap_values = explainer.shap_values(X_shap_array)
            self._debug_value('shap_values', shap_values)
            self._debug_value('explainer.expected_value', explainer.expected_value)

            self._update_cache(request_id, {'status': 'processing', 'progress': 60})

            # Generate visualizations
            print(f"Generating SHAP visualizations for request {request_id}...")
            bar_chart = self._generate_bar_chart(shap_values, X_processed_df, feature_names)
            base_value = self._safe_float(explainer.expected_value)
            waterfall = self._generate_waterfall(
                shap_values[0],
                base_value,
                X_processed_df.iloc[0],
                feature_names
            )

            self._update_cache(request_id, {'status': 'processing', 'progress': 90})

            # Compute feature importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            self._debug_value('mean_abs_shap', mean_abs_shap)
            feature_importance = []
            for i in range(len(feature_names)):
                importance_val = self._safe_float(mean_abs_shap[i])
                feature_importance.append({
                    'feature': feature_names[i],
                    'importance': importance_val
                })
            # Sort by importance (descending)
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            self._debug_value('feature_importance (sorted)', feature_importance)

            # Get individual SHAP values for this specific prediction
            shap_values_for_instance = []
            for i in range(len(feature_names)):
                shap_val = self._safe_float(shap_values[0][i])
                feature_val = self._safe_float(X_processed_df.iloc[0, i])

                shap_values_for_instance.append({
                    'feature': feature_names[i],
                    'shap_value': shap_val,
                    'feature_value': feature_val
                })
            # Sort by absolute SHAP value (most impactful first)
            shap_values_for_instance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            self._debug_value('shap_values_for_instance (sorted)', shap_values_for_instance)

            # Update cache with results
            result = {
                'status': 'completed',
                'progress': 100,
                'bar_chart': bar_chart,
                'waterfall': waterfall,
                'feature_importance': feature_importance[:10],  # Top 10
                'base_value': base_value,
                'shap_values': shap_values_for_instance  # Individual SHAP values for this prediction
            }
            self._debug_value('Final SHAP result payload', result)

            self._update_cache(request_id, result)
            print(f"SHAP generation completed for request {request_id}")

        except Exception as e:
            error_msg = str(e)
            print(f"Error generating SHAP for request {request_id}: {error_msg}")
            traceback.print_exc()
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
