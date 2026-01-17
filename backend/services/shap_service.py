"""
SHAP Service for generating model explanations asynchronously with caching
"""
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any, Optional, Tuple
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib
import time


class ShapService:
    """
    Service for generating SHAP explanations asynchronously with caching

    SHAP (SHapley Additive exPlanations) provides:
        - Feature importance (which features matter most)
        - Direction of impact (does feature increase/decrease prediction)
        - Local explanations (why this specific prediction)
        - Global explanations (general model behavior)

    This service runs SHAP generation in background threads to avoid
    blocking the API, since SHAP computation can take several seconds.

    Features:
        - Async SHAP computation with progress tracking
        - Results caching with TTL to avoid recomputation
        - Support for multiple model types (TreeExplainer, KernelExplainer)

    Supports multiple model types:
        - TreeExplainer: For tree-based models (XGBoost, LightGBM)
        - KernelExplainer: For model-agnostic explanations (CatBoost)
    """

    # Models that use TreeExplainer (fast, tree-based)
    TREE_BASED_MODELS = ['xgboost', 'lightgbm']
    # Models that require KernelExplainer (slower, model-agnostic)
    KERNEL_BASED_MODELS = ['catboost']

    # Cache configuration
    DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
    MAX_CACHE_SIZE = 100  # Maximum number of cached results

    def __init__(self, max_workers: int = 2, cache_ttl: int = None, max_cache_size: int = None):
        """
        Initialize SHAP service with caching

        Args:
            max_workers: Maximum number of concurrent SHAP generation threads
            cache_ttl: Time-to-live for cached results in seconds (default: 1 hour)
            max_cache_size: Maximum number of results to cache (default: 100)
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._request_cache: Dict[str, Dict[str, Any]] = {}  # In-progress request tracking
        self._results_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}  # (result, timestamp)
        self._lock = threading.Lock()
        self._results_lock = threading.Lock()
        self.cache_ttl = cache_ttl or self.DEFAULT_CACHE_TTL
        self.max_cache_size = max_cache_size or self.MAX_CACHE_SIZE

    def _generate_cache_key(self, input_data: pd.DataFrame, model_name: str) -> str:
        """
        Generate a unique cache key from input data and model name

        Args:
            input_data: Raw input DataFrame
            model_name: Name of the model

        Returns:
            str: Hash-based cache key
        """
        # Convert input data to a consistent string representation
        data_str = input_data.to_json(orient='records')
        key_str = f"{model_name}:{data_str}"

        # Generate MD5 hash for consistent key length
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached SHAP result if available and not expired

        Args:
            cache_key: The cache key to look up

        Returns:
            Cached result dict if valid, None if not found or expired
        """
        with self._results_lock:
            if cache_key in self._results_cache:
                result, timestamp = self._results_cache[cache_key]
                # Check if cache entry is still valid
                if time.time() - timestamp < self.cache_ttl:
                    print(f"[CACHE HIT] Returning cached SHAP result for key {cache_key[:8]}...")
                    return result
                else:
                    # Expired, remove from cache
                    print(f"[CACHE EXPIRED] Removing expired entry for key {cache_key[:8]}...")
                    del self._results_cache[cache_key]
            return None

    def _store_cached_result(self, cache_key: str, result: Dict[str, Any]):
        """
        Store SHAP result in cache with timestamp

        Args:
            cache_key: The cache key
            result: The SHAP result to cache
        """
        with self._results_lock:
            # Enforce max cache size (LRU-style: remove oldest entries)
            if len(self._results_cache) >= self.max_cache_size:
                # Remove oldest entries (first 10% of max size)
                entries_to_remove = max(1, self.max_cache_size // 10)
                sorted_keys = sorted(
                    self._results_cache.keys(),
                    key=lambda k: self._results_cache[k][1]  # Sort by timestamp
                )
                for key in sorted_keys[:entries_to_remove]:
                    del self._results_cache[key]
                print(f"[CACHE CLEANUP] Removed {entries_to_remove} oldest entries")

            # Store new result with timestamp
            self._results_cache[cache_key] = (result, time.time())
            print(f"[CACHE STORE] Stored result for key {cache_key[:8]}... (cache size: {len(self._results_cache)})")

    def generate_shap_async(self,
                           model,
                           input_data: pd.DataFrame,
                           request_id: Optional[str] = None,
                           model_name: str = 'xgboost') -> Tuple[str, bool]:
        """
        Start async SHAP generation with caching

        Args:
            model: The pipeline model (includes preprocessing)
            input_data: Raw input DataFrame (14 features)
            request_id: Optional UUID for tracking (generated if not provided)
            model_name: Name of the model (xgboost, lightgbm, catboost)

        Returns:
            Tuple[str, bool]: (request_id, is_cached)
                - request_id: UUID for tracking the SHAP generation
                - is_cached: True if result was returned from cache (ready immediately)
        """
        # Generate cache key from input data and model
        cache_key = self._generate_cache_key(input_data, model_name)

        # Check for cached result
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            # Return cached result immediately
            if request_id is None:
                request_id = str(uuid.uuid4())

            # Store cached result in request cache for retrieval
            with self._lock:
                self._request_cache[request_id] = cached_result.copy()
                self._request_cache[request_id]['from_cache'] = True

            return request_id, True

        # No cache hit - generate new SHAP values
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Initialize request cache entry
        with self._lock:
            self._request_cache[request_id] = {
                'status': 'processing',
                'progress': 0,
                'model_name': model_name,
                'cache_key': cache_key  # Store cache key for later storage
            }

        # Submit to thread pool
        self.executor.submit(self._generate_shap, model, input_data, request_id, model_name, cache_key)

        return request_id, False

    def _generate_shap(self, model, input_data: pd.DataFrame, request_id: str, model_name: str = 'xgboost', cache_key: str = None):
        """
        Internal method to generate SHAP explanations

        This runs in a background thread

        Args:
            model: The pipeline model
            input_data: Raw input DataFrame
            request_id: UUID for this request
            model_name: Name of the model (xgboost, lightgbm, catboost)
            cache_key: Cache key for storing result (optional)
        """
        try:
            # Update progress
            self._update_request_cache(request_id, {'status': 'processing', 'progress': 10})

            # Get the actual model (unwrap from pipeline)
            ml_model = model.named_steps['model']

            # Transform input data through preprocessing steps
            X_processed = model.named_steps['preprocessing'].transform(input_data)
            X_processed = model.named_steps['column_transform'].transform(X_processed)

            self._update_request_cache(request_id, {'status': 'processing', 'progress': 30})

            # Get feature names
            feature_names = self._get_feature_names(model)
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

            # Create appropriate SHAP explainer based on model type
            print(f"Creating SHAP explainer for {model_name} (request {request_id})...")

            if model_name.lower() in self.TREE_BASED_MODELS:
                # Use TreeExplainer for tree-based models (faster)
                explainer = shap.TreeExplainer(ml_model)
                shap_values = explainer.shap_values(X_processed_df)
            else:
                # Use KernelExplainer for non-tree models (CatBoost)
                # KernelExplainer requires a background dataset
                print(f"Using KernelExplainer for {model_name} (this may take longer)...")

                # Create synthetic background data for proper SHAP computation
                # When we only have 1 sample, we need to create variations to establish a baseline
                background = self._create_background_data(X_processed_df, feature_names)

                explainer = shap.KernelExplainer(ml_model.predict, background)
                shap_values = explainer.shap_values(X_processed_df, nsamples=100)

            self._update_request_cache(request_id, {'status': 'processing', 'progress': 60})

            # Generate visualizations
            print(f"Generating SHAP visualizations for request {request_id}...")
            bar_chart = self._generate_bar_chart(shap_values, X_processed_df, feature_names)
            waterfall = self._generate_waterfall(
                shap_values[0],
                explainer.expected_value,
                X_processed_df.iloc[0],
                feature_names
            )

            self._update_request_cache(request_id, {'status': 'processing', 'progress': 90})

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

            # Handle expected_value (may be array for some explainers)
            expected_value = explainer.expected_value
            if hasattr(expected_value, '__len__') and not isinstance(expected_value, str):
                expected_value = float(expected_value[0]) if len(expected_value) > 0 else float(expected_value)
            else:
                expected_value = float(expected_value)

            # Update request cache with results
            result = {
                'status': 'completed',
                'progress': 100,
                'model_name': model_name,
                'bar_chart': bar_chart,
                'waterfall': waterfall,
                'feature_importance': feature_importance[:10],  # Top 10
                'base_value': expected_value,
                'shap_values': shap_values_for_instance  # Individual SHAP values for this prediction
            }

            self._update_request_cache(request_id, result)

            # Store in results cache for future lookups
            if cache_key:
                self._store_cached_result(cache_key, result)

            print(f"SHAP generation completed for {model_name} (request {request_id})")

        except Exception as e:
            error_msg = str(e)
            print(f"Error generating SHAP for request {request_id}: {error_msg}")
            self._update_request_cache(request_id, {
                'status': 'error',
                'error': error_msg
            })

    def _create_background_data(self, X_processed_df: pd.DataFrame, feature_names: list, n_samples: int = 50) -> pd.DataFrame:
        """
        Create synthetic background data for KernelExplainer

        KernelExplainer requires a diverse background dataset to compute
        meaningful SHAP values. When we only have a single prediction,
        we need to create synthetic variations around the input data.

        Args:
            X_processed_df: The processed input data (single row)
            feature_names: List of feature names
            n_samples: Number of background samples to generate

        Returns:
            pd.DataFrame: Background dataset for SHAP computation
        """
        # Get the single input row
        input_row = X_processed_df.iloc[0].values

        # Create background by generating variations
        background_data = []

        # Identify numerical vs binary/categorical features
        numerical_features = ['Engine (cc)', 'Mileage_log']
        binary_features = ['Gear', 'Leasing', 'Condition', 'AIR CONDITION',
                          'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW']

        for i in range(n_samples):
            new_row = input_row.copy()
            for j, fname in enumerate(feature_names):
                if fname in numerical_features:
                    # Add random noise to numerical features (scaled data, so use small noise)
                    # Use larger variation for better SHAP computation
                    noise = np.random.normal(0, 0.5)
                    new_row[j] = input_row[j] + noise
                elif fname in binary_features:
                    # Randomly flip binary features with some probability
                    if np.random.random() < 0.3:
                        new_row[j] = 1 - input_row[j] if input_row[j] in [0, 1] else input_row[j]
                elif fname.startswith('Brand_Grouped_') or fname.startswith('Fuel_Grouped_'):
                    # For one-hot encoded features, randomly set different categories
                    if np.random.random() < 0.2:
                        new_row[j] = 1 - new_row[j] if new_row[j] in [0, 1] else new_row[j]
            background_data.append(new_row)

        # Also add some samples with mean/zero values for better baseline
        zero_row = np.zeros_like(input_row)
        mean_row = input_row * 0.5
        background_data.extend([zero_row, mean_row])

        # Add the original input as well for reference
        background_data.append(input_row)

        background_df = pd.DataFrame(background_data, columns=feature_names)
        return background_df

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
            return self._request_cache.get(request_id)

    def _update_request_cache(self, request_id: str, data: Dict[str, Any]):
        """
        Update request cache with thread safety

        Args:
            request_id: UUID of the request
            data: Data to update
        """
        with self._lock:
            if request_id in self._request_cache:
                self._request_cache[request_id].update(data)
            else:
                self._request_cache[request_id] = data

    def cleanup_request_cache(self, request_id: str):
        """
        Remove completed SHAP from request cache

        Call this after the client has retrieved the results
        to free up memory

        Args:
            request_id: UUID of the request to clean up
        """
        with self._lock:
            if request_id in self._request_cache:
                del self._request_cache[request_id]

    def get_request_cache_size(self) -> int:
        """
        Get number of in-progress/completed requests in cache

        Returns:
            int: Number of items in request cache
        """
        with self._lock:
            return len(self._request_cache)

    def get_results_cache_size(self) -> int:
        """
        Get number of cached SHAP results

        Returns:
            int: Number of items in results cache
        """
        with self._results_lock:
            return len(self._results_cache)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the caches

        Returns:
            Dict with cache statistics
        """
        with self._lock:
            request_cache_size = len(self._request_cache)

        with self._results_lock:
            results_cache_size = len(self._results_cache)
            # Count expired entries
            current_time = time.time()
            expired_count = sum(
                1 for _, (_, timestamp) in self._results_cache.items()
                if current_time - timestamp >= self.cache_ttl
            )

        return {
            'request_cache_size': request_cache_size,
            'results_cache_size': results_cache_size,
            'results_cache_expired': expired_count,
            'cache_ttl_seconds': self.cache_ttl,
            'max_cache_size': self.max_cache_size
        }

    def cleanup_expired_results(self) -> int:
        """
        Clean up expired entries from results cache

        Returns:
            int: Number of entries removed
        """
        removed_count = 0
        current_time = time.time()

        with self._results_lock:
            keys_to_remove = [
                key for key, (_, timestamp) in self._results_cache.items()
                if current_time - timestamp >= self.cache_ttl
            ]
            for key in keys_to_remove:
                del self._results_cache[key]
                removed_count += 1

        if removed_count > 0:
            print(f"[CACHE CLEANUP] Removed {removed_count} expired entries from results cache")

        return removed_count

    def clear_results_cache(self):
        """
        Clear all entries from the results cache
        """
        with self._results_lock:
            count = len(self._results_cache)
            self._results_cache.clear()
            print(f"[CACHE CLEAR] Cleared {count} entries from results cache")

    def shutdown(self):
        """Shutdown the thread pool executor"""
        self.executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except Exception:
            pass
