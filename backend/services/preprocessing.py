"""
Preprocessing functions for the model pipeline
These functions must match those used during model training
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor


# Wrapper class to make CatBoost compatible with scikit-learn 1.6+
class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for CatBoostRegressor to ensure sklearn compatibility."""

    def __init__(self, iterations=1000, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3, loss_function='RMSE', eval_metric='RMSE',
                 random_seed=42, verbose=0, early_stopping_rounds=50):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.random_seed = random_seed
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self._model = None

    def fit(self, X, y):
        self._model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            random_seed=self.random_seed,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds
        )
        self._model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self._model.predict(X)

    def get_params(self, deep=True):
        return {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'loss_function': self.loss_function,
            'eval_metric': self.eval_metric,
            'random_seed': self.random_seed,
            'verbose': self.verbose,
            'early_stopping_rounds': self.early_stopping_rounds
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, 'is_fitted_') and self.is_fitted_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags


def create_car_age(X):
    """Create Car_Age from YOM"""
    X = X.copy()
    X['Car_Age'] = 2025 - X['YOM']
    return X


def create_log_mileage(X):
    """Create Mileage_log from Millage(KM)"""
    X = X.copy()
    X['Mileage_log'] = np.log1p(X['Millage(KM)'])
    return X


def group_brands(X):
    """Group brands to top 15 + Other"""
    X = X.copy()
    top_brands = ['TOYOTA', 'SUZUKI', 'NISSAN', 'HONDA', 'MITSUBISHI',
                  'PERODUA', 'MICRO', 'HYUNDAI', 'MAZDA', 'MERCEDES-BENZ',
                  'TATA', 'KIA', 'DAIHATSU', 'BMW', 'RENAULT']
    X['Brand_Grouped'] = X['Brand'].apply(
        lambda x: x if x in top_brands else 'Other'
    )
    return X


def group_fuel(X):
    """Group fuel types"""
    X = X.copy()
    X['Fuel_Grouped'] = X['Fuel Type'].replace({
        'Electric': 'Alternative',
        'Hybrid': 'Alternative'
    })
    return X


def encode_binary_features(X):
    """Encode binary features"""
    X = X.copy()
    binary_mappings = {
        'Gear': {'Automatic': 1, 'Manual': 0},
        'Leasing': {'Ongoing Lease': 1, 'No Leasing': 0},
        'Condition': {'NEW': 1, 'USED': 0},
        'AIR CONDITION': {'Available': 1, 'Not_Available': 0},
        'POWER STEERING': {'Available': 1, 'Not_Available': 0},
        'POWER MIRROR': {'Available': 1, 'Not_Available': 0},
        'POWER WINDOW': {'Available': 1, 'Not_Available': 0}
    }

    for col, mapping in binary_mappings.items():
        if col in X.columns:
            X[col] = X[col].map(mapping)

    return X


def select_features(X):
    """Drop unnecessary columns"""
    X = X.copy()
    drop_cols = ['YOM', 'Millage(KM)', 'Model', 'Town', 'Date',
                 'Brand', 'Fuel Type', 'Car_Age']

    X.drop(columns=[c for c in drop_cols if c in X.columns],
           inplace=True, errors='ignore')
    return X
