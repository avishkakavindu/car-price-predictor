"""
Preprocessing functions for the model pipeline
These functions must match those used during model training
"""
import numpy as np
import pandas as pd


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
