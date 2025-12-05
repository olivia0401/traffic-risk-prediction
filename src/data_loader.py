"""Data loading and preprocessing for UK DfT accident data"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(data_dir='data'):
    """
    Load and merge 2025 UK DfT accident data

    Args:
        data_dir: Directory containing CSV files

    Returns:
        X: Feature matrix
        y: Target vector (casualty_severity encoded as 0/1/2)
        feature_names: List of feature names
    """
    print("\nLoading 2025 UK DfT accident data...")

    # Load datasets
    collision_df = pd.read_csv(f'{data_dir}/collision_2025.csv',
                                low_memory=False)
    vehicle_df = pd.read_csv(f'{data_dir}/vehicle_2025.csv',
                              low_memory=False)
    casualty_df = pd.read_csv(f'{data_dir}/casualty_2025.csv',
                               low_memory=False)

    print(f"  Collision records: {len(collision_df)}")
    print(f"  Vehicle records: {len(vehicle_df)}")
    print(f"  Casualty records: {len(casualty_df)}")

    # Merge datasets
    print("\nMerging datasets...")
    merged = casualty_df.merge(collision_df, on='collision_index', how='left')
    merged = merged.merge(vehicle_df, on=['collision_index', 'vehicle_reference'],
                          how='left')

    print(f"  Merged records: {len(merged)}")

    # Select numeric and boolean features (similar to PDF document)
    numeric_features = [
        'number_of_vehicles', 'number_of_casualties', 'day_of_week',
        'road_type', 'speed_limit', 'light_conditions', 'weather_conditions',
        'road_surface_conditions', 'special_conditions_at_site',
        'vehicle_type', 'vehicle_manoeuvre', 'sex_of_driver',
        'age_of_driver', 'age_of_casualty', 'sex_of_casualty',
        'casualty_class', 'casualty_type'
    ]

    # Filter to available columns
    available_features = [f for f in numeric_features if f in merged.columns]

    # Prepare feature matrix
    X = merged[available_features].copy()

    # Handle missing values (replace -1 with median)
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].replace(-1, np.nan)
            X[col] = X[col].fillna(X[col].median())

    # Target variable: casualty_severity (1=Fatal, 2=Serious, 3=Slight)
    y = merged['casualty_severity'].copy()

    # Remove invalid targets
    valid_mask = y.isin([1, 2, 3])
    X = X[valid_mask]
    y = y[valid_mask]

    # Encode target: 1→0 (Fatal), 2→1 (Serious), 3→2 (Slight)
    y = y - 1

    print(f"\nFinal dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution:")
    print(f"    Fatal (0):   {(y==0).sum()}")
    print(f"    Serious (1): {(y==1).sum()}")
    print(f"    Slight (2):  {(y==2).sum()}")

    return X.values, y.values, available_features
