"""Data loading and preprocessing for UK DfT accident data"""
import glob
import os

import pandas as pd
import numpy as np


# Features known *before/at* the moment a collision occurs — road, environment,
# time and location context. Deliberately excludes everything that only exists
# once the collision has happened and casualties are recorded (severity,
# number/ages/types of casualties, driver/vehicle post-hoc attributes). Using
# only these makes the severity model an honest *ahead-of-time* risk estimate
# rather than a retrospective description that leaks the answer.
PRE_INCIDENT_FEATURES = [
    'day_of_week', 'road_type', 'speed_limit', 'first_road_class',
    'junction_detail', 'junction_control', 'pedestrian_crossing',
    'light_conditions', 'weather_conditions', 'road_surface_conditions',
    'special_conditions_at_site', 'carriageway_hazards',
    'urban_or_rural_area',
]


def _find_collision_csv(data_dir):
    """Locate a collision CSV in data_dir (any year), or raise a clear error."""
    for pattern in ('collision_*.csv', 'dft-road-casualty-statistics-collision-*.csv'):
        hits = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if hits:
            return hits[-1]  # most recent year if several
    raise FileNotFoundError(
        f"No collision CSV found in {data_dir!r}. Download one from "
        "https://data.dft.gov.uk/road-accidents-safety-data/ "
        "(e.g. dft-road-casualty-statistics-collision-2023.csv)."
    )


def load_leakage_free_severity(data_dir='data', hour_feature=True):
    """
    Build an **honest, ahead-of-time** collision-severity dataset.

    Target: ``collision_severity`` (1=Fatal, 2=Serious, 3=Slight → 0/1/2), taken
    at the *collision* level from the collision file alone — no casualty/vehicle
    join, so none of the post-collision attributes that leaked into the original
    retrospective model can sneak in.

    Features: only ``PRE_INCIDENT_FEATURES`` (plus hour-of-day derived from the
    collision time), i.e. things a planner could know *before* a crash. This is
    the feature restriction the README's leakage note called for.

    Returns (X, y, feature_names).
    """
    path = _find_collision_csv(data_dir)
    print(f"\nLoading leakage-free severity data from {os.path.basename(path)} ...")
    df = pd.read_csv(path, low_memory=False)

    feats = [f for f in PRE_INCIDENT_FEATURES if f in df.columns]
    X = df[feats].copy()

    if hour_feature and 'time' in df.columns:
        hour = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.hour
        X['hour_of_day'] = hour
        feats = feats + ['hour_of_day']

    # DfT encodes missing/unknown as -1 (and 9/99 for some fields); treat -1 as
    # missing and median-fill, matching the original loader's convention.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').replace(-1, np.nan)
        X[col] = X[col].fillna(X[col].median())

    y = pd.to_numeric(df['collision_severity'], errors='coerce')
    valid = y.isin([1, 2, 3])
    X, y = X[valid], y[valid].astype(int) - 1  # 1/2/3 → 0/1/2

    print(f"  Samples: {len(X)} | Features: {X.shape[1]} ({', '.join(X.columns)})")
    print(f"  Class distribution (0=Fatal,1=Serious,2=Slight): "
          f"{np.bincount(y.values).tolist()}")
    return X.values, y.values, list(X.columns)


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
