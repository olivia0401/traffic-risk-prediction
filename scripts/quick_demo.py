#!/usr/bin/env python3
"""
Quick demo script for traffic risk prediction

Demonstrates both severity classification and time series forecasting
without requiring actual data files (uses synthetic data for demo)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from trainer import SeverityClassifier


def generate_demo_severity_data():
    """Generate synthetic severity classification data"""
    print("\n" + "="*70)
    print(" DEMO: CASUALTY SEVERITY CLASSIFICATION")
    print("="*70)
    print("\nGenerating synthetic accident data...")

    # Create imbalanced multi-class dataset
    # 0=Fatal (rare), 1=Serious (medium), 2=Slight (common)
    X, y = make_classification(
        n_samples=10000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=2,
        weights=[0.05, 0.25, 0.70],  # Imbalanced classes
        random_state=42
    )

    print(f"\nDataset created:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution:")
    print(f"    Fatal (0):   {(y==0).sum():>5} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"    Serious (1): {(y==1).sum():>5} ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"    Slight (2):  {(y==2).sum():>5} ({(y==2).sum()/len(y)*100:.1f}%)")

    return X, y


def demo_severity_models():
    """Demo severity classification with all models"""
    X, y = generate_demo_severity_data()

    models = ['lr', 'rf', 'xgb']
    results = {}

    for model_type in models:
        print(f"\n{'-'*70}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'-'*70}")

        classifier = SeverityClassifier(model_type)
        metrics = classifier.train(X, y)
        results[model_type] = metrics

    # Print summary
    print("\n" + "="*70)
    print(" MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'F1 (CV)':<15} {'Recall (CV)':<15}")
    print("-" * 70)
    for model, metrics in results.items():
        f1 = f"{metrics['f1_cv_mean']:.3f} ± {metrics['f1_cv_std']:.3f}"
        recall = f"{metrics['recall_cv_mean']:.3f} ± {metrics['recall_cv_std']:.3f}"
        print(f"{model.upper():<20} {f1:<15} {recall:<15}")

    # Best model
    best = max(results.keys(), key=lambda k: results[k]['f1_cv_mean'])
    print(f"\n🏆 Best model: {best.upper()} (F1={results[best]['f1_cv_mean']:.3f})")


def generate_demo_timeseries_data():
    """Generate synthetic time series data"""
    print("\n\n" + "="*70)
    print(" DEMO: TIME SERIES FORECASTING")
    print("="*70)
    print("\nGenerating synthetic hourly accident counts...")

    # Generate 90 days of hourly data
    dates = pd.date_range(start='2025-01-01', periods=90*24, freq='H')

    # Base trend
    trend = np.linspace(10, 15, len(dates))

    # Weekly seasonality (more accidents on weekends)
    weekly_season = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*7))

    # Daily seasonality (rush hours)
    daily_season = 5 * np.sin(2 * np.pi * (dates.hour - 9) / 24)

    # Random noise
    noise = np.random.normal(0, 2, len(dates))

    # Combine components
    accidents = trend + weekly_season + daily_season + noise
    accidents = np.maximum(accidents, 0)  # No negative accidents

    time_series = pd.Series(accidents, index=dates)

    print(f"\nTime series created:")
    print(f"  Length: {len(time_series)} hours ({len(time_series)//24} days)")
    print(f"  Date range: {time_series.index.min()} to {time_series.index.max()}")
    print(f"  Mean accidents/hour: {time_series.mean():.2f}")
    print(f"  Max accidents/hour: {time_series.max():.0f}")
    print(f"  Min accidents/hour: {time_series.min():.0f}")

    return time_series


def demo_timeseries_simple():
    """Demo time series with simple baseline"""
    time_series = generate_demo_timeseries_data()

    print(f"\n{'-'*70}")
    print("Naive Baseline: Moving Average (7-day window)")
    print(f"{'-'*70}")

    # Simple 7-day moving average
    window = 24 * 7  # 7 days worth of hours
    predictions = time_series.rolling(window=window).mean()

    # Calculate metrics on last 30 days
    test_start = len(time_series) - 24*30
    actuals = time_series[test_start:]
    preds = predictions[test_start:]

    # Remove NaNs
    valid = ~preds.isna()
    actuals = actuals[valid]
    preds = preds[valid]

    mae = np.mean(np.abs(actuals - preds))
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1))) * 100

    print(f"\nPerformance (last 30 days):")
    print(f"  MAE:  {mae:.2f} accidents")
    print(f"  RMSE: {rmse:.2f} accidents")
    print(f"  MAPE: {mape:.2f}%")

    # Show hourly pattern
    print(f"\nAverage accidents by hour of day:")
    hourly_avg = time_series.groupby(time_series.index.hour).mean()
    print("\n  Hour | Avg Accidents | Visualization")
    print("  " + "-"*50)
    for hour, avg in hourly_avg.items():
        bar = '█' * int(avg / 2)
        print(f"  {hour:02d}:00 | {avg:6.2f}       | {bar}")

    # Peak hours
    peak_hours = hourly_avg.nlargest(3)
    print(f"\n🚨 Peak risk hours:")
    for hour, count in peak_hours.items():
        print(f"   {hour:02d}:00 - {count:.2f} accidents/hour")


def main():
    print("\n" + "="*70)
    print(" TRAFFIC RISK PREDICTION - QUICK DEMO")
    print(" (Using synthetic data - no data files required)")
    print("="*70)

    # Demo 1: Severity Classification
    try:
        demo_severity_models()
    except Exception as e:
        print(f"\n❌ Severity demo failed: {e}")
        import traceback
        traceback.print_exc()

    # Demo 2: Time Series Forecasting
    try:
        demo_timeseries_simple()
    except Exception as e:
        print(f"\n❌ Time series demo failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print(" ✅ DEMO COMPLETE")
    print("="*70)
    print("\nThis demo used synthetic data.")
    print("\nTo train on real UK DfT data:")
    print("  1. Download data from: https://www.data.gov.uk/dataset/road-accidents-safety-data")
    print("  2. Place CSV files in data/ directory")
    print("  3. Run: python scripts/train_models.py --task all")
    print("\nFor more info, see README.md")


if __name__ == '__main__':
    main()
