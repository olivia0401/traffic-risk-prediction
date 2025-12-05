#!/usr/bin/env python3
"""
Complete training script for traffic risk prediction models

Usage:
    python scripts/train_models.py --task severity --model rf
    python scripts/train_models.py --task timeseries --model prophet --freq daily
    python scripts/train_models.py --task all
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from data_loader import load_and_prepare_data
from trainer import SeverityClassifier, compare_models
from time_series_predictor import TimeSeriesPredictor, compare_timeseries_models


def train_severity_models(data_dir='data', model_type='all'):
    """
    Train casualty severity classification models

    Args:
        data_dir: Directory containing CSV files
        model_type: 'lr', 'rf', 'mlp', 'xgb', or 'all'
    """
    print("\n" + "="*70)
    print(" CASUALTY SEVERITY CLASSIFICATION")
    print("="*70)

    # Load data
    X, y, feature_names = load_and_prepare_data(data_dir)

    print(f"\nFeatures used: {len(feature_names)}")
    print(f"  {', '.join(feature_names[:5])}...")

    if model_type == 'all':
        # Compare all models
        results = compare_models(X, y)

        # Save best model (highest F1)
        best_model_name = max(results, key=lambda k: results[k]['f1_cv_mean'])
        best_model_type = {
            'Logistic Regression': 'lr',
            'Random Forest': 'rf',
            'MLP': 'mlp',
            'XGBoost': 'xgb'
        }[best_model_name]

        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   F1-score: {results[best_model_name]['f1_cv_mean']:.3f}")

        # Train and save best model
        best_classifier = SeverityClassifier(best_model_type)
        best_classifier.train(X, y)
        best_classifier.save_model('models/severity_best.pkl')

    else:
        # Train single model
        classifier = SeverityClassifier(model_type)
        metrics = classifier.train(X, y)
        classifier.save_model(f'models/severity_{model_type}.pkl')


def train_timeseries_models(data_dir='data', model_type='all', frequency='daily'):
    """
    Train time series forecasting models

    Args:
        data_dir: Directory containing CSV files
        model_type: 'arima', 'sarima', 'prophet', 'lstm', or 'all'
        frequency: 'daily' or 'hourly'
    """
    print("\n" + "="*70)
    print(f" TIME SERIES FORECASTING ({frequency.upper()})")
    print("="*70)

    # Load collision data
    collision_df = pd.read_csv(f'{data_dir}/collision_2025.csv', low_memory=False)
    print(f"\nLoaded {len(collision_df)} collision records")

    # Prepare time series
    predictor = TimeSeriesPredictor()

    if frequency == 'daily':
        time_series = predictor.prepare_daily_data(collision_df)
        print(f"\nDaily time series: {len(time_series)} days")
    elif frequency == 'hourly':
        time_series = predictor.prepare_hourly_data(collision_df)
        print(f"\nHourly time series: {len(time_series)} hours")
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    print(f"Date range: {time_series.index.min()} to {time_series.index.max()}")
    print(f"Mean accidents per {frequency}: {time_series.mean():.2f}")
    print(f"Max accidents: {time_series.max()}")

    if model_type == 'all':
        # Compare all models
        results = compare_timeseries_models(time_series, frequency=frequency)

        # Save best model
        if results:
            best_model_name = min(results, key=lambda k: results[k]['mae'])
            print(f"\n🏆 Best model: {best_model_name}")
            print(f"   MAE: {results[best_model_name]['mae']:.2f}")

    else:
        # Train single model
        predictor = TimeSeriesPredictor(model_type)

        if model_type == 'arima':
            metrics = predictor.train_arima(time_series)
        elif model_type == 'sarima':
            if frequency != 'hourly':
                print("Warning: SARIMA works best with hourly data")
            metrics = predictor.train_sarima(time_series)
        elif model_type == 'prophet':
            metrics = predictor.train_prophet(time_series)
        elif model_type == 'lstm':
            metrics = predictor.train_lstm(time_series)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Save model
        predictor.save_model(f'models/timeseries_{model_type}_{frequency}.pkl')

        # Generate forecast example
        print(f"\n📈 Generating 7-day forecast...")
        if frequency == 'daily':
            forecast = predictor.forecast(steps=7)
        else:
            forecast = predictor.forecast(steps=24*7)  # 7 days worth of hours

        print(f"Forecasted accidents (next 7 days):")
        for i, val in enumerate(forecast[:7] if frequency == 'daily' else forecast[::24][:7]):
            print(f"  Day {i+1}: {val:.1f} accidents")


def main():
    parser = argparse.ArgumentParser(
        description='Train traffic risk prediction models'
    )
    parser.add_argument(
        '--task',
        choices=['severity', 'timeseries', 'all'],
        default='all',
        help='Task to train: severity classification, time series forecasting, or both'
    )
    parser.add_argument(
        '--model',
        default='all',
        help='Model type (task-specific) or "all" to compare models'
    )
    parser.add_argument(
        '--freq',
        choices=['daily', 'hourly'],
        default='daily',
        help='Time series frequency (for timeseries task)'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing data files'
    )

    args = parser.parse_args()

    # Create models directory
    Path('models').mkdir(exist_ok=True)

    if args.task in ['severity', 'all']:
        try:
            train_severity_models(args.data_dir, args.model)
        except FileNotFoundError as e:
            print(f"\n❌ Error: Data files not found in '{args.data_dir}/'")
            print("Please download UK DfT data from:")
            print("https://www.data.gov.uk/dataset/road-accidents-safety-data")
            return 1
        except Exception as e:
            print(f"\n❌ Severity training failed: {e}")
            import traceback
            traceback.print_exc()

    if args.task in ['timeseries', 'all']:
        try:
            train_timeseries_models(args.data_dir, args.model, args.freq)
        except FileNotFoundError as e:
            print(f"\n❌ Error: Data files not found in '{args.data_dir}/'")
            print("Please download UK DfT data from:")
            print("https://www.data.gov.uk/dataset/road-accidents-safety-data")
            return 1
        except Exception as e:
            print(f"\n❌ Time series training failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(" ✅ TRAINING COMPLETE")
    print("="*70)
    print("\nTrained models saved in models/ directory")
    print("\nNext steps:")
    print("  1. Evaluate models on test set")
    print("  2. Deploy best model as API")
    print("  3. Create visualization dashboard")

    return 0


if __name__ == '__main__':
    sys.exit(main())
