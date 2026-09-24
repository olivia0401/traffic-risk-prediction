#!/usr/bin/env python3
"""
Seasonal-naive baselines for the hourly accident-count forecast.

Scores "same hour yesterday" (lag 24) and "same hour last week" (lag 168) on
the same held-out tail the LSTM is scored on (the most recent 20% of hours,
see TimeSeriesPredictor.train_lstm), so the numbers in the README's
forecasting table are directly comparable.

Usage:
    python scripts/naive_baselines.py [--data-dir data] [--test-frac 0.2]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import _find_collision_csv          # noqa: E402
from time_series_predictor import TimeSeriesPredictor  # noqa: E402


def seasonal_naive_scores(values, lag, test_frac=0.2):
    """MAE / RMSE of predicting y[t] with y[t - lag] over the last test_frac of values."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    split = n - max(1, int(round(n * test_frac)))
    if split < lag:
        raise ValueError(f"Series too short ({n}) for lag {lag}")
    actual = values[split:]
    pred = values[split - lag:n - lag]
    err = actual - pred
    return {'mae': float(np.mean(np.abs(err))),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'n_test': int(len(actual))}


def main():
    ap = argparse.ArgumentParser(description="Seasonal-naive hourly baselines")
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--test-frac', type=float, default=0.2)
    args = ap.parse_args()

    path = _find_collision_csv(args.data_dir)
    df = pd.read_csv(path, low_memory=False)
    series = TimeSeriesPredictor().prepare_hourly_data(df)
    print(f"{os.path.basename(path)}: {len(series)} hours, "
          f"test tail = last {args.test_frac:.0%}")

    for name, lag in (('same hour yesterday', 24), ('same hour last week', 168)):
        s = seasonal_naive_scores(series.values, lag, args.test_frac)
        print(f"  {name:<22} MAE {s['mae']:.2f}  RMSE {s['rmse']:.2f}  (n={s['n_test']})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
