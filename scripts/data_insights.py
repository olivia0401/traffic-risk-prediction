#!/usr/bin/env python3
"""
Descriptive statistics quoted in the README's "Data Insights" table, computed
from the collision CSV alone.

Usage:
    python scripts/data_insights.py [--data-dir data]
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import _find_collision_csv  # noqa: E402

# DfT day_of_week coding: 1=Sunday ... 7=Saturday
DAY_NAMES = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday',
             5: 'Thursday', 6: 'Friday', 7: 'Saturday'}


def insights(df):
    """Return the headline descriptive figures for a DfT collision table."""
    hour = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.hour
    by_day = df['day_of_week'].map(DAY_NAMES).value_counts()
    severity = pd.to_numeric(df['collision_severity'], errors='coerce')
    return {
        'collisions': int(len(df)),
        'share_15_to_18h': float(hour.between(15, 17).mean()),   # 15:00-17:59
        'share_fine_no_high_winds': float((df['weather_conditions'] == 1).mean()),
        'busiest_day': by_day.index[0],
        'severity_share': {k: float(v) for k, v in
                           severity.map({1: 'fatal', 2: 'serious', 3: 'slight'})
                           .value_counts(normalize=True).items()},
    }


def main():
    ap = argparse.ArgumentParser(description="Descriptive collision statistics")
    ap.add_argument('--data-dir', default='data')
    args = ap.parse_args()

    path = _find_collision_csv(args.data_dir)
    res = insights(pd.read_csv(path, low_memory=False))
    print(f"{os.path.basename(path)}: {res['collisions']} collisions")
    print(f"  15:00-17:59 share:          {res['share_15_to_18h']:.1%}")
    print(f"  Fine weather (no high wind): {res['share_fine_no_high_winds']:.1%}")
    print(f"  Busiest day:                {res['busiest_day']}")
    print("  Severity: " + ", ".join(f"{k} {v:.1%}" for k, v in res['severity_share'].items()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
