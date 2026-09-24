"""Offline tests: no DfT download needed (tiny synthetic CSVs in tmp_path)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import data_loader  # noqa: E402
from data_loader import (PRE_INCIDENT_FEATURES, _find_table_csv,  # noqa: E402
                         load_leakage_free_severity)
from naive_baselines import seasonal_naive_scores  # noqa: E402
from time_series_predictor import TimeSeriesPredictor  # noqa: E402
from trainer import SeverityClassifier  # noqa: E402


def _collision_csv(path, n=300, seed=0, post_crash=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.integers(1, 6, n) for f in PRE_INCIDENT_FEATURES})
    df.loc[:9, 'road_type'] = -1  # DfT "missing" code -> must be median-filled
    df['collision_index'] = [f'c{i}' for i in range(n)]
    df['date'] = pd.date_range('2023-01-01', periods=n, freq='h').strftime('%d/%m/%Y')
    df['time'] = pd.date_range('2023-01-01', periods=n, freq='h').strftime('%H:%M')
    df['collision_severity'] = rng.choice([1, 2, 3], n, p=[0.1, 0.3, 0.6])
    if post_crash:
        # post-crash casualty fields must never reach the leakage-free model
        df['age_of_casualty'] = rng.integers(1, 90, n)
        df['casualty_type'] = rng.integers(0, 20, n)
    df.to_csv(path, index=False)
    return df


def test_find_table_csv_accepts_both_naming_schemes(tmp_path):
    (tmp_path / 'collision_2023.csv').write_text('x\n')
    (tmp_path / 'dft-road-casualty-statistics-vehicle-2023.csv').write_text('x\n')
    assert _find_table_csv(str(tmp_path), 'collision').endswith('collision_2023.csv')
    assert _find_table_csv(str(tmp_path), 'vehicle').endswith('vehicle-2023.csv')
    with pytest.raises(FileNotFoundError, match='casualty'):
        _find_table_csv(str(tmp_path), 'casualty')


def test_find_table_csv_picks_most_recent_year(tmp_path):
    for year in (2022, 2023):
        (tmp_path / f'collision_{year}.csv').write_text('x\n')
    assert _find_table_csv(str(tmp_path), 'collision').endswith('collision_2023.csv')


def test_leakage_free_loader_excludes_post_crash_fields(tmp_path):
    _collision_csv(tmp_path / 'collision_2023.csv')
    X, y, feats = load_leakage_free_severity(str(tmp_path))
    assert 'age_of_casualty' not in feats and 'casualty_type' not in feats
    assert set(feats) == set(PRE_INCIDENT_FEATURES) | {'hour_of_day'}
    assert set(np.unique(y)) <= {0, 1, 2}
    assert not np.isnan(X).any()
    assert (X[:, feats.index('road_type')] != -1).all()


def test_retrospective_loader_works_with_2023_files(tmp_path):
    # used to hard-code *_2025.csv; any year must now work
    col = _collision_csv(tmp_path / 'collision_2023.csv', n=50, post_crash=False)
    ids = col['collision_index']
    pd.DataFrame({'collision_index': ids, 'vehicle_reference': 1,
                  'vehicle_type': 9, 'age_of_driver': 40}).to_csv(
        tmp_path / 'vehicle_2023.csv', index=False)
    pd.DataFrame({'collision_index': ids, 'vehicle_reference': 1,
                  'casualty_severity': [1, 2, 3, 3, 3] * 10,
                  'age_of_casualty': 30}).to_csv(
        tmp_path / 'casualty_2023.csv', index=False)
    X, y, feats = data_loader.load_and_prepare_data(str(tmp_path))
    assert len(X) == 50 and set(np.unique(y)) == {0, 1, 2}
    assert 'age_of_casualty' in feats   # the retrospective model is leaky by design


def test_hourly_series_is_gap_free():
    df = pd.DataFrame({'date': ['01/01/2023', '01/01/2023', '01/01/2023'],
                       'time': ['00:10', '00:50', '03:00']})
    s = TimeSeriesPredictor().prepare_hourly_data(df)
    assert list(s.values) == [2, 0, 0, 1]
    assert s.index.freq == 'h' or pd.infer_freq(s.index) == 'h'


def test_seasonal_naive_is_exact_on_a_periodic_series():
    values = np.tile(np.arange(24), 30)
    res = seasonal_naive_scores(values, lag=24)
    assert res['mae'] == 0 and res['rmse'] == 0
    assert res['n_test'] == round(len(values) * 0.2)


def test_severity_classifier_cv_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    clf = SeverityClassifier('rf')
    metrics = clf.train(X, y)
    assert 0 <= metrics['f1_cv_mean'] <= 1 and 0 <= metrics['recall_cv_mean'] <= 1
    clf.save_model(str(tmp_path / 'm.pkl'))
    loaded = SeverityClassifier.load_model(str(tmp_path / 'm.pkl'))
    assert (loaded.predict(X) == clf.predict(X)).all()


def test_lstm_scores_a_held_out_tail():
    pytest.importorskip('torch')
    idx = pd.date_range('2023-01-01', periods=400, freq='h')
    s = pd.Series(10 + 5 * np.sin(np.arange(400) * 2 * np.pi / 24), index=idx)
    p = TimeSeriesPredictor('lstm')
    m = p.train_lstm(s, epochs=3, patience=2)
    assert m['n_test'] == 80 and m['mae'] >= 0
    assert len(p.forecast(steps=5)) == 5


def test_quick_demo_exits_zero():
    pytest.importorskip('xgboost')
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'quick_demo.py')],
                       capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=600, env={**os.environ,
                                'PYTHONDONTWRITEBYTECODE': '1',
                                'PYTHONIOENCODING': 'utf-8'})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert '[OK] DEMO COMPLETE' in r.stdout
