# UK Road Casualty Severity Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)](https://www.postgresql.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine-learning and data-engineering study of UK road-safety records: **severity
classification** (a retrospective casualty-level model and an honest leakage-free
collision-level model) plus **temporal analysis** of when accidents cluster. Built on UK
Department for Transport open data (2025 provisional extract and the full-year 2023 file),
combining SQL analytics, time-series forecasting, and multi-class classification.

> **Scope & honesty note.** This is mostly a *post-hoc* analysis of accidents that
> already happened, **not** a real-time risk system. The original (retrospective)
> severity model uses casualty attributes (`age_of_casualty`, `sex_of_casualty`,
> `casualty_class`, `casualty_type`) that are only known **after** a collision is
> recorded, so its scores are not achievable for prediction *before* an accident. A
> separate leakage-free model uses pre-incident context only and scores far lower
> (macro-F1 ~0.35). Treat the results as descriptive/analytical, not as a safety system.
> See [Feature scope & leakage](#feature-scope--leakage).

## Overview

This project demonstrates a full data-science pipeline over real-world road-safety data:
temporal analysis (when do accidents cluster?) and casualty-severity classification
(given a recorded collision, how serious was the casualty?).

**Dataset:** UK DfT road-safety open data. The SQL schema and the original write-up were built on a 2025 provisional extract (48,472 collisions). Every number in the tables below except the retrospective severity scores was re-run on the full-year 2023 collision file (104,258 collisions). Data files are gitignored, so download it first (see [`data/README.md`](data/README.md)).
**Code:** ~1,800 lines of Python (src 980, scripts 699, tests 125) + 989 lines of SQL
**Status:** Functional, with a synthetic-data demo and an offline pytest suite

---

## Project Highlights

- **Multi-Table Data Integration**: collision + vehicle + casualty join (casualty-level rows) for the retrospective model
- **Advanced SQL Analytics**: 989 lines with window functions, CTEs, and custom risk scoring
- **Time Series Forecasting**: 4 approaches (ARIMA, SARIMA, Prophet, LSTM); only the LSTM has a held-out backtest
- **Imbalanced Classification**: Multi-class severity prediction with SMOTE / class weights
- **Modular Code**: separate data, training and forecasting modules, CLI scripts, offline tests
- **No-Download Demo**: Synthetic data generator for instant testing

---

## Architecture

### Project Structure

```
traffic-risk-prediction/
├── src/                           # Core ML modules
│   ├── data_loader.py            # DfT CSV discovery (any year), leak-free + retrospective loaders
│   ├── trainer.py                # Severity classification models
│   └── time_series_predictor.py  # Time series forecasting (4 approaches)
│
├── scripts/                       # Executable workflows
│   ├── quick_demo.py             # Synthetic data demo (no downloads)
│   ├── train_models.py           # Training CLI (severity / timeseries / all)
│   ├── train_severity.py         # Retrospective (leaky) severity models
│   ├── train_severity_leakfree.py # Leakage-free severity model (pre-incident features)
│   ├── naive_baselines.py        # Seasonal-naive forecasting baselines
│   └── data_insights.py          # Descriptive statistics in "Data Insights"
│
├── sql/                           # Database layer (PostgreSQL)
│   ├── 00_schema.sql             # Table definitions & indexes
│   ├── 01_data_cleaning.sql      # Data validation & deduplication
│   ├── 02_feature_engineering.sql # Views & temporal features
│   ├── 03_temporal_analysis.sql  # Time series queries
│   └── 04_risk_prediction.sql    # Risk scoring functions
│
├── tests/                         # Offline pytest suite (synthetic data)
├── notebooks/                     # Jupyter EDA
├── data/                         # CSV storage (gitignored; see data/README.md)
├── models/                       # Trained model artifacts
└── README.md                     # Documentation
```

---

## Key Results & Insights

### Data Insights

Full-year 2023 collision file (104,258 collisions). Reproduce with `python scripts/data_insights.py`.

| Finding | Detail | Interpretation |
|---------|--------|----------------|
| **Afternoon peak** | 25.4% of collisions happen 15:00-17:59 (3 of 24 hours) | Rush-hour concentration |
| **Fine-weather share** | 78.7% of collisions in fine weather without high winds | Reflects exposure (most driving is in fine weather), not weather safety |
| **Friday effect** | Friday has the most collisions | End-of-week temporal pattern |

### Model Performance

**Hourly forecasting — held-out backtest on real 2023 DfT data** (8,760 hours, last 20% kept back for testing):

| Model | MAE | RMSE |
|-------|-----|------|
| same hour yesterday (naive) | 4.92 | 6.97 |
| same hour last week (naive) | 4.49 | 6.42 |
| **LSTM (PyTorch, 2-layer)** | **3.45** | **4.80** |

The LSTM beats both naive baselines. It's trained on the earlier part of the year and scored on the most recent slice, so there's no peeking. Reproduce with `python scripts/train_models.py --task timeseries --model lstm --freq hourly` (LSTM) and `python scripts/naive_baselines.py` (baselines, same test tail).

The old daily ARIMA/Prophet numbers were in-sample, so I've left them out of this comparison.

**Severity classification — two versions.**

Predicting how serious a collision is *before* it happens, from road / weather / time / junction context only (5-fold CV, real 2023 data):

| Model | macro-F1 (CV) | macro-recall (CV) |
|-------|---------------|-------------------|
| Logistic Regression | 0.311 ± 0.002 | 0.463 |
| **Random Forest** | **0.350 ± 0.002** | 0.366 |
| XGBoost | 0.339 ± 0.003 | 0.418 |

An earlier version scored ~0.76, but it used fields you only know *after* a crash — the casualty's age, sex and injury type. That's leakage: it describes accidents instead of predicting them. Rebuilt on pre-incident context only, the honest number is ~0.35. Code: `scripts/train_severity_leakfree.py`.

The two numbers are not a strict ablation. The ~0.76 is casualty-level `casualty_severity` from the 2025 provisional collision + vehicle + casualty join. The ~0.35 is collision-level `collision_severity` from the 2023 collision file alone. The 0.76 has **not** been re-run with the current code: this repo only reproduces from the collision file. To re-run it, put the vehicle and casualty CSVs for the same year in `data/` and run `python scripts/train_severity.py --model rf`.

**Class balance (real 2023 data):** Fatal 1.5% · Serious 22.5% · Slight 76% (104,258 collisions), handled with SMOTE + class weights.

---

## Machine Learning Implementation

### 1. Severity Classification (Multi-Class Imbalanced Learning)

**Location:** `src/trainer.py`

**Challenge:** Highly imbalanced severity (2023 collision level: 1.5% fatal, 22.5% serious, 76% slight)

**Solution:**
```python
# Imbalance handling differs per model (see SeverityClassifier._build_model):
#   LR / RF -> class_weight='balanced'
#   XGB     -> SMOTE inside an imblearn pipeline (resampling only on training folds)
#   MLP     -> StandardScaler + SMOTE
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(n_estimators=100, max_depth=3, eval_metric='mlogloss'))
])
```

> Heads up — the "retrospective" F1 numbers below are the leaky ones (they include post-crash casualty fields). They are historical figures from the 2025 provisional extract and are not re-run by this repo (see above). The honest leakage-free scores are up in [Model Performance](#model-performance).

**Models Implemented:**
1. **Random Forest** (retrospective F1≈0.762; leakage-free ≈0.350)
   - n_estimators=100, class_weight='balanced'
   - No scaling needed (tree-based)

2. **XGBoost** (retrospective F1≈0.74; leakage-free ≈0.339)
   - SMOTE + multi-class logloss
   - n_estimators=100, max_depth=3

3. **MLP Neural Network** (retrospective F1≈0.72)
   - Architecture: [input] → [50 hidden] → [output]
   - SMOTE + StandardScaler

4. **Logistic Regression** (Baseline: F1=0.60)
   - class_weight='balanced'
   - Max iterations: 1,000

**Evaluation:** Stratified 5-Fold CV with macro F1-scoring

#### Feature scope & leakage

The severity classifier is trained on attributes recorded **as part of the accident
report**, including casualty-level fields (`age_of_casualty`, `sex_of_casualty`,
`casualty_class`, `casualty_type`) and collision context. These are only available
*after* a collision has occurred and a casualty exists.

Implications, stated plainly:

- This is **retrospective casualty-severity classification**, useful for understanding
  and describing recorded accidents — e.g. which recorded conditions correlate with more
  severe casualties.
- It is **not** a pre-accident or real-time risk system. A model that consumes
  post-collision casualty attributes cannot run *before* an accident, so the reported F1
  does not transfer to any "predict risk ahead of time" deployment (that would be
  target-time leakage / deployment-time unavailability).
- To repurpose this for genuine ahead-of-time risk estimation, the feature set would need
  to be restricted to variables observable *before* an incident (road/weather/time/location
  context only), and re-evaluated — the numbers here would not carry over.

---

### 2. Time Series Forecasting (4 Approaches)

**Location:** `src/time_series_predictor.py`

**Data Aggregation:**
- Daily: Groups accidents by calendar day
- Hourly: 24-hour seasonality detection
- Missing period imputation with zeros

**Model 1: ARIMA(5,1,2)**
```python
# AutoRegressive Integrated Moving Average
order=(5, 1, 2)  # p=5 AR lags, d=1 differencing, q=2 MA lags
# Use case: Short-term forecasting (1-3 days)
# Reports in-sample fit only (no held-out backtest yet)
```

**Model 2: SARIMA(1,1,1,24)**
```python
# Seasonal ARIMA for hourly patterns
order=(1,1,1), seasonal_order=(1,1,1,24)
# Captures rush hour patterns (7-9 AM, 3-6 PM)
# Reports in-sample fit only (no held-out backtest yet)
```

**Model 3: Prophet (Meta's Forecasting Library)**
```python
# Flexible multi-level seasonality
Prophet(yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05)
# Advantages: Handles holidays, robust to outliers
# Reports in-sample fit only (no held-out backtest yet)
```

**Model 4: LSTM (Deep Learning, PyTorch)**
```python
# 2-layer LSTM, hidden_size=50, batch_first
# Chronological train/val/test split (no shuffle, scaler fit on train only)
# Mini-batch training + early stopping on a validation tail
# Recursive multi-step forecast(): each prediction is fed back as the newest
#   observation and the input window slides forward one step
# Reported MAE/RMSE/MAPE are held-out (test-tail) one-step-ahead errors
```

**Metrics Used:** MAE, RMSE, MAPE (all models); AIC/BIC (ARIMA/SARIMA)

---

## SQL Data Pipeline

**Total:** 989 lines across 5 SQL files

### Schema Design (`00_schema.sql`)

**Tables:**
Row counts are for the 2025 provisional extract.

1. **collision** (48,472 rows)
   - PK: collision_index
   - Fields: date, time, location, weather, severity
   - Indexes on: date, time, severity, lat/lon

2. **vehicle** (87,805 rows)
   - FK: collision_index
   - Fields: vehicle_type, driver_age, manoeuvre

3. **casualty** (60,991 rows)
   - FK: collision_index
   - Fields: casualty_severity, age, type

### Advanced SQL Features

**Window Functions:**
```sql
-- 7-day rolling average
AVG(daily_accidents) OVER (
    ORDER BY date_of_accident
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
) as rolling_7day_avg
```

**CTEs for Complex Queries:**
```sql
WITH high_risk_periods AS (
    SELECT hour_of_day, day_of_week, COUNT(*) as accidents
    FROM collision_with_features
    GROUP BY hour_of_day, day_of_week
)
SELECT * FROM high_risk_periods WHERE accidents > 300;
```

**Custom Risk Scoring Function (PL/pgSQL):**
```sql
CREATE FUNCTION get_risk_score(p_hour INT, p_day INT, ...)
RETURNS TABLE (predicted_risk_level TEXT, expected_accidents INT) AS $$
BEGIN
    RETURN QUERY
    SELECT CASE WHEN COUNT(*) > 300 THEN 'CRITICAL' ...
    FROM collision_with_features WHERE ...;
END;
$$ LANGUAGE plpgsql;
```

---

## Installation & Usage

### Prerequisites
- Python 3.10+
- PostgreSQL 12+ (optional, for SQL analytics)
- pip package manager

### Quick Start (No Data Download)

```bash
# Clone repository
git clone https://github.com/olivia0401/traffic-risk-prediction.git
cd traffic-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run synthetic demo
python scripts/quick_demo.py

# Output: Trains models on synthetic data, shows performance metrics
```

### Full Pipeline with Real Data

```bash
# 1. Download UK DfT data (any year; see data/README.md for file names)
# https://www.data.gov.uk/dataset/road-accidents-safety-data
# The collision CSV alone is enough for everything except the retrospective model.

# 2. Collision file only
python scripts/train_severity_leakfree.py --model rf          # honest severity model
python scripts/train_models.py --task timeseries --model lstm --freq hourly
python scripts/naive_baselines.py                             # forecasting baselines
python scripts/data_insights.py                               # descriptive stats

# 3. Needs collision + vehicle + casualty CSVs of the same year
python scripts/train_models.py --task severity --model rf     # retrospective (leaky) model
python scripts/train_models.py --task all                     # severity + time series

# ARIMA/SARIMA/Prophet need statsmodels / prophet (in requirements.txt)
python scripts/train_models.py --task timeseries --model prophet --freq daily

# Models are saved to models/
```

### SQL Analytics

```bash
# Setup PostgreSQL database
createdb traffic_risk

# Create tables
psql traffic_risk -f sql/00_schema.sql

# Load the CSVs yourself: no loader script is included, and the schema's column
# names (e.g. date_of_accident, time_of_accident) differ from the raw DfT headers
# (date, time), so import into a staging table and INSERT ... SELECT with renames.

# Clean, build feature views, run the analysis
psql traffic_risk -f sql/01_data_cleaning.sql
psql traffic_risk -f sql/02_feature_engineering.sql
psql traffic_risk -f sql/03_temporal_analysis.sql
psql traffic_risk -f sql/04_risk_prediction.sql

# Query risk scores: hour, day_of_week (0=Sunday, PostgreSQL DOW), then DfT codes for
# weather, light, road_surface, urban_rural
# 5 PM, Friday, rain, dark-lit, wet, urban
psql traffic_risk -c "SELECT * FROM get_risk_score(17, 5, 2, 4, 2, 1);"
```

---

## Technical Stack

### Python Dependencies

| Library | Purpose |
|---------|---------|
| **Data Processing** | |
| pandas ≥1.5 | DataFrames, time series |
| numpy ≥1.23 | Numerical computing |
| psycopg2 ≥2.9 | PostgreSQL driver (optional; the Python code does not use it yet) |
| **Machine Learning** | |
| scikit-learn ≥1.2 | ML algorithms, pipelines |
| xgboost ≥1.7 | Gradient boosting |
| imbalanced-learn ≥0.10 | SMOTE resampling |
| **Time Series** | |
| statsmodels ≥0.13 | ARIMA, SARIMA |
| prophet ≥1.1 | Facebook's forecasting |
| torch ≥2.0 | PyTorch for LSTM |
| **Deployment (Optional)** | |
| fastapi ≥0.95 | REST API (planned) |
| uvicorn ≥0.21 | ASGI server |
| **Development** | |
| pytest ≥7.2 | Offline test suite (`tests/`) |
| jupyter | Notebooks |

### Database
- PostgreSQL 12+
- 3 tables with proper indexing
- Window functions, CTEs, custom functions

---

## Code Examples

### Training Severity Classifier

```python
# run from the repo root
from src.data_loader import load_leakage_free_severity
from src.trainer import SeverityClassifier, compare_models

# Pre-incident features only (collision CSV alone)
X, y, features = load_leakage_free_severity('data')

# Random Forest with balanced class weights; prints 5-fold macro-F1 / recall
classifier = SeverityClassifier(model_type='rf')
metrics = classifier.train(X, y)
classifier.save_model('models/severity_leakfree_rf.pkl')

# Or compare LR / RF / MLP / XGB (slow on the full year: MLP + SMOTE)
# results = compare_models(X, y)
```

### Time Series Forecasting

```python
# run from the repo root
import pandas as pd
from src.data_loader import _find_collision_csv
from src.time_series_predictor import TimeSeriesPredictor

collision_df = pd.read_csv(_find_collision_csv('data'), low_memory=False)

predictor = TimeSeriesPredictor(model_type='lstm')
ts = predictor.prepare_hourly_data(collision_df)
metrics = predictor.train_lstm(ts)            # held-out MAE / RMSE / MAPE

next_week = predictor.forecast(steps=24 * 7)  # hourly counts
print(f"Next 7 days: {next_week.sum():.0f} accidents")
```

### SQL Risk Query

```sql
-- High-risk periods query
SELECT hour_of_day,
       day_name,
       COUNT(*) as total_accidents,
       SUM(CASE WHEN collision_severity=1 THEN 1 ELSE 0 END) as fatal,
       AVG(number_of_casualties) as avg_casualties
FROM collision_with_features
WHERE hour_of_day BETWEEN 15 AND 18  -- 3-6 PM
GROUP BY hour_of_day, day_name
ORDER BY total_accidents DESC;
```

---

## Skills Demonstrated

### Machine Learning & Data Science
- **Imbalanced Learning**: SMOTE oversampling, class weighting
- **Time Series Analysis**: 4 forecasting approaches (ARIMA, SARIMA, Prophet, LSTM)
- **Model Selection**: Comparative evaluation with cross-validation
- **Feature Engineering**: Temporal features, categorical encoding
- **Evaluation**: Multi-metric assessment (F1, Recall, MAE, RMSE, AIC)

### SQL & Database Engineering
- **Advanced SQL**: Window functions (ROW_NUMBER, LAG/LEAD, rolling AVG)
- **Query Optimization**: Proper indexing strategy on foreign keys
- **Data Cleaning**: Deduplication, missing value handling, validation
- **CTEs**: Complex multi-step queries
- **Custom Functions**: PL/pgSQL for risk scoring
- **View Creation**: Data abstraction layers

### Python & Software Engineering
- **Modular Design**: Separation of concerns (data, training, prediction)
- **CLI Tools**: argparse for command-line interfaces
- **Error Handling**: Try-except blocks with informative messages
- **Code Organization**: 3 library modules in `src/`, CLI scripts in `scripts/`, offline tests in `tests/`
- **Documentation**: Comprehensive README, docstrings, comments

### Data Engineering
- **ETL Pipeline**: Multi-table joins, aggregation, feature creation
- **Data Validation**: Type checks, constraint enforcement
- **Performance Optimization**: Efficient SQL queries with indexes
- **Database Design**: 3NF normalized schema

---

## Project Structure Details

### Python Modules

**`src/data_loader.py`**
- Finds DfT CSVs for any year (short or official file names)
- Leakage-free loader: collision file only, 13 pre-incident features + hour of day
- Retrospective loader: 3-table left joins, up to 17 features incl. post-crash casualty fields
- Missing value handling (DfT code -1 → median)
- Target encoding (severity 1,2,3 → 0,1,2)

**`src/trainer.py`**
- 4 model types (LR, RF, XGB, MLP)
- SMOTE/class_weight strategies
- Stratified K-Fold cross-validation
- Model comparison framework
- Model persistence (joblib)

**`src/time_series_predictor.py`**
- 4 forecasting models
- Daily/hourly aggregation
- Missing period imputation
- Metric calculation (MAE, RMSE, MAPE, AIC, BIC)
- Model saving/loading

**`scripts/quick_demo.py`**
- Synthetic data generation
- Realistic accident patterns (trend + seasonality)
- No external data required
- Demonstrates severity classification and a naive forecasting baseline; exits non-zero on failure

**`scripts/train_models.py`**
- CLI training interface
- Task selection (severity, timeseries, all)
- Model selection with argparse
- Progress reporting

### SQL Files

**`sql/00_schema.sql`**
- 3 table definitions
- Primary/foreign keys
- Index creation (10 indexes)

**`sql/01_data_cleaning.sql`**
- Missing value replacement
- Duplicate removal (window functions)
- Data validation (future dates, invalid times)
- Quality reporting

**`sql/02_feature_engineering.sql`**
- 10+ temporal features (hour, day_of_week, is_weekend)
- Categorical labeling (weather, light, road_surface)
- 4 aggregated views (hourly, daily, monthly, risk_scores)

**`sql/03_temporal_analysis.sql`**
- 7-day rolling averages
- Month-over-month comparisons
- High-risk scenario identification
- Statistical significance filtering

**`sql/04_risk_prediction.sql`**
- Custom PL/pgSQL function
- Composite risk scoring
- ML dataset export view

---

## Limitations & Future Work

### Current Limitations
1. Classical time-series models (ARIMA/SARIMA/Prophet) still report in-sample
   fit; only the LSTM path uses a held-out chronological backtest
2. Severity classifier uses CV; time-series LSTM uses a chronological hold-out
3. No logging framework (uses print statements)
4. No type hints in Python code
5. Tests are offline and use synthetic data; the SQL files are not exercised by tests
6. The retrospective (leaky) severity scores are historical and not re-run here (needs
   vehicle + casualty files)

### Planned Enhancements
1. **Backtest the classical models** too, for an apples-to-apples comparison with the LSTM
2. **API Wrapper**: FastAPI REST service
3. **Docker**: Containerization
4. **Type Hints**: Python 3.10+ annotations
5. **Hyperparameter Tuning**: GridSearchCV for all models
6. **Feature Importance**: SHAP analysis
7. **Dashboard**: Streamlit visualization
8. **Model Registry**: MLflow tracking

---

## Data Source

**UK Department for Transport Road Safety Data**
- 2025 provisional extract: 48,472 collisions (SQL schema, original write-up)
- Full-year 2023 collision file: 104,258 collisions (all re-run numbers)
- 3 tables: collision, vehicle, casualty
- Public domain, open data license
- Link: https://www.data.gov.uk/dataset/road-accidents-safety-data

---

## License

MIT License - Free to use and modify for educational and commercial purposes.

---

## Author

Built as a data-science portfolio project demonstrating:
- End-to-end pipeline (data → SQL → ML → analysis)
- Modular, readable code
- Retrospective analysis of a real-world public dataset (UK road safety)
- Advanced SQL and Python techniques

---

## Quick Reference

### Command Cheat Sheet

```bash
# Synthetic Demo (No Downloads)
python scripts/quick_demo.py

# Collision CSV only
python scripts/train_severity_leakfree.py --model rf
python scripts/train_models.py --task timeseries --model lstm --freq hourly
python scripts/naive_baselines.py
python scripts/data_insights.py

# Needs collision + vehicle + casualty CSVs
python scripts/train_models.py --task severity --model rf
python scripts/train_models.py --task all

# SQL Setup (load the CSVs yourself; see "SQL Analytics" above)
createdb traffic_risk
psql traffic_risk -f sql/00_schema.sql
psql traffic_risk -f sql/01_data_cleaning.sql

# Run Tests (offline, no data needed)
pytest -v
```

### Model Performance Summary

| Task | Best Model | Metric | Value |
|------|-----------|--------|-------|
| Severity (leakage-free, ahead-of-time) | Random Forest | macro-F1 | 0.350 |
| Severity (retrospective, leaky; historical, 2025 extract, not re-run) | Random Forest | macro-F1 | 0.762 |
| Hourly Forecasting (held-out backtest) | LSTM (PyTorch) | MAE | 3.45 |

---

## Contact

For questions, issues, or collaborations, please open an issue on GitHub.
