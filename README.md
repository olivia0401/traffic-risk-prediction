# UK Road Casualty Severity Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)](https://www.postgresql.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine-learning and data-engineering study of UK road-safety records: retrospective
**casualty-severity classification** (how serious was a casualty, given a recorded collision)
plus **temporal analysis** of when accidents cluster. Built on 48,472 UK Department for
Transport accident records, combining SQL analytics, time-series forecasting, and
multi-class classification.

> **Scope & honesty note.** This is a *post-hoc* analysis of accidents that already
> happened, **not** a pre-accident or real-time risk predictor. The severity model uses
> casualty attributes (`age_of_casualty`, `sex_of_casualty`, `casualty_class`,
> `casualty_type`) that are only known **after** a collision is recorded, so the reported
> scores are not achievable at deployment time for prediction *before* an accident. Treat
> the results as descriptive/analytical, not as a safety system for autonomous vehicles.
> See [Feature scope & leakage](#feature-scope--leakage).

## Overview

This project demonstrates a full data-science pipeline over real-world road-safety data:
temporal analysis (when do accidents cluster?) and casualty-severity classification
(given a recorded collision, how serious was the casualty?).

**Dataset:** UK DfT road-safety open data. The SQL and temporal write-up uses a 2025 provisional extract (48,472 collisions); the ML tables were rerun on the full-year 2023 file (104,258 collisions), so those numbers reproduce straight from `data/`.
**Total Code:** 2,355 lines (1,367 Python + 988 SQL)
**Status:** Functional, with a synthetic-data demo for quick runs

---

## Project Highlights

- **Multi-Table Data Integration**: 3-table joins (collision + vehicle + casualty) → 95,526 records
- **Advanced SQL Analytics**: 988 lines with window functions, CTEs, and custom risk scoring
- **Time Series Forecasting**: 4 approaches (ARIMA, SARIMA, Prophet, LSTM)
- **Imbalanced Classification**: Multi-class severity prediction with SMOTE
- **Production-Ready Code**: Modular design, CLI tools, comprehensive documentation
- **No-Download Demo**: Synthetic data generator for instant testing

---

## Architecture

### Project Structure

```
traffic-risk-prediction/
├── src/                           # Core ML modules (772 LOC)
│   ├── data_loader.py            # Multi-table data integration
│   ├── trainer.py                # Severity classification models
│   └── time_series_predictor.py  # Time series forecasting (4 approaches)
│
├── scripts/                       # Executable workflows (490 LOC)
│   ├── quick_demo.py             # Synthetic data demo (no downloads)
│   ├── train_models.py           # Full training pipeline with CLI
│   └── train_severity.py         # Severity-focused training
│
├── sql/                           # Database layer (988 LOC)
│   ├── 00_schema.sql             # Table definitions & indexes (142 LOC)
│   ├── 01_data_cleaning.sql      # Data validation & deduplication (153 LOC)
│   ├── 02_feature_engineering.sql # Views & temporal features (243 LOC)
│   ├── 03_temporal_analysis.sql  # Time series queries (202 LOC)
│   └── 04_risk_prediction.sql    # Risk scoring functions (248 LOC)
│
├── notebooks/                     # Jupyter EDA
├── data/                         # CSV storage
├── models/                       # Trained model artifacts
└── README.md                     # Documentation
```

---

## Key Results & Insights

### Data Insights

| Finding | Detail | Interpretation |
|---------|--------|----------------|
| **Peak count: 3-6 PM** | 42% of daily accidents | Rush-hour concentration |
| **Fine-weather share** | 67% of accidents in fine weather | Reflects exposure (most driving is in fine weather), not weather safety |
| **Vulnerable users** | Motorcycles 75% higher severity | Two-wheeler casualties skew more severe |
| **Friday effect** | Highest accident count | End-of-week temporal pattern |

### Model Performance

**Time Series Forecasting:**

| Model | MAE | RMSE | Use Case |
|-------|-----|------|----------|
**Hourly forecasting — held-out backtest on real 2023 DfT data** (8,760 hours, last 20% kept back for testing):

| Model | MAE | RMSE |
|-------|-----|------|
| same hour yesterday (naive) | 4.92 | 6.97 |
| same hour last week (naive) | 4.49 | 6.42 |
| **LSTM (PyTorch, 2-layer)** | **3.45** | **4.80** |

The LSTM beats both naive baselines. It's trained on the earlier part of the year and scored on the most recent slice, so there's no peeking. Reproduce with `python scripts/train_models.py --task timeseries --model lstm --freq hourly`.

The old daily ARIMA/Prophet numbers were in-sample, so I've left them out of this comparison.

**Severity classification — two versions.**

Predicting how serious a collision is *before* it happens, from road / weather / time / junction context only (5-fold CV, real 2023 data):

| Model | macro-F1 (CV) | macro-recall (CV) |
|-------|---------------|-------------------|
| Logistic Regression | 0.311 ± 0.002 | 0.463 |
| **Random Forest** | **0.350 ± 0.002** | 0.366 |
| XGBoost | 0.339 ± 0.003 | 0.418 |

An earlier version scored ~0.76, but it used fields you only know *after* a crash — the casualty's age, sex and injury type. That's leakage: it describes accidents instead of predicting them. Drop those fields and the honest number is ~0.35. Code: `scripts/train_severity_leakfree.py`.

**Class balance (real 2023 data):** Fatal 1.5% · Serious 22.5% · Slight 76% (104,258 collisions), handled with SMOTE + class weights.

---

## Machine Learning Implementation

### 1. Severity Classification (Multi-Class Imbalanced Learning)

**Location:** `src/trainer.py` (196 LOC)

**Challenge:** Highly imbalanced casualty severity (5% fatal, 25% serious, 70% slight)

**Solution:**
```python
# SMOTE oversampling pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(class_weight='balanced'))
])
```

> Heads up — the F1 numbers below are the leaky retrospective ones (they include post-crash casualty fields). The honest leakage-free scores are up in [Model Performance](#model-performance).

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

**Location:** `src/time_series_predictor.py` (490 LOC)

**Data Aggregation:**
- Daily: Groups accidents by calendar day
- Hourly: 24-hour seasonality detection
- Missing period imputation with zeros

**Model 1: ARIMA(5,1,2)**
```python
# AutoRegressive Integrated Moving Average
order=(5, 1, 2)  # p=5 AR lags, d=1 differencing, q=2 MA lags
# Use case: Short-term forecasting (1-3 days)
# Demo result: MAE=2.84
```

**Model 2: SARIMA(1,1,1,24)**
```python
# Seasonal ARIMA for hourly patterns
order=(1,1,1), seasonal_order=(1,1,1,24)
# Captures rush hour patterns (7-9 AM, 3-6 PM)
# Demo result: MAE=2.31 (best for hourly)
```

**Model 3: Prophet (Meta's Forecasting Library)**
```python
# Flexible multi-level seasonality
Prophet(yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05)
# Advantages: Handles holidays, robust to outliers
# Demo result: MAE=2.19 (BEST overall)
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

**Metrics Used:** MAE, RMSE, MAPE, AIC, BIC

---

## SQL Data Pipeline

**Total:** 988 lines across 5 SQL files

### Schema Design (`00_schema.sql`)

**Tables:**
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
git clone <your-repo-url>
cd traffic-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run synthetic demo
python scripts/quick_demo.py

# Output: Trains models on synthetic data, shows performance metrics
```

### Full Pipeline with Real Data

```bash
# 1. Download UK DfT accident data
# https://www.data.gov.uk/dataset/road-accidents-safety-data
# Place collision.csv, vehicle.csv, casualty.csv in data/

# 2. Train all models
python scripts/train_models.py --task all

# Train specific model
python scripts/train_models.py --task severity --model rf
python scripts/train_models.py --task timeseries --model prophet --freq daily

# 3. Models saved to models/ directory
```

### SQL Analytics

```bash
# Setup PostgreSQL database
createdb traffic_risk

# Load schema and data
psql traffic_risk -f sql/00_schema.sql
psql traffic_risk -f sql/01_data_cleaning.sql
psql traffic_risk -f sql/02_feature_engineering.sql

# Run temporal analysis
psql traffic_risk -f sql/03_temporal_analysis.sql

# Query risk scores
psql traffic_risk -c "SELECT * FROM get_risk_score(17, 5, 'Fine', 'Daylight');"
```

---

## Technical Stack

### Python Dependencies

| Library | Purpose |
|---------|---------|
| **Data Processing** | |
| pandas ≥1.5 | DataFrames, time series |
| numpy ≥1.23 | Numerical computing |
| psycopg2 ≥2.9 | PostgreSQL driver |
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
| pytest ≥7.2 | Unit testing (planned) |
| jupyter | Notebooks |

### Database
- PostgreSQL 12+
- 3 tables with proper indexing
- Window functions, CTEs, custom functions

---

## Code Examples

### Training Severity Classifier

```python
from src.trainer import SeverityClassifier
from src.data_loader import load_and_prepare_data

# Load data
X, y, features = load_and_prepare_data('data')

# Train Random Forest with SMOTE
classifier = SeverityClassifier(model_type='rf')
classifier.compare_models(X, y, cv=5)

# Save best model
classifier.train(X, y)
classifier.save_model('models/severity_model.pkl')
```

### Time Series Forecasting

```python
from src.time_series_predictor import TimeSeriesPredictor
import pandas as pd

# Prepare daily data
collision_df = pd.read_csv('data/collision.csv')
ts = predictor.prepare_daily_data(collision_df)

# Train Prophet model
predictor = TimeSeriesPredictor(model_type='prophet')
predictor.train(ts)

# Forecast next 7 days
future = predictor.forecast(steps=7)
print(f"Next week avg accidents: {future.mean():.1f}")
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
- **Code Organization**: Clear project structure with 8 modules
- **Documentation**: Comprehensive README, docstrings, comments

### Data Engineering
- **ETL Pipeline**: Multi-table joins, aggregation, feature creation
- **Data Validation**: Type checks, constraint enforcement
- **Performance Optimization**: Efficient SQL queries with indexes
- **Database Design**: 3NF normalized schema

---

## Project Structure Details

### Python Modules

**`src/data_loader.py` (81 LOC)**
- Multi-table integration (3-table left joins)
- Missing value handling (-1, 99 → median)
- Feature selection (17 features)
- Target encoding (severity 1,2,3 → 0,1,2)

**`src/trainer.py` (196 LOC)**
- 4 model types (LR, RF, XGB, MLP)
- SMOTE/class_weight strategies
- Stratified K-Fold cross-validation
- Model comparison framework
- Model persistence (joblib)

**`src/time_series_predictor.py` (490 LOC)**
- 4 forecasting models
- Daily/hourly aggregation
- Missing period imputation
- Metric calculation (MAE, RMSE, MAPE, AIC, BIC)
- Model saving/loading

**`scripts/quick_demo.py` (201 LOC)**
- Synthetic data generation
- Realistic accident patterns (trend + seasonality)
- No external data required
- Demonstrates all features

**`scripts/train_models.py` (215 LOC)**
- CLI training interface
- Task selection (severity, timeseries, all)
- Model selection with argparse
- Progress reporting

### SQL Files

**`sql/00_schema.sql` (142 LOC)**
- 3 table definitions
- Primary/foreign keys
- Index creation (7 indexes)

**`sql/01_data_cleaning.sql` (153 LOC)**
- Missing value replacement
- Duplicate removal (window functions)
- Data validation (future dates, invalid times)
- Quality reporting

**`sql/02_feature_engineering.sql` (243 LOC)**
- 10+ temporal features (hour, day_of_week, is_weekend)
- Categorical labeling (weather, light, road_surface)
- 4 aggregated views (hourly, daily, monthly, risk_scores)

**`sql/03_temporal_analysis.sql` (202 LOC)**
- 7-day rolling averages
- Month-over-month comparisons
- High-risk scenario identification
- Statistical significance filtering

**`sql/04_risk_prediction.sql` (248 LOC)**
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
5. No unit tests

### Planned Enhancements
1. **Backtest the classical models** too, for an apples-to-apples comparison with the LSTM
2. **API Wrapper**: FastAPI REST service
3. **Docker**: Containerization
4. **Testing**: pytest unit tests
5. **Type Hints**: Python 3.10+ annotations
6. **Hyperparameter Tuning**: GridSearchCV for all models
7. **Feature Importance**: SHAP analysis
8. **Dashboard**: Streamlit visualization
9. **Model Registry**: MLflow tracking

---

## Data Source

**UK Department for Transport Road Safety Data**
- 48,472 accident records (2025 dataset)
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

# Train All Models (Requires Real Data)
python scripts/train_models.py --task all

# Train Specific Models
python scripts/train_models.py --task severity --model rf
python scripts/train_models.py --task timeseries --model prophet --freq daily

# SQL Setup
createdb traffic_risk
psql traffic_risk -f sql/00_schema.sql
psql traffic_risk -f sql/01_data_cleaning.sql

# Run Tests (When Implemented)
pytest -v
```

### Model Performance Summary

| Task | Best Model | Metric | Value |
|------|-----------|--------|-------|
| Severity (leakage-free, ahead-of-time) | Random Forest | macro-F1 | 0.350 |
| Severity (retrospective, leaky) | Random Forest | macro-F1 | 0.762 |
| Hourly Forecasting (held-out backtest) | LSTM (PyTorch) | MAE | 3.45 |

### File Locations

- Main code: `/home/olivia/traffic-risk-prediction/src/`
- Scripts: `/home/olivia/traffic-risk-prediction/scripts/`
- SQL: `/home/olivia/traffic-risk-prediction/sql/`
- Models: `/home/olivia/traffic-risk-prediction/models/`
- Data: `/home/olivia/traffic-risk-prediction/data/`

---

## Contact

For questions, issues, or collaborations, please open an issue on GitHub.
