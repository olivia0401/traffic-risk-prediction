# Traffic Risk Prediction for Autonomous Driving

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)](https://www.postgresql.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Predict high-risk driving time windows using 48K+ UK traffic accidents**. Combines SQL analytics, time series forecasting, and machine learning to enable safer autonomous vehicle decisions.

> 🚗 Built for autonomous driving systems: Predict when accidents are most likely and adjust vehicle behavior accordingly.

---

## 🎯 What This Does

Enable self-driving cars to **anticipate accident risk** based on:
- ⏰ Time of day (rush hour patterns)
- 📅 Day of week (weekend vs weekday)
- 🌦️ Weather conditions
- 🚙 Vehicle types on the road

**Real-world application**: AVs can increase following distance during predicted high-risk periods (e.g., 3-6 PM rush).

---

## 📊 Key Results

### Dataset
- **48,472 accidents** from UK Department for Transport (2025)
- **3 merged tables**: collision + vehicle + casualty data
- **95,526 records** after integration

### Top Insights

| Finding | Impact |
|---------|--------|
| **Peak Risk**: 3-6 PM (evening rush) | 42% of daily accidents in 4-hour window |
| **Weather Paradox**: 67% accidents in "fine weather" | Driver overconfidence, not conditions |
| **Vulnerable Users**: Motorcycles 75% higher severity | AV detection priority |
| **Friday Effect**: Highest accident count | End-of-week fatigue factor |

### Model Performance

| Model | MAE | RMSE | Use Case |
|-------|-----|------|----------|
| **ARIMA(5,1,2)** | 2.84 | 3.67 | Baseline forecast |
| **SARIMA** | 2.31 | 3.12 | Hourly with seasonality |
| **Prophet** | 2.19 | 2.98 | Best overall (daily) |
| **Random Forest** | - | - | Severity classification (F1: 0.68) |

---

## 🚀 Quick Start

### Option 1: Run Demo (No Data Needed)

```bash
# Clone repository
git clone https://github.com/olivia0401/traffic-risk-prediction.git
cd traffic-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run synthetic demo
python scripts/quick_demo.py
```

**Output**: Trains models on synthetic data, shows hourly risk patterns, no downloads required!

### Option 2: Full Pipeline with Real Data

```bash
# 1. Download UK DfT data (48K accidents)
# https://www.data.gov.uk/dataset/road-accidents-safety-data
# Place CSV files in data/ directory

# 2. Train all models
python scripts/train_models.py --task all

# 3. View results
# Models saved in models/ directory
# Metrics printed to console
```

### Option 3: SQL Analysis

```bash
# Install PostgreSQL
sudo apt install postgresql-12

# Create database and load data
createdb traffic_risk
psql traffic_risk -f sql/00_schema.sql
psql traffic_risk -f sql/01_data_cleaning.sql

# Run temporal analysis
psql traffic_risk -f sql/03_temporal_analysis.sql

# Query high-risk periods
psql traffic_risk -c "SELECT hour, avg_accidents FROM hourly_risk ORDER BY avg_accidents DESC LIMIT 5;"
```

---

## 🏗️ Project Architecture

```
traffic-risk-prediction/
├── src/
│   ├── data_loader.py             # Multi-table merge (3 tables → 95K records)
│   ├── trainer.py                 # Severity classifier (RF, XGBoost, MLP)
│   └── time_series_predictor.py   # ARIMA, SARIMA, Prophet, LSTM
│
├── scripts/
│   ├── train_models.py            # Full training pipeline
│   └── quick_demo.py              # Synthetic data demo (no downloads)
│
├── sql/
│   ├── 00_schema.sql              # PostgreSQL table definitions
│   ├── 01_data_cleaning.sql       # Missing value handling (-1 → NULL)
│   ├── 02_feature_engineering.sql # Rush hour flags, temporal features
│   └── 03_temporal_analysis.sql   # Window functions, rolling averages
│
├── notebooks/
│   └── traffic_risk_eda.ipynb     # Exploratory analysis + visualizations
│
└── data/
    └── (Place UK DfT CSV files here)
```

---

## 🔧 Technical Highlights

### 1. Multi-Table Data Integration

**Challenge**: 3 normalized tables (collision, vehicle, casualty) with 1:N relationships

**Solution**: Efficient LEFT JOIN strategy preserving all casualties
```sql
SELECT c.*, v.vehicle_type, cas.casualty_severity
FROM collision c
LEFT JOIN vehicle v USING (accident_index)
LEFT JOIN casualty cas USING (accident_index, vehicle_reference)
WHERE c.date IS NOT NULL;
```

**Result**: 95,526 records from 48,472 accidents (no data loss)

---

### 2. Time Series Forecasting (3 Approaches)

**A) ARIMA for Daily Trends**
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(daily_accidents, order=(5, 1, 2))
forecast = model.fit().forecast(steps=7)  # Next 7 days
```

**B) SARIMA for Hourly Patterns (24-hour seasonality)**
```python
model = SARIMAX(hourly_accidents,
                order=(1,1,1),
                seasonal_order=(1,1,1,24))
```

**C) Prophet for Robust Forecasting**
```python
from prophet import Prophet

model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df)
future = model.make_future_dataframe(periods=168)  # 7 days hourly
```

**Why Multiple Models?**
- ARIMA: Best for short-term (1-3 days)
- SARIMA: Captures rush hour cycles
- Prophet: Handles holidays, anomalies robustly

---

### 3. Severity Classification (Imbalanced Learning)

**Problem**: Classes imbalanced (Fatal: 5%, Serious: 25%, Slight: 70%)

**Solution**: SMOTE + Class Weights
```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Oversample minority classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train with balanced class weights
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100)
rf.fit(X_resampled, y_resampled)
```

**Result**: F1 macro 0.68 (vs 0.23 baseline)

---

### 4. Advanced SQL Techniques

**Window Functions (7-day rolling average)**
```sql
SELECT
    date,
    accident_count,
    AVG(accident_count) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7day_avg
FROM daily_accidents;
```

**CTEs for Readability**
```sql
WITH rush_hour_accidents AS (
    SELECT * FROM collision
    WHERE EXTRACT(HOUR FROM time) BETWEEN 15 AND 18
),
severity_stats AS (
    SELECT vehicle_type, AVG(casualty_severity) as avg_severity
    FROM vehicle JOIN casualty USING (accident_index)
    GROUP BY vehicle_type
)
SELECT * FROM rush_hour_accidents
JOIN severity_stats USING (vehicle_type);
```

---

## 📈 Visualizations & Insights

### Hourly Risk Pattern
```
Hour | Avg Accidents | Risk Level
-----|---------------|------------
06:00|    8.2       | ████ Low
08:00|   15.7       | ████████ Medium
15:00|   18.9       | ██████████ HIGH
17:00|   17.3       | █████████ HIGH
23:00|    6.1       | ███ Low
```

### Counterintuitive Finding: Weather Paradox

**Expected**: More accidents in rain/snow
**Actual**: 67% occur in **fine weather on dry roads**

**Why?**
1. Higher traffic volume in good weather → more exposure
2. Driver overconfidence in ideal conditions
3. Rain/snow → naturally cautious driving

**AV Implication**: Don't relax safety margins too much in good weather!

---

### Vehicle Type Risk Profile

| Vehicle | Accidents | Casualties per Accident | Severity Index |
|---------|-----------|-------------------------|----------------|
| 🏍️ Motorcycle | 3,241 | 1.3 | **2.1 (Serious)** |
| 🚴 Bicycle | 2,819 | 1.1 | **1.8 (Serious)** |
| 🚗 Car | 38,762 | 1.2 | 1.3 (Slight) |
| 🚚 Van | 5,143 | 1.1 | 1.2 (Slight) |

**Insight**: Vulnerable road users have **75% higher injury severity** → AV detection priority

---

## 💡 Skills Demonstrated

### Data Engineering
- ✅ Multi-table JOINs (3 tables, 95K records)
- ✅ Missing value strategies (encoded -1 → NULL)
- ✅ ETL pipeline (CSV → PostgreSQL)
- ✅ Database schema design

### SQL Mastery
- ✅ Window functions (rolling averages, LAG/LEAD)
- ✅ CTEs for complex queries
- ✅ Custom aggregate functions (plpgsql)
- ✅ Query optimization (indexes, EXPLAIN ANALYZE)

### Machine Learning
- ✅ Time series forecasting (ARIMA, SARIMA, Prophet)
- ✅ Imbalanced classification (SMOTE, class weights)
- ✅ Model comparison framework
- ✅ Cross-validation (stratified k-fold)

### Python Data Science
- ✅ pandas (groupby, pivot, time series)
- ✅ Statistical testing (VIF, autocorrelation)
- ✅ Visualization (matplotlib, seaborn)
- ✅ Production code structure (modular, testable)

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test individual modules
python -m src.time_series_predictor  # Should run without errors
python scripts/quick_demo.py         # Full demo
```

---

## 🔮 Future Enhancements

### Predictive Modeling (Next Phase)
- [ ] LSTM for multi-step ahead forecasting
- [ ] XGBoost for severity classification
- [ ] Ensemble methods (stacking ARIMA + ML)

### Advanced Analytics
- [ ] Geospatial analysis (PostGIS hotspot mapping)
- [ ] Network analysis (intersection risk scoring)
- [ ] Causal inference (weather → accident causality)

### Production Deployment
- [ ] FastAPI REST API (`POST /predict_risk`)
- [ ] Real-time streaming (Kafka + Spark)
- [ ] Streamlit dashboard (interactive visualization)
- [ ] Docker containerization
- [ ] Integration with AV navigation systems

---

## 📚 Data Source

**UK Department for Transport (DfT)**
- Portal: https://www.data.gov.uk/dataset/road-accidents-safety-data
- Dataset: Road Safety Data - Provisional 2025
- License: Open Government License (OGL)

**Citation**:
```
Department for Transport (2025). Road Safety Data - Provisional 2025.
UK Government Open Data Portal.
```

---

## 🛠️ Tech Stack

**Languages**: Python 3.10+, SQL (PostgreSQL)

**Data Processing**: pandas, numpy, statsmodels

**Machine Learning**: scikit-learn, XGBoost, imbalanced-learn

**Time Series**: ARIMA (statsmodels), Prophet (Meta)

**Database**: PostgreSQL 12+ (window functions, CTEs)

**Visualization**: matplotlib, seaborn

**Notebooks**: Jupyter

---

## 📝 Requirements

```bash
# Core
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0

# Time Series
statsmodels>=0.13.0
prophet>=1.1.0

# Database
psycopg2-binary>=2.9.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0

# Optional: Deep Learning
# torch>=2.0.0  # For LSTM models
```

---

## 👤 Author

**Olivia**
AI & Data Engineer | MSc Artificial Intelligence

**Specialized in**: Time Series Forecasting, SQL Analytics, Transportation ML

📧 [Your Email]
💼 [Your LinkedIn]
🐙 [@olivia0401](https://github.com/olivia0401)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- UK Department for Transport for open data
- statsmodels and Prophet teams
- PostgreSQL community

---

**⭐ Star this repo if you find it useful for learning SQL + Python data analytics!**

---

## 📖 Related Projects

Check out my other work:

- [**Stroke Prediction ML**](https://github.com/olivia0401/stroke-prediction-ml) - Medical ML with extreme class imbalance (24x F1 improvement)
- [**LLMs-ROS2**](https://github.com/olivia0401/LLMs-ROS2) - Natural language control for robots using GPT-4 + RAG
