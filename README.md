# Traffic Risk Prediction for Autonomous Driving

Predict high-risk time windows using UK traffic collision data to support safer autonomous driving decisions.

## 🎯 Business Objective

Forecast traffic accident risk levels in specific time periods (hour/day/week) using historical UK Department for Transport (DfT) collision data.

## 📊 Dataset

- **Source**: UK Department for Transport (DfT) Road Casualty Statistics - Provisional 2025
- **Download**: [Official UK Gov Data Portal](https://www.data.gov.uk/dataset/road-accidents-safety-data)
- **Size**: 48,472 accidents, 87,805 vehicles, 60,991 casualties (95,526 total records after merge)
- **Time Range**: 2025 provisional data
- **Features**: Weather, road conditions, vehicle types, temporal patterns, severity levels

## 🔍 Key Findings

- **Peak Risk Hours**: 3:00-6:00 PM (evening rush)
- **Highest Risk Day**: Friday
- **Counterintuitive**: Most accidents occur in **fine weather on dry roads**
- **Vulnerable Users**: Motorcycles and pedal cycles have higher casualty severity

## 📁 Project Structure

```
traffic-risk-prediction/
├── notebooks/
│   └── traffic_risk_eda.ipynb    # Python EDA with pandas, seaborn
├── sql/
│   ├── 00_schema.sql              # Database schema
│   ├── 01_data_cleaning.sql       # Data cleaning queries
│   ├── 02_feature_engineering.sql # Feature creation
│   ├── 03_temporal_analysis.sql   # Time series analysis
│   └── 04_risk_prediction.sql     # Risk scoring functions
├── data/
└── README.md
```

## 🛠️ Technologies

### Python Stack
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: statsmodels (VIF)
- **Time Series**: Rolling windows, temporal aggregation

### SQL Stack
- **Database**: PostgreSQL 12+
- **Features**: Window functions, CTEs, custom functions
- **Analysis**: Multi-table JOINs, temporal aggregation

## 🚀 Quick Start

### Python Analysis

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# Download 2025 data from UK Gov portal
# Place CSV files in data/ directory:
# - collision_2025.csv
# - vehicle_2025.csv
# - casualty_2025.csv

# Run notebook
jupyter notebook notebooks/traffic_risk_eda.ipynb
```

### SQL Analysis

```bash
# Create database
createdb traffic_risk

# Run SQL scripts
cd sql/
psql traffic_risk -f 00_schema.sql
psql traffic_risk -f 01_data_cleaning.sql
psql traffic_risk -f 02_feature_engineering.sql
```

## 📈 Analysis Highlights

### Temporal Patterns
- 15-minute granularity accident distribution
- 7-day rolling average trends
- Weekday vs weekend comparison
- Seasonal variations across 2025 data

### Risk Factors
- Environmental conditions (weather, lighting, road surface)
- Vehicle type risk profiling
- Multi-collinearity detection (VIF analysis)
- Feature importance for modeling

### Data Quality
- Missing value treatment (-1 coded values → NULL)
- Duplicate detection and removal
- Outlier analysis (IQR method)
- 3-table merge strategy (collision + vehicle + casualty)

## 💡 Skills Demonstrated

**Data Science**
- Exploratory Data Analysis (EDA)
- Time series analysis
- Statistical testing (VIF, correlation)
- Feature engineering

**SQL**
- Database design and normalization
- Window functions and CTEs
- Multi-table JOINs
- Query optimization

**Visualization**
- Temporal trends (line plots, bar charts)
- Distribution analysis (histograms, boxplots)
- Categorical relationships (grouped bars, heatmaps)

## 📝 Future Work

- Time series forecasting models (ARIMA, Prophet)
- Machine learning classification (severity prediction)
- Geospatial analysis (hotspot mapping)
- Real-time risk API for autonomous vehicles

## 👤 Author

Created for demonstrating end-to-end data analysis capabilities in ML/AI Engineer roles.
