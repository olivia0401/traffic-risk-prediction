# SQL Analysis Scripts

SQL implementation of the Python data analysis pipeline using PostgreSQL.

## Files

| File | Purpose |
|------|---------|
| `00_schema.sql` | Create database tables |
| `01_data_cleaning.sql` | Clean and deduplicate data |
| `02_feature_engineering.sql` | Create temporal features and labels |
| `03_temporal_analysis.sql` | Time series analysis with rolling windows |
| `04_risk_prediction.sql` | Risk scoring and ML dataset export |

## Quick Start

```bash
# Create database
createdb traffic_risk

# Run scripts in order
psql traffic_risk -f 00_schema.sql
psql traffic_risk -f 01_data_cleaning.sql
psql traffic_risk -f 02_feature_engineering.sql
psql traffic_risk -f 03_temporal_analysis.sql
psql traffic_risk -f 04_risk_prediction.sql
```

## Key SQL Skills

- Window functions (rolling averages)
- CTEs and subqueries
- Multi-table JOINs
- Feature engineering with CASE statements
- Time series aggregation
- Custom functions for risk prediction

## Example Query

```sql
-- Get peak accident hours
SELECT hour_of_day, accident_count
FROM hourly_accident_stats
ORDER BY accident_count DESC
LIMIT 5;
```
