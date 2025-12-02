-- ============================================================================
-- TEMPORAL ANALYSIS QUERIES
-- ============================================================================
-- Purpose: Time series analysis queries for accident prediction
-- Replicates Python temporal analysis and rolling window calculations
-- ============================================================================

-- ============================================================================
-- 1. PEAK HOUR ANALYSIS
-- ============================================================================

-- Top 5 highest-risk hours by accident count
SELECT
    hour_of_day,
    accident_count,
    ROUND(100.0 * accident_count / SUM(accident_count) OVER (), 2) as pct_of_total,
    fatal_count,
    serious_count
FROM hourly_accident_stats
ORDER BY accident_count DESC
LIMIT 5;

-- ============================================================================
-- 2. DAILY ACCIDENT TRENDS WITH 7-DAY ROLLING AVERAGE
-- ============================================================================
-- Replicates pandas rolling(7).mean()

WITH daily_counts AS (
    SELECT
        date_of_accident,
        COUNT(*) as daily_accidents,
        AVG(number_of_casualties) as avg_daily_casualties
    FROM collision
    GROUP BY date_of_accident
)
SELECT
    date_of_accident,
    daily_accidents,
    avg_daily_casualties,

    -- 7-day rolling average
    AVG(daily_accidents) OVER (
        ORDER BY date_of_accident
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as rolling_7day_avg,

    -- 7-day rolling max
    MAX(daily_accidents) OVER (
        ORDER BY date_of_accident
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as rolling_7day_max,

    -- Month-over-month change
    LAG(daily_accidents, 30) OVER (ORDER BY date_of_accident) as accidents_30days_ago,
    daily_accidents - LAG(daily_accidents, 30) OVER (ORDER BY date_of_accident) as month_change

FROM daily_counts
ORDER BY date_of_accident;

-- ============================================================================
-- 3. 15-MINUTE SLOT ANALYSIS
-- ============================================================================

-- Accidents by 15-minute time slots (96 slots per day)
SELECT
    time_slot_15min,
    LPAD((time_slot_15min / 4)::TEXT, 2, '0') || ':' ||
    LPAD(((time_slot_15min % 4) * 15)::TEXT, 2, '0') as time_label,
    COUNT(*) as accident_count,
    AVG(number_of_casualties) as avg_casualties,
    SUM(CASE WHEN collision_severity IN (1, 2) THEN 1 ELSE 0 END) as serious_fatal_count
FROM collision_with_features
GROUP BY time_slot_15min
ORDER BY time_slot_15min;

-- ============================================================================
-- 4. RUSH HOUR VS OFF-PEAK COMPARISON
-- ============================================================================

SELECT
    time_period,
    COUNT(*) as accident_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct_of_total,
    AVG(number_of_casualties) as avg_casualties,
    AVG(number_of_vehicles) as avg_vehicles,

    -- Severity breakdown
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) as fatal,
    SUM(CASE WHEN collision_severity = 2 THEN 1 ELSE 0 END) as serious,
    SUM(CASE WHEN collision_severity = 3 THEN 1 ELSE 0 END) as slight

FROM collision_with_features
GROUP BY time_period
ORDER BY accident_count DESC;

-- ============================================================================
-- 5. WEEKEND VS WEEKDAY ANALYSIS
-- ============================================================================

SELECT
    CASE WHEN is_weekend THEN 'Weekend' ELSE 'Weekday' END as day_type,
    COUNT(*) as accident_count,
    AVG(number_of_casualties) as avg_casualties,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as fatal_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 2 THEN 1 ELSE 0 END) / COUNT(*), 2) as serious_rate_pct
FROM collision_with_features
GROUP BY is_weekend
ORDER BY is_weekend;

-- ============================================================================
-- 6. SEASONAL TRENDS (MONTHLY AGGREGATION)
-- ============================================================================

SELECT
    month,
    month_name,
    COUNT(*) as accident_count,
    AVG(number_of_casualties) as avg_casualties,

    -- Compare to previous month
    LAG(COUNT(*)) OVER (ORDER BY month) as prev_month_count,
    COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY month) as month_on_month_change,

    -- Severity distribution
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as fatal_pct,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 2 THEN 1 ELSE 0 END) / COUNT(*), 2) as serious_pct

FROM collision_with_features
GROUP BY month, month_name
ORDER BY month;

-- ============================================================================
-- 7. HIGH-RISK TIME WINDOWS (FOR AUTONOMOUS DRIVING)
-- ============================================================================
-- Identify specific hour + day combinations with highest accident rates

SELECT
    day_name,
    hour_of_day,
    COUNT(*) as accident_count,
    AVG(number_of_casualties) as avg_casualties,

    -- Risk classification
    CASE
        WHEN COUNT(*) >= 300 THEN 'Critical Risk'
        WHEN COUNT(*) >= 200 THEN 'High Risk'
        WHEN COUNT(*) >= 100 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as risk_classification,

    -- Environmental factors
    ROUND(100.0 * SUM(CASE WHEN weather_conditions IN (2, 3, 5, 6, 7) THEN 1 ELSE 0 END) / COUNT(*), 1) as adverse_weather_pct,
    ROUND(100.0 * SUM(CASE WHEN light_conditions IN (4, 5, 6) THEN 1 ELSE 0 END) / COUNT(*), 1) as darkness_pct

FROM collision_with_features
GROUP BY day_name, day_of_week, hour_of_day
HAVING COUNT(*) >= 50  -- Filter for statistical significance
ORDER BY accident_count DESC
LIMIT 20;

-- ============================================================================
-- 8. TIME-BASED FORECASTING FEATURES
-- ============================================================================
-- Create lagged features for time series prediction

WITH daily_features AS (
    SELECT
        date_of_accident,
        COUNT(*) as accidents,
        AVG(number_of_casualties) as avg_casualties,
        EXTRACT(DOW FROM date_of_accident) as dow,
        EXTRACT(MONTH FROM date_of_accident) as month
    FROM collision
    GROUP BY date_of_accident
)
SELECT
    date_of_accident,
    accidents as target,

    -- Lagged features (for forecasting)
    LAG(accidents, 1) OVER (ORDER BY date_of_accident) as lag_1day,
    LAG(accidents, 7) OVER (ORDER BY date_of_accident) as lag_7days,
    LAG(accidents, 30) OVER (ORDER BY date_of_accident) as lag_30days,

    -- Rolling statistics
    AVG(accidents) OVER (
        ORDER BY date_of_accident
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) as rolling_7day_mean,

    STDDEV(accidents) OVER (
        ORDER BY date_of_accident
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) as rolling_7day_stddev,

    -- Calendar features
    dow,
    month,
    CASE WHEN dow IN (0, 6) THEN 1 ELSE 0 END as is_weekend

FROM daily_features
ORDER BY date_of_accident;
