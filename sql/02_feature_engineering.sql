-- ============================================================================
-- FEATURE ENGINEERING QUERIES
-- ============================================================================
-- Purpose: Create derived features for machine learning and analysis
-- Replicates Python feature engineering in SQL
-- ============================================================================

-- ============================================================================
-- 1. CREATE ENRICHED COLLISION VIEW WITH TIME FEATURES
-- ============================================================================

CREATE OR REPLACE VIEW collision_with_features AS
SELECT
    c.*,

    -- Extract temporal features
    EXTRACT(HOUR FROM time_of_accident) as hour_of_day,
    EXTRACT(DOW FROM date_of_accident) as day_of_week,  -- 0=Sunday, 6=Saturday
    EXTRACT(MONTH FROM date_of_accident) as month,
    EXTRACT(YEAR FROM date_of_accident) as year,
    TO_CHAR(date_of_accident, 'Day') as day_name,
    TO_CHAR(date_of_accident, 'Month') as month_name,

    -- 15-minute time slots (0-95, representing 00:00 to 23:45)
    EXTRACT(HOUR FROM time_of_accident) * 4 +
    FLOOR(EXTRACT(MINUTE FROM time_of_accident) / 15) as time_slot_15min,

    -- Derived temporal flags
    CASE
        WHEN EXTRACT(HOUR FROM time_of_accident) BETWEEN 7 AND 9 THEN 'Morning Rush'
        WHEN EXTRACT(HOUR FROM time_of_accident) BETWEEN 15 AND 18 THEN 'Evening Rush'
        WHEN EXTRACT(HOUR FROM time_of_accident) BETWEEN 22 AND 6 THEN 'Night'
        ELSE 'Off-Peak'
    END as time_period,

    CASE
        WHEN EXTRACT(DOW FROM date_of_accident) IN (0, 6) THEN TRUE
        ELSE FALSE
    END as is_weekend,

    -- Categorical labels (human-readable)
    CASE collision_severity
        WHEN 1 THEN 'Fatal'
        WHEN 2 THEN 'Serious'
        WHEN 3 THEN 'Slight'
    END as severity_label,

    CASE weather_conditions
        WHEN 1 THEN 'Fine (no wind)'
        WHEN 2 THEN 'Rain (no wind)'
        WHEN 3 THEN 'Snow (no wind)'
        WHEN 4 THEN 'Fine (wind)'
        WHEN 5 THEN 'Rain (wind)'
        WHEN 6 THEN 'Snow (wind)'
        WHEN 7 THEN 'Fog/mist'
        WHEN 8 THEN 'Other'
        ELSE 'Unknown'
    END as weather_label,

    CASE light_conditions
        WHEN 1 THEN 'Daylight'
        WHEN 4 THEN 'Dark-lit'
        WHEN 5 THEN 'Dark-unlit'
        WHEN 6 THEN 'Dark-no lighting'
        WHEN 7 THEN 'Lighting unknown'
        ELSE 'Unknown'
    END as light_label,

    CASE road_surface_conditions
        WHEN 1 THEN 'Dry'
        WHEN 2 THEN 'Wet/Damp'
        WHEN 3 THEN 'Snow'
        WHEN 4 THEN 'Frost/Ice'
        WHEN 5 THEN 'Flood'
        WHEN 6 THEN 'Oil'
        WHEN 7 THEN 'Mud'
        ELSE 'Unknown'
    END as road_surface_label,

    CASE road_type
        WHEN 1 THEN 'Roundabout'
        WHEN 2 THEN 'One-way'
        WHEN 3 THEN 'Dual carriageway'
        WHEN 6 THEN 'Single carriageway'
        WHEN 7 THEN 'Slip Road'
        ELSE 'Unknown'
    END as road_type_label,

    CASE urban_or_rural_area
        WHEN 1 THEN 'Urban'
        WHEN 2 THEN 'Rural'
        ELSE 'Unknown'
    END as area_type_label

FROM collision c;

-- ============================================================================
-- 2. CREATE FULL ACCIDENT DATA VIEW (MERGED TABLES)
-- ============================================================================
-- Replicates the Python pd.merge() operations

CREATE OR REPLACE VIEW full_accident_data AS
SELECT
    -- Collision data
    c.*,

    -- Vehicle data
    v.vehicle_id,
    v.vehicle_reference,
    v.vehicle_type,
    v.vehicle_manoeuvre,
    v.sex_of_driver,
    v.age_of_driver,
    v.age_band_of_driver,
    v.driver_imd_decile,

    -- Vehicle labels
    CASE v.vehicle_type
        WHEN 1 THEN 'Motorcycle <=50cc'
        WHEN 2 THEN 'Motorcycle <=125cc'
        WHEN 3 THEN 'Motorcycle <=500cc'
        WHEN 4 THEN 'Motorcycle >500cc'
        WHEN 5 THEN 'Taxi'
        WHEN 8 THEN 'Minibus'
        WHEN 9 THEN 'Bus/Coach'
        WHEN 19 THEN 'Pedal cycle'
        ELSE 'Other'
    END as vehicle_type_label,

    -- Casualty data
    cas.casualty_id,
    cas.casualty_reference,
    cas.casualty_severity,
    cas.age_of_casualty,
    cas.sex_of_casualty,

    -- Casualty labels
    CASE cas.casualty_severity
        WHEN 1 THEN 'Fatal'
        WHEN 2 THEN 'Serious'
        WHEN 3 THEN 'Slight'
    END as casualty_severity_label

FROM collision_with_features c
LEFT JOIN vehicle v ON c.collision_index = v.collision_index
LEFT JOIN casualty cas ON c.collision_index = cas.collision_index
    AND v.vehicle_reference = cas.vehicle_reference;

-- ============================================================================
-- 3. CREATE AGGREGATED RISK FEATURES
-- ============================================================================

-- Hourly accident statistics
CREATE OR REPLACE VIEW hourly_accident_stats AS
SELECT
    hour_of_day,
    COUNT(*) as accident_count,
    AVG(number_of_casualties) as avg_casualties,
    AVG(number_of_vehicles) as avg_vehicles,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) as fatal_count,
    SUM(CASE WHEN collision_severity = 2 THEN 1 ELSE 0 END) as serious_count,
    SUM(CASE WHEN collision_severity = 3 THEN 1 ELSE 0 END) as slight_count
FROM collision_with_features
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- Day of week statistics
CREATE OR REPLACE VIEW daily_accident_stats AS
SELECT
    day_of_week,
    day_name,
    COUNT(*) as accident_count,
    AVG(number_of_casualties) as avg_casualties,
    ROUND(100.0 * SUM(CASE WHEN collision_severity IN (1, 2) THEN 1 ELSE 0 END) / COUNT(*), 2) as serious_fatal_pct
FROM collision_with_features
GROUP BY day_of_week, day_name
ORDER BY day_of_week;

-- Monthly trends
CREATE OR REPLACE VIEW monthly_accident_trends AS
SELECT
    year,
    month,
    month_name,
    COUNT(*) as accident_count,
    AVG(number_of_casualties) as avg_casualties,
    AVG(number_of_vehicles) as avg_vehicles
FROM collision_with_features
GROUP BY year, month, month_name
ORDER BY year, month;

-- ============================================================================
-- 4. RISK SCORING FEATURES
-- ============================================================================

-- Calculate risk score based on environmental conditions
CREATE OR REPLACE VIEW collision_risk_scores AS
SELECT
    collision_index,
    date_of_accident,
    time_of_accident,
    collision_severity,

    -- Risk factors (binary)
    CASE WHEN light_conditions IN (4, 5, 6) THEN 1 ELSE 0 END as dark_risk,
    CASE WHEN weather_conditions IN (2, 3, 5, 6, 7) THEN 1 ELSE 0 END as adverse_weather_risk,
    CASE WHEN road_surface_conditions IN (2, 3, 4) THEN 1 ELSE 0 END as poor_surface_risk,
    CASE WHEN speed_limit >= 60 THEN 1 ELSE 0 END as high_speed_risk,
    CASE WHEN is_weekend THEN 1 ELSE 0 END as weekend_risk,

    -- Composite risk score (0-5)
    (CASE WHEN light_conditions IN (4, 5, 6) THEN 1 ELSE 0 END +
     CASE WHEN weather_conditions IN (2, 3, 5, 6, 7) THEN 1 ELSE 0 END +
     CASE WHEN road_surface_conditions IN (2, 3, 4) THEN 1 ELSE 0 END +
     CASE WHEN speed_limit >= 60 THEN 1 ELSE 0 END +
     CASE WHEN is_weekend THEN 1 ELSE 0 END) as total_risk_score,

    -- Risk level category
    CASE
        WHEN (CASE WHEN light_conditions IN (4, 5, 6) THEN 1 ELSE 0 END +
              CASE WHEN weather_conditions IN (2, 3, 5, 6, 7) THEN 1 ELSE 0 END +
              CASE WHEN road_surface_conditions IN (2, 3, 4) THEN 1 ELSE 0 END +
              CASE WHEN speed_limit >= 60 THEN 1 ELSE 0 END +
              CASE WHEN is_weekend THEN 1 ELSE 0 END) >= 3 THEN 'High Risk'
        WHEN (CASE WHEN light_conditions IN (4, 5, 6) THEN 1 ELSE 0 END +
              CASE WHEN weather_conditions IN (2, 3, 5, 6, 7) THEN 1 ELSE 0 END +
              CASE WHEN road_surface_conditions IN (2, 3, 4) THEN 1 ELSE 0 END +
              CASE WHEN speed_limit >= 60 THEN 1 ELSE 0 END +
              CASE WHEN is_weekend THEN 1 ELSE 0 END) = 2 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as risk_level

FROM collision_with_features;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON VIEW collision_with_features IS 'Collision data with engineered temporal and categorical features';
COMMENT ON VIEW full_accident_data IS 'Merged collision, vehicle, and casualty data for analysis';
COMMENT ON VIEW hourly_accident_stats IS 'Accident statistics aggregated by hour of day';
COMMENT ON VIEW daily_accident_stats IS 'Accident statistics aggregated by day of week';
COMMENT ON VIEW collision_risk_scores IS 'Risk scoring based on environmental and temporal factors';
