-- ============================================================================
-- RISK PREDICTION & SCORING QUERIES
-- ============================================================================
-- Purpose: Predict accident risk for specific time windows
-- Used for autonomous driving route planning and safety alerts
-- ============================================================================

-- ============================================================================
-- 1. REAL-TIME RISK SCORING FOR CURRENT CONDITIONS
-- ============================================================================
-- Example: Predict risk for "Friday, 5:00 PM, rainy, urban area"

CREATE OR REPLACE FUNCTION get_risk_score(
    p_hour INT,
    p_day_of_week INT,
    p_weather INT,
    p_light INT,
    p_road_surface INT,
    p_urban_rural INT
)
RETURNS TABLE (
    predicted_risk_level VARCHAR,
    expected_accidents NUMERIC,
    historical_fatal_rate NUMERIC,
    recommendation TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE
            WHEN COUNT(*) > 300 THEN 'CRITICAL'
            WHEN COUNT(*) > 200 THEN 'HIGH'
            WHEN COUNT(*) > 100 THEN 'MEDIUM'
            ELSE 'LOW'
        END::VARCHAR as predicted_risk_level,

        ROUND(COUNT(*)::NUMERIC /
              NULLIF(COUNT(DISTINCT date_of_accident), 0), 2) as expected_accidents,

        ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) /
              NULLIF(COUNT(*), 0), 3) as historical_fatal_rate,

        CASE
            WHEN COUNT(*) > 300 THEN 'Avoid this time window if possible. Consider alternative routes.'
            WHEN COUNT(*) > 200 THEN 'High accident risk. Drive with extreme caution.'
            WHEN COUNT(*) > 100 THEN 'Moderate risk. Stay alert to road conditions.'
            ELSE 'Low risk period for travel.'
        END::TEXT as recommendation

    FROM collision_with_features
    WHERE hour_of_day = p_hour
      AND day_of_week = p_day_of_week
      AND (weather_conditions = p_weather OR p_weather IS NULL)
      AND (light_conditions = p_light OR p_light IS NULL)
      AND (road_surface_conditions = p_road_surface OR p_road_surface IS NULL)
      AND (urban_or_rural_area = p_urban_rural OR p_urban_rural IS NULL);
END;
$$ LANGUAGE plpgsql;

-- Example usage:
-- SELECT * FROM get_risk_score(17, 5, 2, 4, 2, 1);
-- (5:00 PM, Friday, Rain, Dark-lit, Wet, Urban)

-- ============================================================================
-- 2. HOURLY RISK HEATMAP DATA
-- ============================================================================
-- Used for visualizing risk by hour and day of week

SELECT
    day_of_week,
    day_name,
    hour_of_day,
    COUNT(*) as accident_count,
    ROUND(AVG(number_of_casualties), 2) as avg_casualties,

    -- Risk score (normalized 0-100)
    ROUND(
        100.0 * (COUNT(*) - MIN(COUNT(*)) OVER ()) /
        NULLIF((MAX(COUNT(*)) OVER () - MIN(COUNT(*)) OVER ()), 0),
        1
    ) as normalized_risk_score,

    -- Severity-weighted risk
    ROUND(
        (SUM(CASE WHEN collision_severity = 1 THEN 10
                  WHEN collision_severity = 2 THEN 5
                  WHEN collision_severity = 3 THEN 1
                  ELSE 0 END)::NUMERIC / NULLIF(COUNT(*), 0)),
        2
    ) as severity_weighted_score

FROM collision_with_features
GROUP BY day_of_week, day_name, hour_of_day
ORDER BY day_of_week, hour_of_day;

-- ============================================================================
-- 3. ENVIRONMENTAL CONDITION RISK MATRIX
-- ============================================================================

SELECT
    weather_label,
    light_label,
    road_surface_label,
    COUNT(*) as accident_count,
    ROUND(AVG(number_of_casualties), 2) as avg_casualties_per_accident,
    ROUND(100.0 * SUM(CASE WHEN collision_severity IN (1, 2) THEN 1 ELSE 0 END) /
          NULLIF(COUNT(*), 0), 2) as serious_fatal_rate_pct,

    -- Risk classification
    CASE
        WHEN COUNT(*) > 1000 AND
             100.0 * SUM(CASE WHEN collision_severity IN (1, 2) THEN 1 ELSE 0 END) /
             NULLIF(COUNT(*), 0) > 25 THEN 'CRITICAL'
        WHEN COUNT(*) > 500 THEN 'HIGH'
        WHEN COUNT(*) > 100 THEN 'MEDIUM'
        ELSE 'LOW'
    END as risk_level

FROM collision_with_features
WHERE weather_label IS NOT NULL
  AND light_label IS NOT NULL
  AND road_surface_label IS NOT NULL
GROUP BY weather_label, light_label, road_surface_label
HAVING COUNT(*) >= 10  -- Minimum sample size for significance
ORDER BY accident_count DESC;

-- ============================================================================
-- 4. VEHICLE-TYPE SPECIFIC RISK PROFILING
-- ============================================================================

SELECT
    v.vehicle_type_label,
    COUNT(*) as total_accidents,
    ROUND(AVG(c.number_of_casualties), 2) as avg_casualties,

    -- Severity distribution
    ROUND(100.0 * SUM(CASE WHEN cas.casualty_severity = 1 THEN 1 ELSE 0 END) /
          NULLIF(COUNT(*), 0), 2) as fatal_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN cas.casualty_severity = 2 THEN 1 ELSE 0 END) /
          NULLIF(COUNT(*), 0), 2) as serious_rate_pct,

    -- High-risk conditions for this vehicle type
    MODE() WITHIN GROUP (ORDER BY c.weather_label) as most_common_weather,
    MODE() WITHIN GROUP (ORDER BY c.light_label) as most_common_light,

    -- Risk score (higher = more dangerous)
    ROUND(
        (100.0 * SUM(CASE WHEN cas.casualty_severity = 1 THEN 1 ELSE 0 END) /
         NULLIF(COUNT(*), 0)) *
        (AVG(c.number_of_casualties)),
        2
    ) as composite_risk_score

FROM full_accident_data c
JOIN vehicle v ON c.collision_index = v.collision_index
LEFT JOIN casualty cas ON c.collision_index = cas.collision_index
    AND v.vehicle_reference = cas.vehicle_reference
WHERE v.vehicle_type_label IS NOT NULL
GROUP BY v.vehicle_type_label
HAVING COUNT(*) >= 100
ORDER BY composite_risk_score DESC;

-- ============================================================================
-- 5. PREDICTIVE FEATURES FOR ML MODEL EXPORT
-- ============================================================================
-- Export clean dataset for Python scikit-learn models

CREATE OR REPLACE VIEW ml_training_dataset AS
SELECT
    -- Target variable
    collision_severity as target,

    -- Temporal features
    hour_of_day,
    day_of_week,
    month,
    CASE WHEN is_weekend THEN 1 ELSE 0 END as is_weekend,
    time_slot_15min,

    -- Environmental features
    COALESCE(weather_conditions, 1) as weather_conditions,
    COALESCE(light_conditions, 1) as light_conditions,
    COALESCE(road_surface_conditions, 1) as road_surface_conditions,

    -- Road context
    COALESCE(road_type, 6) as road_type,
    COALESCE(speed_limit, 30) as speed_limit,
    COALESCE(urban_or_rural_area, 1) as urban_or_rural_area,
    COALESCE(junction_detail, 0) as junction_detail,

    -- Accident characteristics
    number_of_vehicles,
    number_of_casualties,

    -- Risk flags
    CASE WHEN light_conditions IN (4, 5, 6) THEN 1 ELSE 0 END as is_dark,
    CASE WHEN weather_conditions IN (2, 3, 5, 6, 7) THEN 1 ELSE 0 END as is_adverse_weather,
    CASE WHEN road_surface_conditions IN (2, 3, 4) THEN 1 ELSE 0 END as is_poor_surface,
    CASE WHEN speed_limit >= 60 THEN 1 ELSE 0 END as is_high_speed,

    -- Unique identifier
    collision_index

FROM collision_with_features
WHERE collision_severity IS NOT NULL
  AND hour_of_day IS NOT NULL
  AND day_of_week IS NOT NULL;

-- ============================================================================
-- 6. TOP 20 HIGHEST RISK SCENARIOS
-- ============================================================================

SELECT
    hour_of_day,
    day_name,
    weather_label,
    light_label,
    road_surface_label,
    area_type_label,
    COUNT(*) as historical_accidents,
    ROUND(AVG(number_of_casualties), 2) as avg_casualties,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) as fatal_count,
    SUM(CASE WHEN collision_severity = 2 THEN 1 ELSE 0 END) as serious_count,

    -- Composite risk score (weighted by severity)
    ROUND(
        (SUM(CASE WHEN collision_severity = 1 THEN 10
                  WHEN collision_severity = 2 THEN 5
                  WHEN collision_severity = 3 THEN 1 END)::NUMERIC /
         NULLIF(COUNT(*), 0)),
        2
    ) as risk_score

FROM collision_with_features
WHERE weather_label IS NOT NULL
  AND light_label IS NOT NULL
GROUP BY hour_of_day, day_name, day_of_week,
         weather_label, light_label, road_surface_label, area_type_label
HAVING COUNT(*) >= 20  -- Minimum observations for statistical validity
ORDER BY risk_score DESC, historical_accidents DESC
LIMIT 20;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON FUNCTION get_risk_score IS 'Returns real-time risk prediction for given conditions';
COMMENT ON VIEW ml_training_dataset IS 'Clean feature set for machine learning model training';
