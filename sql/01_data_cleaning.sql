-- ============================================================================
-- DATA CLEANING QUERIES
-- ============================================================================
-- Purpose: Clean and standardize data after initial CSV import
-- Replicates Python preprocessing steps in SQL
-- ============================================================================

-- ============================================================================
-- 1. REPLACE CODED MISSING VALUES WITH NULL
-- ============================================================================
-- UK DfT data uses -1, 99, and 'Unknown' to represent missing values
-- Convert these to NULL for proper analysis

-- Collision table
UPDATE collision
SET
    longitude = CASE WHEN longitude = -1 THEN NULL ELSE longitude END,
    latitude = CASE WHEN latitude = -1 THEN NULL ELSE latitude END,
    weather_conditions = CASE WHEN weather_conditions IN (-1, 99) THEN NULL ELSE weather_conditions END,
    light_conditions = CASE WHEN light_conditions IN (-1, 99) THEN NULL ELSE light_conditions END,
    road_surface_conditions = CASE WHEN road_surface_conditions IN (-1, 99) THEN NULL ELSE road_surface_conditions END,
    junction_detail = CASE WHEN junction_detail = 99 THEN NULL ELSE junction_detail END,
    junction_control = CASE WHEN junction_control = 9 THEN NULL ELSE junction_control END,
    road_type = CASE WHEN road_type = 9 THEN NULL ELSE road_type END,
    speed_limit = CASE WHEN speed_limit = -1 THEN NULL ELSE speed_limit END;

-- Vehicle table
UPDATE vehicle
SET
    vehicle_type = CASE WHEN vehicle_type = 99 THEN NULL ELSE vehicle_type END,
    vehicle_manoeuvre = CASE WHEN vehicle_manoeuvre = 99 THEN NULL ELSE vehicle_manoeuvre END,
    age_of_driver = CASE WHEN age_of_driver = -1 THEN NULL ELSE age_of_driver END,
    sex_of_driver = CASE WHEN sex_of_driver = -1 THEN NULL ELSE sex_of_driver END,
    driver_imd_decile = CASE WHEN driver_imd_decile = -1 THEN NULL ELSE driver_imd_decile END;

-- Casualty table
UPDATE casualty
SET
    age_of_casualty = CASE WHEN age_of_casualty = -1 THEN NULL ELSE age_of_casualty END,
    sex_of_casualty = CASE WHEN sex_of_casualty = -1 THEN NULL ELSE sex_of_casualty END,
    casualty_imd_decile = CASE WHEN casualty_imd_decile = -1 THEN NULL ELSE casualty_imd_decile END,
    pedestrian_location = CASE WHEN pedestrian_location = -1 THEN NULL ELSE pedestrian_location END;

-- ============================================================================
-- 2. REMOVE DUPLICATE RECORDS
-- ============================================================================

-- Remove duplicate collision records (keep earliest)
DELETE FROM collision
WHERE collision_index IN (
    SELECT collision_index
    FROM (
        SELECT
            collision_index,
            ROW_NUMBER() OVER (
                PARTITION BY collision_index
                ORDER BY created_at
            ) as rn
        FROM collision
    ) t
    WHERE rn > 1
);

-- Remove duplicate vehicle records
DELETE FROM vehicle
WHERE vehicle_id IN (
    SELECT vehicle_id
    FROM (
        SELECT
            vehicle_id,
            ROW_NUMBER() OVER (
                PARTITION BY collision_index, vehicle_reference
                ORDER BY created_at
            ) as rn
        FROM vehicle
    ) t
    WHERE rn > 1
);

-- Remove duplicate casualty records
DELETE FROM casualty
WHERE casualty_id IN (
    SELECT casualty_id
    FROM (
        SELECT
            casualty_id,
            ROW_NUMBER() OVER (
                PARTITION BY collision_index, vehicle_reference, casualty_reference
                ORDER BY created_at
            ) as rn
        FROM casualty
    ) t
    WHERE rn > 1
);

-- ============================================================================
-- 3. DATA VALIDATION
-- ============================================================================

-- Check for invalid dates (future dates)
SELECT
    COUNT(*) as future_dates_count,
    MAX(date_of_accident) as latest_date
FROM collision
WHERE date_of_accident > CURRENT_DATE;

-- Check for invalid time values
SELECT
    COUNT(*) as invalid_time_count
FROM collision
WHERE time_of_accident < '00:00:00' OR time_of_accident > '23:59:59';

-- Check severity codes are valid (1, 2, or 3)
SELECT
    collision_severity,
    COUNT(*) as count
FROM collision
WHERE collision_severity NOT IN (1, 2, 3)
GROUP BY collision_severity;

-- ============================================================================
-- 4. DATA QUALITY REPORT
-- ============================================================================

-- Missing value summary for key collision fields
SELECT
    'collision' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) - COUNT(longitude) as longitude_nulls,
    COUNT(*) - COUNT(latitude) as latitude_nulls,
    COUNT(*) - COUNT(weather_conditions) as weather_nulls,
    COUNT(*) - COUNT(light_conditions) as light_nulls,
    COUNT(*) - COUNT(road_surface_conditions) as road_surface_nulls,
    ROUND(100.0 * (COUNT(*) - COUNT(longitude)) / COUNT(*), 2) as longitude_null_pct
FROM collision;

-- Missing value summary for vehicle fields
SELECT
    'vehicle' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) - COUNT(age_of_driver) as age_nulls,
    COUNT(*) - COUNT(vehicle_type) as vehicle_type_nulls,
    ROUND(100.0 * (COUNT(*) - COUNT(age_of_driver)) / COUNT(*), 2) as age_null_pct
FROM vehicle;

-- Missing value summary for casualty fields
SELECT
    'casualty' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) - COUNT(age_of_casualty) as age_nulls,
    COUNT(*) - COUNT(casualty_severity) as severity_nulls,
    ROUND(100.0 * (COUNT(*) - COUNT(age_of_casualty)) / COUNT(*), 2) as age_null_pct
FROM casualty;
