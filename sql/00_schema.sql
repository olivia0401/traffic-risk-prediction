-- ============================================================================
-- DATABASE SCHEMA FOR UK TRAFFIC COLLISION DATA
-- ============================================================================
-- Purpose: Create tables for storing UK DfT collision, vehicle, and casualty data
-- Database: PostgreSQL 12+
-- Author: Data Engineering Team
-- ============================================================================

-- Drop existing tables if they exist
DROP TABLE IF EXISTS casualty CASCADE;
DROP TABLE IF EXISTS vehicle CASCADE;
DROP TABLE IF EXISTS collision CASCADE;

-- ============================================================================
-- COLLISION TABLE
-- ============================================================================
-- Stores one record per traffic accident
CREATE TABLE collision (
    collision_index VARCHAR(50) PRIMARY KEY,
    collision_reference VARCHAR(50),
    date_of_accident DATE NOT NULL,
    time_of_accident TIME NOT NULL,

    -- Location
    longitude NUMERIC(10, 6),
    latitude NUMERIC(10, 6),
    local_authority_district VARCHAR(100),
    first_road_class INTEGER,
    first_road_number INTEGER,
    road_type INTEGER,

    -- Environment
    weather_conditions INTEGER,
    light_conditions INTEGER,
    road_surface_conditions INTEGER,
    urban_or_rural_area INTEGER,

    -- Junction details
    junction_detail INTEGER,
    junction_control INTEGER,

    -- Collision details
    collision_severity INTEGER,
    number_of_vehicles INTEGER,
    number_of_casualties INTEGER,
    speed_limit INTEGER,

    -- Metadata
    police_force INTEGER,
    did_police_officer_attend BOOLEAN,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- VEHICLE TABLE
-- ============================================================================
-- Stores one record per vehicle involved in an accident
CREATE TABLE vehicle (
    vehicle_id SERIAL PRIMARY KEY,
    collision_index VARCHAR(50) NOT NULL REFERENCES collision(collision_index) ON DELETE CASCADE,
    vehicle_reference VARCHAR(50) NOT NULL,

    -- Vehicle details
    vehicle_type INTEGER,
    vehicle_manoeuvre INTEGER,
    vehicle_location INTEGER,

    -- Driver details
    sex_of_driver INTEGER,
    age_of_driver INTEGER,
    age_band_of_driver VARCHAR(20),
    driver_imd_decile INTEGER,
    driver_home_area_type INTEGER,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(collision_index, vehicle_reference)
);

-- ============================================================================
-- CASUALTY TABLE
-- ============================================================================
-- Stores one record per casualty (injured/killed person) in an accident
CREATE TABLE casualty (
    casualty_id SERIAL PRIMARY KEY,
    collision_index VARCHAR(50) NOT NULL REFERENCES collision(collision_index) ON DELETE CASCADE,
    vehicle_reference VARCHAR(50) NOT NULL,
    casualty_reference VARCHAR(50) NOT NULL,

    -- Casualty details
    casualty_class INTEGER,
    casualty_severity INTEGER,
    sex_of_casualty INTEGER,
    age_of_casualty INTEGER,
    age_band_of_casualty VARCHAR(20),

    -- Additional info
    casualty_type INTEGER,
    pedestrian_location INTEGER,
    pedestrian_movement INTEGER,
    casualty_imd_decile INTEGER,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(collision_index, vehicle_reference, casualty_reference)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Collision indexes
CREATE INDEX idx_collision_date ON collision(date_of_accident);
CREATE INDEX idx_collision_time ON collision(time_of_accident);
CREATE INDEX idx_collision_severity ON collision(collision_severity);
CREATE INDEX idx_collision_location ON collision(longitude, latitude);

-- Vehicle indexes
CREATE INDEX idx_vehicle_collision ON vehicle(collision_index);
CREATE INDEX idx_vehicle_type ON vehicle(vehicle_type);
CREATE INDEX idx_vehicle_driver_age ON vehicle(age_of_driver);

-- Casualty indexes
CREATE INDEX idx_casualty_collision ON casualty(collision_index);
CREATE INDEX idx_casualty_severity ON casualty(casualty_severity);
CREATE INDEX idx_casualty_age ON casualty(age_of_casualty);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE collision IS 'UK Department for Transport traffic collision records';
COMMENT ON TABLE vehicle IS 'Vehicles involved in UK traffic collisions';
COMMENT ON TABLE casualty IS 'Casualties (injuries/deaths) from UK traffic collisions';

COMMENT ON COLUMN collision.collision_severity IS '1=Fatal, 2=Serious, 3=Slight';
COMMENT ON COLUMN casualty.casualty_severity IS '1=Fatal, 2=Serious, 3=Slight';
