-- Initialize TimescaleDB extension and create hypertables
-- This script runs automatically when the PostgreSQL container starts

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schema for PlantOS
CREATE SCHEMA IF NOT EXISTS plantos;

-- Set default schema
SET search_path TO plantos, public;

-- Create sensor_readings table for time-series data
CREATE TABLE IF NOT EXISTS sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    sensor_id UUID NOT NULL,
    plant_id UUID,
    sensor_type VARCHAR(50) NOT NULL,
    value NUMERIC(10,4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    quality_score NUMERIC(3,2) DEFAULT 1.0,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_id_time 
    ON sensor_readings (sensor_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_plant_id_time 
    ON sensor_readings (plant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_type_time 
    ON sensor_readings (sensor_type, time DESC);

-- Create data retention policy (keep 1 year of data)
SELECT add_retention_policy('sensor_readings', INTERVAL '1 year', if_not_exists => TRUE);

-- Create plants table
CREATE TABLE IF NOT EXISTS plants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    species_id UUID,
    location VARCHAR(255),
    optimal_moisture_min NUMERIC(5,2) DEFAULT 30.0,
    optimal_moisture_max NUMERIC(5,2) DEFAULT 70.0,
    water_frequency_hours INTEGER DEFAULT 24,
    last_watered_at TIMESTAMPTZ,
    health_score NUMERIC(3,2),
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create plant_species table
CREATE TABLE IF NOT EXISTS plant_species (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    scientific_name VARCHAR(255),
    care_instructions JSONB,
    optimal_conditions JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create watering_events table
CREATE TABLE IF NOT EXISTS watering_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plant_id UUID NOT NULL REFERENCES plants(id),
    amount_ml INTEGER NOT NULL,
    trigger_type VARCHAR(50) NOT NULL, -- 'automated', 'manual', 'scheduled'
    moisture_before NUMERIC(5,2),
    moisture_after NUMERIC(5,2),
    watered_at TIMESTAMPTZ DEFAULT NOW(),
    duration_seconds INTEGER,
    metadata JSONB
);

-- Create sensors table
CREATE TABLE IF NOT EXISTS sensors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plant_id UUID REFERENCES plants(id),
    sensor_type VARCHAR(50) NOT NULL,
    gpio_pin INTEGER,
    calibration_data JSONB,
    status VARCHAR(50) DEFAULT 'active',
    last_reading_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create system_events table for logging
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) DEFAULT 'info',
    message TEXT NOT NULL,
    details JSONB,
    occurred_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create some sample data for development
INSERT INTO plant_species (name, scientific_name, care_instructions, optimal_conditions) 
VALUES 
    ('Spider Plant', 'Chlorophytum comosum', 
     '{"watering": "weekly", "light": "indirect", "humidity": "moderate"}',
     '{"moisture_min": 30, "moisture_max": 60, "temp_min": 18, "temp_max": 24}'),
    ('Pothos', 'Epipremnum aureum',
     '{"watering": "when dry", "light": "low to bright", "humidity": "moderate"}',
     '{"moisture_min": 25, "moisture_max": 55, "temp_min": 16, "temp_max": 26}')
ON CONFLICT (name) DO NOTHING;

-- Grant permissions to plantos_user
GRANT USAGE ON SCHEMA plantos TO plantos_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA plantos TO plantos_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA plantos TO plantos_user;

-- Show successful initialization
SELECT 'TimescaleDB initialization completed successfully' AS status;