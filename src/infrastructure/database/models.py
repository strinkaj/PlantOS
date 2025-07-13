"""
SQLAlchemy models for PlantOS database tables.

These models map domain entities to database tables with proper
relationships, indexes, and constraints for optimal performance.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4
from decimal import Decimal

from sqlalchemy import (
    Column, String, Integer, Numeric, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.infrastructure.database.config import Base


class TimestampedMixin:
    """Mixin for models with created_at and updated_at timestamps."""
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, server_default=func.now())


class PlantSpeciesModel(Base, TimestampedMixin):
    """Plant species database model."""
    __tablename__ = 'plant_species'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    scientific_name = Column(String(255), nullable=False, unique=True, index=True)
    
    # Optimal ranges
    optimal_moisture_min = Column(Numeric(5, 2), nullable=False)
    optimal_moisture_max = Column(Numeric(5, 2), nullable=False)
    optimal_temp_min = Column(Numeric(4, 1), nullable=False)
    optimal_temp_max = Column(Numeric(4, 1), nullable=False)
    optimal_humidity = Column(Numeric(5, 2), nullable=False)
    
    # Care settings
    water_frequency_hours = Column(Integer, nullable=False)
    light_requirements = Column(String(20), nullable=False)
    
    # Additional data
    care_instructions = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    plants = relationship("PlantModel", back_populates="species")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('optimal_moisture_min >= 0 AND optimal_moisture_min <= 100', name='check_moisture_min_range'),
        CheckConstraint('optimal_moisture_max >= 0 AND optimal_moisture_max <= 100', name='check_moisture_max_range'),
        CheckConstraint('optimal_moisture_min < optimal_moisture_max', name='check_moisture_range_order'),
        CheckConstraint('optimal_temp_min < optimal_temp_max', name='check_temp_range_order'),
        CheckConstraint('optimal_humidity >= 0 AND optimal_humidity <= 100', name='check_humidity_range'),
        CheckConstraint('water_frequency_hours > 0', name='check_water_frequency_positive'),
        CheckConstraint("light_requirements IN ('low', 'medium', 'bright', 'direct')", name='check_light_requirements'),
    )


class PlantModel(Base, TimestampedMixin):
    """Plant database model."""
    __tablename__ = 'plants'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    species_id = Column(UUID(as_uuid=True), ForeignKey('plant_species.id'), nullable=True, index=True)
    location = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default='active', index=True)
    health_score = Column(Numeric(3, 2), nullable=True)
    last_watered_at = Column(DateTime, nullable=True)
    last_fertilized_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=False, default='')
    metadata = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    species = relationship("PlantSpeciesModel", back_populates="plants")
    sensors = relationship("SensorModel", back_populates="plant")
    actuators = relationship("ActuatorModel", back_populates="plant")
    sensor_readings = relationship("SensorReadingModel", back_populates="plant")
    care_events = relationship("CareEventModel", back_populates="plant")
    health_assessments = relationship("PlantHealthModel", back_populates="plant")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('active', 'dormant', 'sick', 'deceased', 'maintenance')", name='check_plant_status'),
        CheckConstraint('health_score IS NULL OR (health_score >= 0 AND health_score <= 1)', name='check_health_score_range'),
        Index('idx_plants_status_created', 'status', 'created_at'),
        Index('idx_plants_species_status', 'species_id', 'status'),
    )


class SensorModel(Base, TimestampedMixin):
    """Sensor database model."""
    __tablename__ = 'sensors'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    plant_id = Column(UUID(as_uuid=True), ForeignKey('plants.id'), nullable=True, index=True)
    sensor_type = Column(String(50), nullable=False, index=True)
    model = Column(String(100), nullable=False)
    location = Column(String(255), nullable=False)
    gpio_pin = Column(Integer, nullable=True)
    i2c_address = Column(Integer, nullable=True)
    calibration_offset = Column(Numeric(10, 6), nullable=False, default=0.0)
    calibration_multiplier = Column(Numeric(10, 6), nullable=False, default=1.0)
    status = Column(String(20), nullable=False, default='active', index=True)
    last_reading_at = Column(DateTime, nullable=True)
    metadata = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    plant = relationship("PlantModel", back_populates="sensors")
    readings = relationship("SensorReadingModel", back_populates="sensor")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('gpio_pin IS NULL OR (gpio_pin >= 0 AND gpio_pin <= 40)', name='check_gpio_pin_range'),
        CheckConstraint('i2c_address IS NULL OR (i2c_address >= 0 AND i2c_address <= 127)', name='check_i2c_address_range'),
        CheckConstraint('calibration_multiplier != 0', name='check_calibration_multiplier_nonzero'),
        CheckConstraint("status IN ('active', 'inactive', 'maintenance', 'error')", name='check_sensor_status'),
        UniqueConstraint('gpio_pin', name='uq_sensor_gpio_pin'),
        UniqueConstraint('i2c_address', name='uq_sensor_i2c_address'),
        Index('idx_sensors_type_status', 'sensor_type', 'status'),
        Index('idx_sensors_plant_type', 'plant_id', 'sensor_type'),
    )


class ActuatorModel(Base, TimestampedMixin):
    """Actuator database model."""
    __tablename__ = 'actuators'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    plant_id = Column(UUID(as_uuid=True), ForeignKey('plants.id'), nullable=True, index=True)
    actuator_type = Column(String(50), nullable=False, index=True)
    model = Column(String(100), nullable=False)
    location = Column(String(255), nullable=False)
    gpio_pin = Column(Integer, nullable=True)
    max_runtime_seconds = Column(Integer, nullable=False, default=300)
    status = Column(String(20), nullable=False, default='idle', index=True)
    last_operation_at = Column(DateTime, nullable=True)
    total_runtime_seconds = Column(Integer, nullable=False, default=0)
    operation_count = Column(Integer, nullable=False, default=0)
    metadata = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    plant = relationship("PlantModel", back_populates="actuators")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('gpio_pin IS NULL OR (gpio_pin >= 0 AND gpio_pin <= 40)', name='check_actuator_gpio_pin_range'),
        CheckConstraint('max_runtime_seconds > 0', name='check_max_runtime_positive'),
        CheckConstraint('total_runtime_seconds >= 0', name='check_total_runtime_nonnegative'),
        CheckConstraint('operation_count >= 0', name='check_operation_count_nonnegative'),
        CheckConstraint("status IN ('idle', 'active', 'maintenance', 'error')", name='check_actuator_status'),
        CheckConstraint("actuator_type IN ('pump', 'valve', 'fan', 'light', 'heater')", name='check_actuator_type'),
        UniqueConstraint('gpio_pin', name='uq_actuator_gpio_pin'),
        Index('idx_actuators_type_status', 'actuator_type', 'status'),
        Index('idx_actuators_plant_type', 'plant_id', 'actuator_type'),
    )


class SensorReadingModel(Base):
    """Sensor reading database model (TimescaleDB hypertable)."""
    __tablename__ = 'sensor_readings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    time = Column(DateTime, nullable=False, index=True)  # Partition key for TimescaleDB
    sensor_id = Column(UUID(as_uuid=True), ForeignKey('sensors.id'), nullable=False, index=True)
    plant_id = Column(UUID(as_uuid=True), ForeignKey('plants.id'), nullable=True, index=True)
    sensor_type = Column(String(50), nullable=False, index=True)
    value = Column(Numeric(10, 4), nullable=False)
    unit = Column(String(20), nullable=False)
    quality_score = Column(Numeric(3, 2), nullable=False, default=1.0)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    sensor = relationship("SensorModel", back_populates="readings")
    plant = relationship("PlantModel", back_populates="sensor_readings")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='check_quality_score_range'),
        Index('idx_sensor_readings_sensor_time', 'sensor_id', 'time'),
        Index('idx_sensor_readings_plant_time', 'plant_id', 'time'),
        Index('idx_sensor_readings_type_time', 'sensor_type', 'time'),
    )


class CareEventModel(Base):
    """Care event database model."""
    __tablename__ = 'care_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    plant_id = Column(UUID(as_uuid=True), ForeignKey('plants.id'), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    triggered_by = Column(String(20), nullable=False, index=True)
    amount = Column(Numeric(10, 2), nullable=True)
    unit = Column(String(20), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    notes = Column(Text, nullable=False, default='')
    metadata = Column(JSON, nullable=False, default=dict)
    performed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    performed_by = Column(String(255), nullable=True)
    
    # Relationships
    plant = relationship("PlantModel", back_populates="care_events")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('amount IS NULL OR amount >= 0', name='check_amount_nonnegative'),
        CheckConstraint('duration_seconds IS NULL OR duration_seconds > 0', name='check_duration_positive'),
        CheckConstraint("event_type IN ('watering', 'fertilizing', 'pruning', 'repotting', 'treatment')", name='check_event_type'),
        CheckConstraint("triggered_by IN ('automated', 'manual', 'scheduled', 'emergency')", name='check_triggered_by'),
        Index('idx_care_events_plant_performed', 'plant_id', 'performed_at'),
        Index('idx_care_events_type_performed', 'event_type', 'performed_at'),
    )


class PlantHealthModel(Base):
    """Plant health assessment database model."""
    __tablename__ = 'plant_health_assessments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    plant_id = Column(UUID(as_uuid=True), ForeignKey('plants.id'), nullable=False, index=True)
    overall_score = Column(Numeric(3, 2), nullable=False)
    moisture_score = Column(Numeric(3, 2), nullable=False)
    growth_score = Column(Numeric(3, 2), nullable=False)
    care_consistency_score = Column(Numeric(3, 2), nullable=False)
    factors = Column(JSON, nullable=False, default=list)
    recommendations = Column(JSON, nullable=False, default=list)
    assessed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    assessed_by = Column(String(255), nullable=False, default='system')
    
    # Relationships
    plant = relationship("PlantModel", back_populates="health_assessments")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('overall_score >= 0 AND overall_score <= 1', name='check_overall_score_range'),
        CheckConstraint('moisture_score >= 0 AND moisture_score <= 1', name='check_moisture_score_range'),
        CheckConstraint('growth_score >= 0 AND growth_score <= 1', name='check_growth_score_range'),
        CheckConstraint('care_consistency_score >= 0 AND care_consistency_score <= 1', name='check_care_score_range'),
        Index('idx_health_assessments_plant_assessed', 'plant_id', 'assessed_at'),
        Index('idx_health_assessments_score_assessed', 'overall_score', 'assessed_at'),
    )


class SystemEventModel(Base):
    """System event logging model."""
    __tablename__ = 'system_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    event_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, default='info', index=True)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    occurred_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("severity IN ('debug', 'info', 'warning', 'error', 'critical')", name='check_event_severity'),
        Index('idx_system_events_type_occurred', 'event_type', 'occurred_at'),
        Index('idx_system_events_severity_occurred', 'severity', 'occurred_at'),
    )


# TimescaleDB specific functions and triggers
CREATE_HYPERTABLE_SQL = """
-- Create hypertable for sensor_readings if it doesn't exist
SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE);

-- Add retention policy to keep data for 1 year
SELECT add_retention_policy('sensor_readings', INTERVAL '1 year', if_not_exists => TRUE);

-- Create continuous aggregates for hourly averages
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    sensor_id,
    plant_id,
    sensor_type,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    COUNT(*) as reading_count
FROM sensor_readings
GROUP BY bucket, sensor_id, plant_id, sensor_type;

-- Add policy to refresh the continuous aggregate
SELECT add_continuous_aggregate_policy('sensor_readings_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);
"""

# Function to create TimescaleDB specific features
async def setup_timescaledb_features(db_manager):
    """Setup TimescaleDB hypertables and continuous aggregates."""
    try:
        await db_manager.execute_raw_sql(CREATE_HYPERTABLE_SQL)
        print("TimescaleDB features setup completed")
    except Exception as e:
        print(f"TimescaleDB setup warning: {e}")
        # Don't fail if TimescaleDB extension is not available