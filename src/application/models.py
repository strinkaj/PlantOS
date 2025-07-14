"""
Pydantic models for PlantOS API requests and responses.

These models provide automatic validation, serialization, and API documentation.
They serve as the boundary between external API and internal domain models.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, PositiveFloat, constr, confloat

from src.core.domain.value_objects import PlantID, SensorID, SpeciesID
from src.core.domain.entities import Plant, PlantSpecies, Sensor, SensorReading


# Base models for common patterns
class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IdentifiedModel(BaseModel):
    """Base model with UUID identifier."""
    id: UUID
    
    class Config:
        json_encoders = {
            UUID: str
        }


# Configuration models
class APIConfig(BaseModel):
    """Configuration for FastAPI application."""
    title: str = "PlantOS API"
    version: str = "1.0.0"
    description: str = "Production-grade plant care automation system"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]


# Status and Health models
class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyStatus(BaseModel):
    """Status of a system dependency."""
    name: str
    status: HealthStatus
    response_time_ms: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: HealthStatus
    timestamp: datetime
    version: str
    dependencies: List[DependencyStatus]
    metrics: Dict[str, Any] = {}


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


# Location models
class LocationRequest(BaseModel):
    """Location data for plant placement."""
    room: constr(min_length=1, max_length=100, strip_whitespace=True)
    position: Optional[constr(max_length=200)] = None
    light_level: Optional[confloat(ge=0, le=10)] = None

    @validator('light_level')
    def validate_light_level(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('Light level must be between 0 and 10')
        return v


# Plant models
class PlantCreateRequest(BaseModel):
    """Request model for creating a new plant."""
    name: constr(min_length=2, max_length=100, strip_whitespace=True)
    species_id: UUID
    location: Optional[LocationRequest] = None
    notes: Optional[constr(max_length=1000)] = None

    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Plant name cannot be empty')
        return v.strip()


class PlantUpdateRequest(BaseModel):
    """Request model for updating a plant."""
    name: Optional[constr(min_length=2, max_length=100, strip_whitespace=True)] = None
    location: Optional[LocationRequest] = None
    notes: Optional[constr(max_length=1000)] = None


class PlantResponse(BaseModel, TimestampedModel, IdentifiedModel):
    """Response model for plant data."""
    name: str
    species_id: UUID
    location: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    health_score: Optional[float] = None
    status: str = "active"

    @classmethod
    def from_domain(cls, plant: Plant) -> "PlantResponse":
        """Create response model from domain entity."""
        return cls(
            id=plant.id.value,
            name=plant.name,
            species_id=plant.species_id.value,
            location={
                "room": plant.location.room,
                "position": plant.location.position,
                "light_level": plant.location.light_level
            } if plant.location else None,
            notes=plant.notes,
            health_score=plant.health_score.value if plant.health_score else None,
            status=plant.status,
            created_at=plant.created_at,
            updated_at=plant.updated_at
        )


class PlantListResponse(BaseModel):
    """Response model for paginated plant list."""
    items: List[PlantResponse]
    total: int
    offset: int
    limit: int


class PlantHealthResponse(BaseModel):
    """Response model for plant health analysis."""
    plant_id: UUID
    health_score: float
    status: str
    last_watered: Optional[datetime] = None
    moisture_level: Optional[float] = None
    light_level: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    recommendations: List[str] = []
    alerts: List[str] = []


# Sensor models
class SensorCreateRequest(BaseModel):
    """Request model for registering a new sensor."""
    sensor_type: str
    hardware_id: constr(min_length=1, max_length=100, strip_whitespace=True)
    name: constr(min_length=1, max_length=100, strip_whitespace=True)
    location: Optional[constr(max_length=200)] = None
    plant_id: Optional[UUID] = None
    configuration: Optional[Dict[str, Any]] = {}

    @validator('sensor_type')
    def validate_sensor_type(cls, v):
        allowed_types = ['moisture', 'temperature', 'humidity', 'ph', 'light', 'nutrient']
        if v not in allowed_types:
            raise ValueError(f'Sensor type must be one of: {", ".join(allowed_types)}')
        return v


class SensorUpdateRequest(BaseModel):
    """Request model for updating sensor configuration."""
    name: Optional[constr(min_length=1, max_length=100, strip_whitespace=True)] = None
    location: Optional[constr(max_length=200)] = None
    configuration: Optional[Dict[str, Any]] = None
    active: Optional[bool] = None


class SensorResponse(BaseModel, TimestampedModel, IdentifiedModel):
    """Response model for sensor data."""
    sensor_type: str
    hardware_id: str
    name: str
    location: Optional[str] = None
    plant_id: Optional[UUID] = None
    configuration: Dict[str, Any] = {}
    calibration_data: Dict[str, Any] = {}
    last_reading_at: Optional[datetime] = None
    status: str = "active"
    active: bool = True

    @classmethod
    def from_domain(cls, sensor: Sensor) -> "SensorResponse":
        """Create response model from domain entity."""
        return cls(
            id=sensor.id.value,
            sensor_type=sensor.sensor_type.value,
            hardware_id=sensor.hardware_id,
            name=sensor.name,
            location=sensor.location,
            plant_id=sensor.plant_id.value if sensor.plant_id else None,
            configuration=sensor.configuration,
            calibration_data=sensor.calibration_data,
            last_reading_at=sensor.last_reading_at,
            status=sensor.status,
            active=sensor.active,
            created_at=sensor.created_at,
            updated_at=sensor.updated_at
        )


class SensorListResponse(BaseModel):
    """Response model for paginated sensor list."""
    items: List[SensorResponse]
    total: int
    offset: int
    limit: int


# Sensor reading models
class SensorReadingCreateRequest(BaseModel):
    """Request model for creating a sensor reading."""
    sensor_id: UUID
    sensor_type: str
    value: float
    unit: str = "unknown"
    timestamp: Optional[datetime] = None
    quality: Optional[str] = "good"
    metadata: Optional[Dict[str, Any]] = {}

    @validator('quality')
    def validate_quality(cls, v):
        allowed_qualities = ['good', 'uncertain', 'bad', 'unknown']
        if v not in allowed_qualities:
            raise ValueError(f'Quality must be one of: {", ".join(allowed_qualities)}')
        return v

    @validator('value')
    def validate_value(cls, v, values):
        """Validate sensor value based on sensor type."""
        sensor_type = values.get('sensor_type')
        if sensor_type == 'moisture' and (v < 0 or v > 100):
            raise ValueError('Moisture value must be between 0 and 100')
        elif sensor_type == 'temperature' and (v < -50 or v > 80):
            raise ValueError('Temperature must be between -50°C and 80°C')
        elif sensor_type == 'humidity' and (v < 0 or v > 100):
            raise ValueError('Humidity value must be between 0 and 100')
        elif sensor_type == 'ph' and (v < 0 or v > 14):
            raise ValueError('pH value must be between 0 and 14')
        elif sensor_type == 'light' and v < 0:
            raise ValueError('Light value must be non-negative')
        return v


class SensorReadingBatchRequest(BaseModel):
    """Request model for batch sensor reading ingestion."""
    readings: List[SensorReadingCreateRequest]
    batch_id: Optional[str] = None

    @validator('readings')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError('Batch cannot be empty')
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 readings')
        return v


class SensorReadingResponse(BaseModel):
    """Response model for sensor reading data."""
    id: UUID
    sensor_id: UUID
    timestamp: datetime
    value: float
    unit: str
    quality: str
    metadata: Dict[str, Any] = {}

    @classmethod
    def from_domain(cls, reading: SensorReading) -> "SensorReadingResponse":
        """Create response model from domain entity."""
        return cls(
            id=reading.id.value,
            sensor_id=reading.sensor_id.value,
            timestamp=reading.timestamp,
            value=reading.value,
            unit=reading.unit,
            quality=reading.quality,
            metadata=reading.metadata
        )


class SensorDataListResponse(BaseModel):
    """Response model for time-series sensor data."""
    sensor_id: UUID
    readings: List[SensorReadingResponse]
    total_count: int
    time_range: Dict[str, Optional[str]]
    pagination: Dict[str, int]


# Aggregation models
class SensorAggregationRequest(BaseModel):
    """Request model for sensor data aggregation."""
    sensor_ids: List[UUID]
    start_time: datetime
    end_time: datetime
    aggregation_type: str = "avg"  # avg, min, max, sum, count
    time_bucket: str = "1hour"  # 1min, 5min, 15min, 1hour, 1day
    group_by_sensor: bool = False

    @validator('aggregation_type')
    def validate_aggregation_type(cls, v):
        allowed_types = ['avg', 'min', 'max', 'sum', 'count', 'stddev']
        if v not in allowed_types:
            raise ValueError(f'Aggregation type must be one of: {", ".join(allowed_types)}')
        return v

    @validator('time_bucket')
    def validate_time_bucket(cls, v):
        allowed_buckets = ['1min', '5min', '15min', '30min', '1hour', '6hour', '12hour', '1day']
        if v not in allowed_buckets:
            raise ValueError(f'Time bucket must be one of: {", ".join(allowed_buckets)}')
        return v

    @root_validator
    def validate_time_range(cls, values):
        start_time = values.get('start_time')
        end_time = values.get('end_time')
        if start_time and end_time and start_time >= end_time:
            raise ValueError('Start time must be before end time')
        return values


class SensorAggregationResponse(BaseModel):
    """Response model for aggregated sensor data."""
    aggregation_type: str
    time_bucket: str
    time_range: Dict[str, str]
    data_points: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]


# Schedule models
class ScheduleCreateRequest(BaseModel):
    """Request model for creating a watering schedule."""
    plant_id: UUID
    schedule_type: str = "automatic"
    start_date: datetime
    time_of_day: str  # "HH:MM" format
    interval_days: PositiveInt
    duration_seconds: PositiveInt
    amount_ml: Optional[PositiveFloat] = None
    enabled: bool = True

    @validator('time_of_day')
    def validate_time_format(cls, v):
        try:
            hour, minute = map(int, v.split(':'))
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                raise ValueError()
        except (ValueError, AttributeError):
            raise ValueError('Time must be in HH:MM format')
        return v

    @validator('schedule_type')
    def validate_schedule_type(cls, v):
        allowed_types = ['automatic', 'manual', 'conditional']
        if v not in allowed_types:
            raise ValueError(f'Schedule type must be one of: {", ".join(allowed_types)}')
        return v


class ScheduleUpdateRequest(BaseModel):
    """Request model for updating a watering schedule."""
    time_of_day: Optional[str] = None
    interval_days: Optional[PositiveInt] = None
    duration_seconds: Optional[PositiveInt] = None
    amount_ml: Optional[PositiveFloat] = None
    enabled: Optional[bool] = None

    @validator('time_of_day')
    def validate_time_format(cls, v):
        if v is not None:
            try:
                hour, minute = map(int, v.split(':'))
                if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                    raise ValueError()
            except (ValueError, AttributeError):
                raise ValueError('Time must be in HH:MM format')
        return v


class ScheduleResponse(BaseModel, TimestampedModel, IdentifiedModel):
    """Response model for watering schedule."""
    plant_id: UUID
    schedule_type: str
    start_date: datetime
    time_of_day: str
    interval_days: int
    duration_seconds: int
    amount_ml: Optional[float] = None
    enabled: bool
    last_executed: Optional[datetime] = None
    next_execution: Optional[datetime] = None


class ScheduleListResponse(BaseModel):
    """Response model for paginated schedule list."""
    items: List[ScheduleResponse]
    total: int
    offset: int
    limit: int


# Manual watering models
class ManualWateringRequest(BaseModel):
    """Request model for manual watering trigger."""
    plant_id: UUID
    duration_seconds: PositiveInt
    amount_ml: Optional[PositiveFloat] = None

    @validator('duration_seconds')
    def validate_duration(cls, v):
        if v > 600:  # 10 minutes max
            raise ValueError('Duration cannot exceed 600 seconds')
        return v


# System info models
class SystemInfoResponse(BaseModel):
    """Response model for system information."""
    version: str
    environment: str
    features: Dict[str, bool]
    limits: Dict[str, int]
    supported_sensors: List[str]


# Pagination models
class PaginationParams(BaseModel):
    """Common pagination parameters."""
    offset: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(50, ge=1, le=1000, description="Maximum number of items to return")

    class Config:
        validate_all = True


