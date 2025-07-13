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

from src.shared.types import PlantID, SensorID, SpeciesID
from src.core.domain.entities import PlantStatus, CareEventType, TriggerType


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


# Request models (for API inputs)
class CreatePlantRequest(BaseModel):
    """Request model for creating a new plant."""
    name: constr(min_length=1, max_length=255, strip_whitespace=True)
    species_id: Optional[UUID] = None
    location: Optional[constr(max_length=255)] = None
    notes: Optional[str] = ""
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Plant name cannot be empty')
        return v.strip()


class UpdatePlantRequest(BaseModel):
    """Request model for updating a plant."""
    name: Optional[constr(min_length=1, max_length=255, strip_whitespace=True)] = None
    species_id: Optional[UUID] = None
    location: Optional[constr(max_length=255)] = None
    status: Optional[PlantStatus] = None
    notes: Optional[str] = None
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Plant name cannot be empty')
        return v.strip() if v else v


class CreateSpeciesRequest(BaseModel):
    """Request model for creating a plant species."""
    name: constr(min_length=1, max_length=255, strip_whitespace=True)
    scientific_name: constr(min_length=1, max_length=255, strip_whitespace=True)
    optimal_moisture_min: confloat(ge=0, le=100) = Field(..., description="Minimum optimal moisture percentage")
    optimal_moisture_max: confloat(ge=0, le=100) = Field(..., description="Maximum optimal moisture percentage")
    optimal_temp_min: float = Field(..., description="Minimum optimal temperature in Celsius")
    optimal_temp_max: float = Field(..., description="Maximum optimal temperature in Celsius")
    optimal_humidity: confloat(ge=0, le=100) = Field(..., description="Optimal humidity percentage")
    water_frequency_hours: PositiveInt = Field(..., description="Watering frequency in hours")
    light_requirements: constr(regex=r'^(low|medium|bright|direct)$') = Field(..., description="Light requirements")
    care_instructions: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('optimal_moisture_max')
    def moisture_max_greater_than_min(cls, v, values):
        if 'optimal_moisture_min' in values and v <= values['optimal_moisture_min']:
            raise ValueError('Maximum moisture must be greater than minimum moisture')
        return v
    
    @validator('optimal_temp_max')
    def temp_max_greater_than_min(cls, v, values):
        if 'optimal_temp_min' in values and v <= values['optimal_temp_min']:
            raise ValueError('Maximum temperature must be greater than minimum temperature')
        return v


class CreateSensorRequest(BaseModel):
    """Request model for registering a sensor."""
    plant_id: Optional[UUID] = None
    sensor_type: constr(min_length=1, max_length=50)
    model: constr(min_length=1, max_length=100)
    location: constr(min_length=1, max_length=255)
    gpio_pin: Optional[int] = Field(None, ge=0, le=40, description="GPIO pin number for Raspberry Pi")
    i2c_address: Optional[int] = Field(None, ge=0, le=127, description="I2C address in hexadecimal")
    calibration_offset: Decimal = Field(Decimal('0.0'), description="Calibration offset value")
    calibration_multiplier: Decimal = Field(Decimal('1.0'), description="Calibration multiplier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('calibration_multiplier')
    def multiplier_not_zero(cls, v):
        if v == 0:
            raise ValueError('Calibration multiplier cannot be zero')
        return v


class WaterPlantRequest(BaseModel):
    """Request model for watering a plant."""
    amount_ml: PositiveInt = Field(..., description="Amount of water in milliliters")
    duration_seconds: Optional[PositiveInt] = Field(None, description="Watering duration in seconds")
    notes: Optional[str] = ""
    
    @validator('amount_ml')
    def reasonable_water_amount(cls, v):
        if v > 2000:  # 2 liters safety limit
            raise ValueError('Water amount too large (max 2000ml)')
        return v
    
    @validator('duration_seconds')
    def reasonable_duration(cls, v):
        if v is not None and v > 300:  # 5 minutes safety limit
            raise ValueError('Watering duration too long (max 300 seconds)')
        return v


class SensorReadingRequest(BaseModel):
    """Request model for submitting sensor readings."""
    sensor_id: UUID
    value: Union[float, int, Decimal]
    unit: constr(min_length=1, max_length=20)
    quality_score: confloat(ge=0, le=1) = 1.0
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow()


# Response models (for API outputs)
class PlantResponse(IdentifiedModel, TimestampedModel):
    """Response model for plant data."""
    name: str
    species_id: Optional[UUID]
    species_name: Optional[str] = None
    location: Optional[str]
    status: PlantStatus
    health_score: Optional[float]
    last_watered_at: Optional[datetime]
    last_fertilized_at: Optional[datetime]
    notes: str
    metadata: Dict[str, Any]
    
    @validator('health_score')
    def health_score_range(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError('Health score must be between 0 and 1')
        return v


class PlantSummaryResponse(BaseModel):
    """Simplified plant response for list views."""
    id: UUID
    name: str
    status: PlantStatus
    health_score: Optional[float]
    last_watered_at: Optional[datetime]
    needs_attention: bool = False


class SpeciesResponse(IdentifiedModel, TimestampedModel):
    """Response model for plant species data."""
    name: str
    scientific_name: str
    optimal_moisture_min: float
    optimal_moisture_max: float
    optimal_temp_min: float
    optimal_temp_max: float
    optimal_humidity: float
    water_frequency_hours: int
    light_requirements: str
    care_instructions: Dict[str, Any]


class SensorResponse(IdentifiedModel, TimestampedModel):
    """Response model for sensor data."""
    plant_id: Optional[UUID]
    sensor_type: str
    model: str
    location: str
    gpio_pin: Optional[int]
    i2c_address: Optional[int]
    status: str
    last_reading_at: Optional[datetime]
    metadata: Dict[str, Any]


class SensorReadingResponse(IdentifiedModel):
    """Response model for sensor reading data."""
    sensor_id: UUID
    plant_id: Optional[UUID]
    sensor_type: str
    value: Decimal
    unit: str
    quality_score: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]


class CareEventResponse(IdentifiedModel):
    """Response model for care event data."""
    plant_id: UUID
    event_type: CareEventType
    triggered_by: TriggerType
    amount: Optional[Decimal]
    unit: Optional[str]
    duration_seconds: Optional[int]
    notes: str
    metadata: Dict[str, Any]
    performed_at: datetime
    performed_by: Optional[str]


class WateringEventResponse(CareEventResponse):
    """Response model for watering event data."""
    water_amount: int
    moisture_before: Optional[float]
    moisture_after: Optional[float]


class PlantHealthResponse(BaseModel):
    """Response model for plant health assessment."""
    plant_id: UUID
    overall_score: float
    moisture_score: float
    growth_score: float
    care_consistency_score: float
    factors: List[str]
    recommendations: List[str]
    assessed_at: datetime
    assessed_by: str


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    status: str = "healthy"
    timestamp: datetime
    services: Dict[str, str]
    hardware: Dict[str, str]
    database: Dict[str, Any]
    metrics: Dict[str, float]


class PlantCareStatsResponse(BaseModel):
    """Response model for plant care statistics."""
    plant_id: UUID
    total_waterings: int
    total_water_ml: int
    avg_days_between_watering: float
    last_watering_date: Optional[datetime]
    care_consistency_score: float
    time_period_days: int


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationErrorResponse(BaseModel):
    """Validation error response with field details."""
    error: str = "Validation failed"
    detail: str
    field_errors: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Pagination models
class PaginationParams(BaseModel):
    """Query parameters for pagination."""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    size: int = Field(20, ge=1, le=100, description="Page size (max 100)")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    
    @validator('pages', pre=True, always=True)
    def calculate_pages(cls, v, values):
        if 'total' in values and 'size' in values:
            import math
            return math.ceil(values['total'] / values['size'])
        return v


# Query parameter models
class PlantQueryParams(PaginationParams):
    """Query parameters for plant listing."""
    status: Optional[PlantStatus] = None
    species_id: Optional[UUID] = None
    search: Optional[str] = None
    sort_by: str = Field("created_at", regex=r'^(name|created_at|updated_at|health_score)$')
    sort_order: str = Field("desc", regex=r'^(asc|desc)$')


class SensorReadingQueryParams(PaginationParams):
    """Query parameters for sensor reading queries."""
    sensor_id: Optional[UUID] = None
    plant_id: Optional[UUID] = None
    sensor_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @root_validator
    def validate_time_range(cls, values):
        start_time = values.get('start_time')
        end_time = values.get('end_time')
        
        if start_time and end_time and start_time >= end_time:
            raise ValueError('start_time must be before end_time')
        
        return values


# Configuration models
class APIConfig(BaseModel):
    """API configuration model."""
    title: str = "PlantOS API"
    version: str = "1.0.0"
    description: str = "Production-grade plant care automation system"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Pagination defaults
    default_page_size: int = 20
    max_page_size: int = 100
    
    # CORS
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    cors_headers: List[str] = ["*"]
    
    class Config:
        env_prefix = "API_"