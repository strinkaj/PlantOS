"""
Domain entities for PlantOS.

These represent the core business objects in our plant care system.
All entities are immutable and type-safe with comprehensive validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from src.shared.types import (
    PlantID, SensorID, SpeciesID, 
    WaterAmount, TemperatureCelsius, HumidityPercent, 
    MoisturePercent, QualityScore, HealthScore,
    DurationSeconds, IntervalHours
)


class PlantStatus(str, Enum):
    """Current status of a plant in the system."""
    ACTIVE = "active"
    DORMANT = "dormant"
    SICK = "sick"
    DECEASED = "deceased"
    MAINTENANCE = "maintenance"


class CareEventType(str, Enum):
    """Types of care events that can be applied to plants."""
    WATERING = "watering"
    FERTILIZING = "fertilizing"
    PRUNING = "pruning"
    REPOTTING = "repotting"
    TREATMENT = "treatment"


class TriggerType(str, Enum):
    """How a care event was triggered."""
    AUTOMATED = "automated"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class MoistureRange:
    """Optimal moisture range for a plant species."""
    minimum: MoisturePercent
    maximum: MoisturePercent
    
    def __post_init__(self):
        if self.minimum >= self.maximum:
            raise ValueError("Minimum moisture must be less than maximum")
        if not (0 <= self.minimum <= 100 and 0 <= self.maximum <= 100):
            raise ValueError("Moisture percentages must be between 0 and 100")


@dataclass(frozen=True)
class TemperatureRange:
    """Optimal temperature range for a plant species."""
    minimum: TemperatureCelsius
    maximum: TemperatureCelsius
    
    def __post_init__(self):
        if self.minimum >= self.maximum:
            raise ValueError("Minimum temperature must be less than maximum")


@dataclass(frozen=True)
class PlantSpecies:
    """Plant species with care requirements and characteristics."""
    id: SpeciesID
    name: str
    scientific_name: str
    optimal_moisture: MoistureRange
    optimal_temperature: TemperatureRange
    optimal_humidity: HumidityPercent
    water_frequency_hours: IntervalHours
    light_requirements: str  # "low", "medium", "bright", "direct"
    care_instructions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Species name cannot be empty")
        if not (0 <= self.optimal_humidity <= 100):
            raise ValueError("Humidity must be between 0 and 100")
        if self.water_frequency_hours <= 0:
            raise ValueError("Water frequency must be positive")


@dataclass
class Plant:
    """A plant entity with care history and current status."""
    id: PlantID
    name: str
    species_id: Optional[SpeciesID] = None
    location: Optional[str] = None
    status: PlantStatus = PlantStatus.ACTIVE
    health_score: Optional[HealthScore] = None
    last_watered_at: Optional[datetime] = None
    last_fertilized_at: Optional[datetime] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Plant name cannot be empty")
        if self.health_score is not None and not (0 <= self.health_score <= 1):
            raise ValueError("Health score must be between 0 and 1")
    
    def update_health_score(self, score: HealthScore) -> 'Plant':
        """Update plant health score and timestamp."""
        if not (0 <= score <= 1):
            raise ValueError("Health score must be between 0 and 1")
        
        self.health_score = score
        self.updated_at = datetime.utcnow()
        return self
    
    def mark_watered(self, watered_at: Optional[datetime] = None) -> 'Plant':
        """Mark plant as watered at specified time."""
        self.last_watered_at = watered_at or datetime.utcnow()
        self.updated_at = datetime.utcnow()
        return self
    
    def update_status(self, status: PlantStatus) -> 'Plant':
        """Update plant status with timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()
        return self
    
    def needs_watering(self, current_moisture: MoisturePercent, species: PlantSpecies) -> bool:
        """Determine if plant needs watering based on current moisture and species requirements."""
        if current_moisture < species.optimal_moisture.minimum:
            return True
        
        # Check if it's been too long since last watering
        if self.last_watered_at:
            hours_since_watered = (datetime.utcnow() - self.last_watered_at).total_seconds() / 3600
            return hours_since_watered >= species.water_frequency_hours
        
        return False


@dataclass(frozen=True)
class SensorReading:
    """A sensor reading with quality assessment and metadata."""
    id: UUID
    sensor_id: SensorID
    plant_id: Optional[PlantID]
    sensor_type: str
    value: Decimal
    unit: str
    quality_score: QualityScore
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not (0 <= self.quality_score <= 1):
            raise ValueError("Quality score must be between 0 and 1")
        if not self.unit.strip():
            raise ValueError("Unit cannot be empty")


@dataclass(frozen=True)
class CareEvent:
    """A care event applied to a plant."""
    id: UUID
    plant_id: PlantID
    event_type: CareEventType
    triggered_by: TriggerType
    amount: Optional[Decimal] = None  # Amount of water, fertilizer, etc.
    unit: Optional[str] = None
    duration_seconds: Optional[DurationSeconds] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    performed_at: datetime = field(default_factory=datetime.utcnow)
    performed_by: Optional[str] = None  # User ID or system identifier
    
    def __post_init__(self):
        if self.amount is not None and self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("Duration must be positive")


@dataclass(frozen=True)
class WateringEvent(CareEvent):
    """Specialized care event for watering operations."""
    water_amount: WaterAmount
    moisture_before: Optional[MoisturePercent] = None
    moisture_after: Optional[MoisturePercent] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.water_amount <= 0:
            raise ValueError("Water amount must be positive")
        
        # Validate moisture levels if provided
        if self.moisture_before is not None and not (0 <= self.moisture_before <= 100):
            raise ValueError("Moisture before must be between 0 and 100")
        if self.moisture_after is not None and not (0 <= self.moisture_after <= 100):
            raise ValueError("Moisture after must be between 0 and 100")


@dataclass
class PlantHealthAssessment:
    """Assessment of plant health based on multiple factors."""
    plant_id: PlantID
    overall_score: HealthScore
    moisture_score: HealthScore
    growth_score: HealthScore
    care_consistency_score: HealthScore
    factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    assessed_by: str = "system"
    
    def __post_init__(self):
        scores = [self.overall_score, self.moisture_score, self.growth_score, self.care_consistency_score]
        for score in scores:
            if not (0 <= score <= 1):
                raise ValueError("All health scores must be between 0 and 1")


@dataclass(frozen=True)
class Sensor:
    """A sensor device attached to a plant or environment."""
    id: SensorID
    plant_id: Optional[PlantID]
    sensor_type: str
    model: str
    location: str
    gpio_pin: Optional[int] = None
    i2c_address: Optional[int] = None
    calibration_offset: Decimal = Decimal('0.0')
    calibration_multiplier: Decimal = Decimal('1.0')
    status: str = "active"  # active, inactive, maintenance, error
    last_reading_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.sensor_type.strip():
            raise ValueError("Sensor type cannot be empty")
        if self.gpio_pin is not None and not (0 <= self.gpio_pin <= 40):  # Raspberry Pi GPIO range
            raise ValueError("GPIO pin must be between 0 and 40")
        if self.calibration_multiplier == 0:
            raise ValueError("Calibration multiplier cannot be zero")


@dataclass(frozen=True)
class Actuator:
    """An actuator device (pump, valve, etc.) for plant care."""
    id: UUID
    plant_id: Optional[PlantID]
    actuator_type: str  # pump, valve, fan, light, heater
    model: str
    location: str
    gpio_pin: Optional[int] = None
    max_runtime_seconds: DurationSeconds = 300  # Safety limit: 5 minutes
    status: str = "idle"  # idle, active, maintenance, error
    last_operation_at: Optional[datetime] = None
    total_runtime_seconds: int = 0
    operation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.actuator_type.strip():
            raise ValueError("Actuator type cannot be empty")
        if self.gpio_pin is not None and not (0 <= self.gpio_pin <= 40):
            raise ValueError("GPIO pin must be between 0 and 40")
        if self.max_runtime_seconds <= 0:
            raise ValueError("Max runtime must be positive")


# Factory functions for common entity creation
def create_plant(name: str, species_id: Optional[SpeciesID] = None, location: Optional[str] = None) -> Plant:
    """Factory function to create a new plant with sensible defaults."""
    return Plant(
        id=PlantID(uuid4()),
        name=name.strip(),
        species_id=species_id,
        location=location,
        status=PlantStatus.ACTIVE,
        health_score=HealthScore(0.8),  # Start with good health
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


def create_watering_event(
    plant_id: PlantID, 
    water_amount: WaterAmount, 
    trigger: TriggerType = TriggerType.AUTOMATED,
    moisture_before: Optional[MoisturePercent] = None
) -> WateringEvent:
    """Factory function to create a watering event."""
    return WateringEvent(
        id=uuid4(),
        plant_id=plant_id,
        event_type=CareEventType.WATERING,
        triggered_by=trigger,
        water_amount=water_amount,
        moisture_before=moisture_before,
        amount=Decimal(str(water_amount)),
        unit="ml",
        performed_at=datetime.utcnow()
    )


def create_sensor_reading(
    sensor_id: SensorID,
    plant_id: Optional[PlantID],
    sensor_type: str,
    value: Decimal,
    unit: str,
    quality_score: QualityScore = QualityScore(1.0)
) -> SensorReading:
    """Factory function to create a sensor reading."""
    return SensorReading(
        id=uuid4(),
        sensor_id=sensor_id,
        plant_id=plant_id,
        sensor_type=sensor_type,
        value=value,
        unit=unit,
        quality_score=quality_score,
        timestamp=datetime.utcnow()
    )