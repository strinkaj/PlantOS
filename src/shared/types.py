"""
Type definitions for PlantOS.

This module contains all custom type definitions used throughout
the application to ensure type safety and clear interfaces.
"""

from typing import NewType, Optional
from uuid import UUID
from decimal import Decimal
from enum import Enum

# Domain-specific type aliases for better type safety
PlantID = NewType('PlantID', UUID)
SensorID = NewType('SensorID', UUID)
SpeciesID = NewType('SpeciesID', UUID)
UserID = NewType('UserID', UUID)
ScheduleID = NewType('ScheduleID', UUID)
SensorReadingID = NewType('SensorReadingID', UUID)

# Hardware-specific types
PinNumber = NewType('PinNumber', int)
SensorReading = NewType('SensorReading', Decimal)
WaterAmount = NewType('WaterAmount', int)  # milliliters
TemperatureCelsius = NewType('TemperatureCelsius', Decimal)
HumidityPercent = NewType('HumidityPercent', Decimal)
MoisturePercent = NewType('MoisturePercent', Decimal)
PHLevel = NewType('PHLevel', Decimal)

# Time-related types
DurationSeconds = NewType('DurationSeconds', int)
IntervalHours = NewType('IntervalHours', int)

# Quality and health scores (0.0 to 1.0)
QualityScore = NewType('QualityScore', float)
HealthScore = NewType('HealthScore', float)


# Enums for domain objects
class ScheduleType(str, Enum):
    """Types of watering schedules."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    CONDITIONAL = "conditional"


class SensorType(str, Enum):
    """Types of sensors supported by the system."""
    MOISTURE = "moisture"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PH = "ph"
    LIGHT = "light"
    NUTRIENT = "nutrient"


class Location:
    """Location value object for plant placement."""
    
    def __init__(self, room: str, position: Optional[str] = None, light_level: Optional[float] = None):
        self.room = room
        self.position = position
        self.light_level = light_level
        
    def __str__(self):
        return f"{self.room}" + (f" - {self.position}" if self.position else "")