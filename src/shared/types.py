"""
Type definitions for PlantOS.

This module contains all custom type definitions used throughout
the application to ensure type safety and clear interfaces.
"""

from typing import NewType
from uuid import UUID
from decimal import Decimal

# Domain-specific type aliases for better type safety
PlantID = NewType('PlantID', UUID)
SensorID = NewType('SensorID', UUID)
SpeciesID = NewType('SpeciesID', UUID)
UserID = NewType('UserID', UUID)

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