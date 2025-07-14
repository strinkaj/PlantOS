"""
Application business logic and use cases.

This module contains all the use case implementations that orchestrate
the business logic of the PlantOS system following Clean Architecture principles.
"""

# Plant management use cases
from .plant_management import (
    CreatePlantUseCase,
    GetPlantUseCase,
    ListPlantsUseCase,
    UpdatePlantUseCase,
    DeletePlantUseCase,
    CalculatePlantHealthUseCase
)

# Sensor data use cases
from .sensor_data import (
    IngestSensorDataUseCase,
    BatchIngestSensorDataUseCase,
    GetSensorDataUseCase,
    GetAggregatedSensorDataUseCase
)

# Sensor management use cases
from .sensor_management import (
    RegisterSensorUseCase,
    UpdateSensorUseCase,
    ListSensorsUseCase,
    GetSensorReadingsUseCase
)

# Schedule management use cases
from .schedule_management import (
    CreateScheduleUseCase,
    UpdateScheduleUseCase,
    ListSchedulesUseCase,
    TriggerManualWateringUseCase
)

__all__ = [
    # Plant management
    "CreatePlantUseCase",
    "GetPlantUseCase", 
    "ListPlantsUseCase",
    "UpdatePlantUseCase",
    "DeletePlantUseCase",
    "CalculatePlantHealthUseCase",
    
    # Sensor data
    "IngestSensorDataUseCase",
    "BatchIngestSensorDataUseCase", 
    "GetSensorDataUseCase",
    "GetAggregatedSensorDataUseCase",
    
    # Sensor management
    "RegisterSensorUseCase",
    "UpdateSensorUseCase",
    "ListSensorsUseCase", 
    "GetSensorReadingsUseCase",
    
    # Schedule management
    "CreateScheduleUseCase",
    "UpdateScheduleUseCase",
    "ListSchedulesUseCase",
    "TriggerManualWateringUseCase"
]