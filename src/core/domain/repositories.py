"""
Repository interfaces for PlantOS domain entities.

These abstract interfaces define the contracts for data access,
enabling easy testing with mock repositories and different storage implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from src.shared.types import PlantID, SensorID, SpeciesID, ScheduleID
from src.core.domain.entities import (
    Plant, PlantSpecies, SensorReading, CareEvent, WateringEvent,
    PlantHealthAssessment, Sensor, Actuator, PlantStatus, WateringSchedule
)


class PlantRepository(ABC):
    """Repository interface for Plant entities."""
    
    @abstractmethod
    async def create(self, plant: Plant) -> Plant:
        """Create a new plant in the repository."""
        pass
    
    @abstractmethod
    async def get_by_id(self, plant_id: PlantID) -> Optional[Plant]:
        """Retrieve a plant by its ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Plant]:
        """Retrieve a plant by its name."""
        pass
    
    @abstractmethod
    async def get_all(self, status: Optional[PlantStatus] = None) -> List[Plant]:
        """Retrieve all plants, optionally filtered by status."""
        pass
    
    @abstractmethod
    async def get_by_species(self, species_id: SpeciesID) -> List[Plant]:
        """Retrieve all plants of a specific species."""
        pass
    
    @abstractmethod
    async def update(self, plant: Plant) -> Plant:
        """Update an existing plant."""
        pass
    
    @abstractmethod
    async def delete(self, plant_id: PlantID) -> bool:
        """Delete a plant by ID. Returns True if deleted, False if not found."""
        pass
    
    @abstractmethod
    async def count(self, status: Optional[PlantStatus] = None) -> int:
        """Count total plants, optionally filtered by status."""
        pass
    
    @abstractmethod
    async def search(self, query: str) -> List[Plant]:
        """Search plants by name or notes."""
        pass


class PlantSpeciesRepository(ABC):
    """Repository interface for PlantSpecies entities."""
    
    @abstractmethod
    async def create(self, species: PlantSpecies) -> PlantSpecies:
        """Create a new plant species."""
        pass
    
    @abstractmethod
    async def get_by_id(self, species_id: SpeciesID) -> Optional[PlantSpecies]:
        """Retrieve a species by its ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[PlantSpecies]:
        """Retrieve a species by its common name."""
        pass
    
    @abstractmethod
    async def get_by_scientific_name(self, scientific_name: str) -> Optional[PlantSpecies]:
        """Retrieve a species by its scientific name."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[PlantSpecies]:
        """Retrieve all plant species."""
        pass
    
    @abstractmethod
    async def update(self, species: PlantSpecies) -> PlantSpecies:
        """Update an existing species."""
        pass
    
    @abstractmethod
    async def delete(self, species_id: SpeciesID) -> bool:
        """Delete a species by ID."""
        pass
    
    @abstractmethod
    async def search(self, query: str) -> List[PlantSpecies]:
        """Search species by name or scientific name."""
        pass


class SensorReadingRepository(ABC):
    """Repository interface for SensorReading entities."""
    
    @abstractmethod
    async def create(self, reading: SensorReading) -> SensorReading:
        """Store a new sensor reading."""
        pass
    
    @abstractmethod
    async def create_batch(self, readings: List[SensorReading]) -> List[SensorReading]:
        """Store multiple sensor readings efficiently."""
        pass
    
    @abstractmethod
    async def get_by_id(self, reading_id: UUID) -> Optional[SensorReading]:
        """Retrieve a specific reading by ID."""
        pass
    
    @abstractmethod
    async def get_latest_for_sensor(self, sensor_id: SensorID) -> Optional[SensorReading]:
        """Get the most recent reading for a sensor."""
        pass
    
    @abstractmethod
    async def get_latest_for_plant(self, plant_id: PlantID, sensor_type: Optional[str] = None) -> List[SensorReading]:
        """Get the latest readings for all sensors on a plant."""
        pass
    
    @abstractmethod
    async def get_readings_for_plant(
        self, 
        plant_id: PlantID,
        sensor_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[SensorReading]:
        """Get readings for a plant within a time range."""
        pass
    
    @abstractmethod
    async def get_readings_for_sensor(
        self,
        sensor_id: SensorID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[SensorReading]:
        """Get readings for a specific sensor within a time range."""
        pass
    
    @abstractmethod
    async def get_by_sensor_id(
        self,
        sensor_id: SensorID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        quality_filter: Optional[str] = None
    ) -> List[SensorReading]:
        """Get readings for a sensor with optional filtering."""
        pass
    
    @abstractmethod
    async def get_average_for_period(
        self,
        sensor_id: SensorID,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[float]:
        """Get average reading value for a sensor over a time period."""
        pass
    
    @abstractmethod
    async def delete_old_readings(self, older_than: datetime) -> int:
        """Delete readings older than specified date. Returns count deleted."""
        pass


class CareEventRepository(ABC):
    """Repository interface for CareEvent entities."""
    
    @abstractmethod
    async def create(self, event: CareEvent) -> CareEvent:
        """Store a new care event."""
        pass
    
    @abstractmethod
    async def get_by_id(self, event_id: UUID) -> Optional[CareEvent]:
        """Retrieve a care event by ID."""
        pass
    
    @abstractmethod
    async def get_events_for_plant(
        self,
        plant_id: PlantID,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[CareEvent]:
        """Get care events for a plant, optionally filtered."""
        pass
    
    @abstractmethod
    async def get_latest_watering(self, plant_id: PlantID) -> Optional[WateringEvent]:
        """Get the most recent watering event for a plant."""
        pass
    
    @abstractmethod
    async def get_watering_history(
        self,
        plant_id: PlantID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[WateringEvent]:
        """Get watering history for a plant."""
        pass
    
    @abstractmethod
    async def update(self, event: CareEvent) -> CareEvent:
        """Update an existing care event."""
        pass
    
    @abstractmethod
    async def delete(self, event_id: UUID) -> bool:
        """Delete a care event."""
        pass
    
    @abstractmethod
    async def get_care_summary(self, plant_id: PlantID, days: int = 30) -> Dict[str, Any]:
        """Get summary of care events for a plant over specified days."""
        pass


class SensorRepository(ABC):
    """Repository interface for Sensor entities."""
    
    @abstractmethod
    async def create(self, sensor: Sensor) -> Sensor:
        """Register a new sensor."""
        pass
    
    @abstractmethod
    async def get_by_id(self, sensor_id: SensorID) -> Optional[Sensor]:
        """Retrieve a sensor by ID."""
        pass
    
    @abstractmethod
    async def get_by_hardware_id(self, hardware_id: str) -> Optional[Sensor]:
        """Retrieve a sensor by its hardware ID."""
        pass
    
    @abstractmethod
    async def get_sensors_for_plant(self, plant_id: PlantID) -> List[Sensor]:
        """Get all sensors attached to a plant."""
        pass
    
    @abstractmethod
    async def get_by_type(self, sensor_type: str) -> List[Sensor]:
        """Get all sensors of a specific type."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str) -> List[Sensor]:
        """Get all sensors with a specific status."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Sensor]:
        """Get all registered sensors."""
        pass
    
    @abstractmethod
    async def update(self, sensor: Sensor) -> Sensor:
        """Update sensor configuration."""
        pass
    
    @abstractmethod
    async def delete(self, sensor_id: SensorID) -> bool:
        """Unregister a sensor."""
        pass
    
    @abstractmethod
    async def update_last_reading_time(self, sensor_id: SensorID, timestamp: datetime) -> bool:
        """Update the last reading timestamp for a sensor."""
        pass


class ActuatorRepository(ABC):
    """Repository interface for Actuator entities."""
    
    @abstractmethod
    async def create(self, actuator: Actuator) -> Actuator:
        """Register a new actuator."""
        pass
    
    @abstractmethod
    async def get_by_id(self, actuator_id: UUID) -> Optional[Actuator]:
        """Retrieve an actuator by ID."""
        pass
    
    @abstractmethod
    async def get_actuators_for_plant(self, plant_id: PlantID) -> List[Actuator]:
        """Get all actuators for a plant."""
        pass
    
    @abstractmethod
    async def get_by_type(self, actuator_type: str) -> List[Actuator]:
        """Get all actuators of a specific type."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str) -> List[Actuator]:
        """Get all actuators with a specific status."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Actuator]:
        """Get all registered actuators."""
        pass
    
    @abstractmethod
    async def update(self, actuator: Actuator) -> Actuator:
        """Update actuator configuration."""
        pass
    
    @abstractmethod
    async def delete(self, actuator_id: UUID) -> bool:
        """Unregister an actuator."""
        pass
    
    @abstractmethod
    async def update_operation_stats(
        self, 
        actuator_id: UUID, 
        operation_time: datetime,
        runtime_seconds: int
    ) -> bool:
        """Update actuator operation statistics."""
        pass


class PlantHealthRepository(ABC):
    """Repository interface for PlantHealthAssessment entities."""
    
    @abstractmethod
    async def create(self, assessment: PlantHealthAssessment) -> PlantHealthAssessment:
        """Store a new health assessment."""
        pass
    
    @abstractmethod
    async def get_latest_for_plant(self, plant_id: PlantID) -> Optional[PlantHealthAssessment]:
        """Get the most recent health assessment for a plant."""
        pass
    
    @abstractmethod
    async def get_history_for_plant(
        self,
        plant_id: PlantID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PlantHealthAssessment]:
        """Get health assessment history for a plant."""
        pass
    
    @abstractmethod
    async def get_plants_by_health_score(
        self,
        min_score: float = 0.0,
        max_score: float = 1.0
    ) -> List[PlantHealthAssessment]:
        """Get plants within a health score range."""
        pass
    
    @abstractmethod
    async def delete_old_assessments(self, older_than: datetime) -> int:
        """Delete assessments older than specified date."""
        pass


class WateringScheduleRepository(ABC):
    """Repository interface for WateringSchedule entities."""
    
    @abstractmethod
    async def create(self, schedule: WateringSchedule) -> WateringSchedule:
        """Create a new watering schedule."""
        pass
    
    @abstractmethod
    async def get_by_id(self, schedule_id: ScheduleID) -> Optional[WateringSchedule]:
        """Retrieve a schedule by its ID."""
        pass
    
    @abstractmethod
    async def get_by_plant_id(self, plant_id: PlantID) -> List[WateringSchedule]:
        """Get all schedules for a specific plant."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[WateringSchedule]:
        """Get all watering schedules."""
        pass
    
    @abstractmethod
    async def get_due_schedules(self, current_time: datetime) -> List[WateringSchedule]:
        """Get schedules that are due for execution."""
        pass
    
    @abstractmethod
    async def get_enabled_schedules(self) -> List[WateringSchedule]:
        """Get all enabled schedules."""
        pass
    
    @abstractmethod
    async def update(self, schedule: WateringSchedule) -> WateringSchedule:
        """Update an existing schedule."""
        pass
    
    @abstractmethod
    async def delete(self, schedule_id: ScheduleID) -> bool:
        """Delete a schedule by ID."""
        pass
    
    @abstractmethod
    async def mark_executed(self, schedule_id: ScheduleID, execution_time: datetime) -> bool:
        """Mark a schedule as executed and update next execution time."""
        pass