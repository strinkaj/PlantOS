"""
Sensor management use cases.

These use cases handle sensor registration, configuration updates,
and sensor lifecycle management within the plant care system.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

import structlog

from src.core.domain.entities import Sensor, SensorReading
from src.core.domain.repositories import SensorRepository, SensorReadingRepository
from src.shared.types import SensorID, SensorType, PlantID
from src.shared.exceptions import (
    SensorNotFoundError,
    SensorAlreadyExistsError,
    ValidationError
)
from src.shared.contracts import require, ensure

logger = structlog.get_logger(__name__)


class RegisterSensorUseCase:
    """Use case for registering a new sensor in the system."""
    
    def __init__(self, sensor_repo: SensorRepository):
        self.sensor_repo = sensor_repo
        
    @require(lambda sensor_type, hardware_id, name: sensor_type and hardware_id and name, 
             "Sensor type, hardware ID, and name are required")
    @require(lambda sensor_type, hardware_id, name: len(hardware_id.strip()) > 0,
             "Hardware ID cannot be empty")
    @require(lambda sensor_type, hardware_id, name: len(name.strip()) >= 2,
             "Sensor name must be at least 2 characters")
    async def execute(
        self,
        sensor_type: str,
        hardware_id: str,
        name: str,
        location: Optional[str] = None,
        plant_id: Optional[UUID] = None,
        configuration: Optional[Dict[str, Any]] = None
    ) -> Sensor:
        """
        Register a new sensor in the system.
        
        Args:
            sensor_type: Type of sensor (moisture, temperature, etc.)
            hardware_id: Unique hardware identifier
            name: Human-readable sensor name
            location: Optional physical location description
            plant_id: Optional ID of associated plant
            configuration: Optional sensor configuration parameters
            
        Returns:
            Created sensor entity
            
        Raises:
            SensorAlreadyExistsError: If hardware ID already exists
            ValidationError: If input validation fails
        """
        logger.info(
            "Registering new sensor",
            sensor_type=sensor_type,
            hardware_id=hardware_id,
            name=name
        )
        
        # Validate sensor type
        allowed_types = ['moisture', 'temperature', 'humidity', 'ph', 'light', 'nutrient']
        if sensor_type not in allowed_types:
            raise ValidationError(f"Sensor type must be one of: {', '.join(allowed_types)}")
            
        # Check if hardware ID already exists
        existing_sensor = await self.sensor_repo.get_by_hardware_id(hardware_id.strip())
        if existing_sensor:
            raise SensorAlreadyExistsError(f"Sensor with hardware ID '{hardware_id}' already exists")
            
        # Create sensor entity
        sensor = Sensor(
            id=SensorID(uuid4()),
            sensor_type=SensorType(sensor_type),
            hardware_id=hardware_id.strip(),
            name=name.strip(),
            location=location,
            plant_id=PlantID(plant_id) if plant_id else None,
            configuration=configuration or {},
            calibration_data={},
            last_reading_at=None,
            status="active",
            active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Save to repository
        created_sensor = await self.sensor_repo.create(sensor)
        
        logger.info(
            "Sensor registered successfully",
            sensor_id=str(created_sensor.id),
            hardware_id=hardware_id
        )
        
        return created_sensor


class UpdateSensorUseCase:
    """Use case for updating sensor configuration and metadata."""
    
    def __init__(self, sensor_repo: SensorRepository):
        self.sensor_repo = sensor_repo
        
    @require(lambda sensor_id: sensor_id is not None, "Sensor ID cannot be None")
    async def execute(
        self,
        sensor_id: UUID,
        name: Optional[str] = None,
        location: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
        active: Optional[bool] = None
    ) -> Sensor:
        """
        Update an existing sensor's configuration.
        
        Args:
            sensor_id: ID of the sensor to update
            name: Optional new name
            location: Optional new location
            configuration: Optional new configuration
            active: Optional active status
            
        Returns:
            Updated sensor entity
            
        Raises:
            SensorNotFoundError: If sensor doesn't exist
        """
        logger.info("Updating sensor", sensor_id=str(sensor_id))
        
        # Get existing sensor
        sensor = await self.sensor_repo.get_by_id(SensorID(sensor_id))
        if not sensor:
            raise SensorNotFoundError(f"Sensor with ID {sensor_id} not found")
            
        # Update fields
        if name is not None:
            if len(name.strip()) < 2:
                raise ValidationError("Sensor name must be at least 2 characters")
            sensor.name = name.strip()
            
        if location is not None:
            sensor.location = location
            
        if configuration is not None:
            # Merge with existing configuration
            sensor.configuration = {**sensor.configuration, **configuration}
            
        if active is not None:
            sensor.active = active
            if not active:
                sensor.status = "inactive"
            else:
                sensor.status = "active"
                
        sensor.updated_at = datetime.utcnow()
        
        # Save changes
        updated_sensor = await self.sensor_repo.update(sensor)
        
        logger.info("Sensor updated successfully", sensor_id=str(sensor_id))
        return updated_sensor


class ListSensorsUseCase:
    """Use case for listing sensors with filtering options."""
    
    def __init__(self, sensor_repo: SensorRepository):
        self.sensor_repo = sensor_repo
        
    @require(lambda offset, limit: offset >= 0, "Offset must be non-negative")
    @require(lambda offset, limit: limit > 0, "Limit must be positive")
    @require(lambda offset, limit: limit <= 1000, "Limit cannot exceed 1000")
    async def execute(
        self,
        offset: int = 0,
        limit: int = 50,
        sensor_type: Optional[str] = None,
        plant_id: Optional[UUID] = None,
        active_only: bool = True
    ) -> tuple[List[Sensor], int]:
        """
        List sensors with pagination and filtering.
        
        Args:
            offset: Number of sensors to skip
            limit: Maximum number of sensors to return
            sensor_type: Optional sensor type filter
            plant_id: Optional plant ID filter
            active_only: Whether to return only active sensors
            
        Returns:
            Tuple of (sensors list, total count)
        """
        logger.info(
            "Listing sensors",
            offset=offset,
            limit=limit,
            sensor_type=sensor_type,
            plant_id=str(plant_id) if plant_id else None,
            active_only=active_only
        )
        
        # Get all sensors
        all_sensors = await self.sensor_repo.get_all()
        
        # Apply filters
        filtered_sensors = []
        for sensor in all_sensors:
            # Active filter
            if active_only and not sensor.active:
                continue
                
            # Sensor type filter
            if sensor_type and sensor.sensor_type.value != sensor_type:
                continue
                
            # Plant ID filter
            if plant_id and (not sensor.plant_id or sensor.plant_id.value != plant_id):
                continue
                
            filtered_sensors.append(sensor)
            
        # Apply pagination
        total_count = len(filtered_sensors)
        paginated_sensors = filtered_sensors[offset:offset + limit]
        
        logger.info(
            "Sensors listed successfully",
            returned_count=len(paginated_sensors),
            total_count=total_count
        )
        
        return paginated_sensors, total_count


class GetSensorReadingsUseCase:
    """Use case for retrieving recent readings for a specific sensor."""
    
    def __init__(
        self,
        sensor_repo: SensorRepository,
        sensor_reading_repo: SensorReadingRepository
    ):
        self.sensor_repo = sensor_repo
        self.sensor_reading_repo = sensor_reading_repo
        
    @require(lambda sensor_id: sensor_id is not None, "Sensor ID cannot be None")
    @require(lambda sensor_id, limit: limit > 0, "Limit must be positive")
    @require(lambda sensor_id, limit: limit <= 1000, "Limit cannot exceed 1000")
    async def execute(
        self,
        sensor_id: UUID,
        limit: int = 100,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get recent readings for a specific sensor.
        
        Args:
            sensor_id: ID of the sensor
            limit: Maximum number of readings to return
            hours: Number of hours back to look for readings
            
        Returns:
            Dictionary with sensor info and readings
            
        Raises:
            SensorNotFoundError: If sensor doesn't exist
        """
        logger.info(
            "Getting sensor readings",
            sensor_id=str(sensor_id),
            limit=limit,
            hours=hours
        )
        
        # Verify sensor exists
        sensor = await self.sensor_repo.get_by_id(SensorID(sensor_id))
        if not sensor:
            raise SensorNotFoundError(f"Sensor with ID {sensor_id} not found")
            
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get readings
        readings = await self.sensor_reading_repo.get_by_sensor_id(
            sensor_id=SensorID(sensor_id),
            start_time=start_time,
            end_time=end_time
        )
        
        # Sort by timestamp (newest first) and apply limit
        readings.sort(key=lambda r: r.timestamp, reverse=True)
        recent_readings = readings[:limit]
        
        # Calculate basic statistics
        values = [r.value for r in readings]
        statistics = {}
        if values:
            statistics = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": recent_readings[0].value if recent_readings else None,
                "latest_timestamp": recent_readings[0].timestamp.isoformat() if recent_readings else None
            }
            
        # Quality distribution
        quality_counts = {"good": 0, "uncertain": 0, "bad": 0, "unknown": 0}
        for reading in readings:
            quality_counts[reading.quality] = quality_counts.get(reading.quality, 0) + 1
            
        result = {
            "sensor": {
                "id": str(sensor.id.value),
                "name": sensor.name,
                "type": sensor.sensor_type.value,
                "hardware_id": sensor.hardware_id,
                "location": sensor.location,
                "active": sensor.active,
                "last_reading_at": sensor.last_reading_at.isoformat() if sensor.last_reading_at else None
            },
            "readings": [
                {
                    "id": str(reading.id.value),
                    "timestamp": reading.timestamp.isoformat(),
                    "value": reading.value,
                    "unit": reading.unit,
                    "quality": reading.quality,
                    "metadata": reading.metadata
                }
                for reading in recent_readings
            ],
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "statistics": statistics,
            "data_quality": quality_counts
        }
        
        logger.info(
            "Sensor readings retrieved successfully",
            sensor_id=str(sensor_id),
            readings_count=len(recent_readings)
        )
        
        return result