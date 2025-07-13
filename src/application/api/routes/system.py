"""
System management and administration API routes.

This module provides endpoints for system configuration, sensor management,
watering schedules, and administrative operations.
"""

from datetime import datetime, time
from typing import List, Optional, Dict, Any
from uuid import UUID
import os

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from pydantic import Field
import structlog
import aioredis

from src.core.domain.entities import Sensor, WateringSchedule
from src.core.domain.repositories import SensorRepository, WateringScheduleRepository
from src.core.domain.value_objects import SensorID, PlantID, SensorType, ScheduleType
from src.core.use_cases.sensor_management import (
    RegisterSensorUseCase,
    UpdateSensorUseCase,
    ListSensorsUseCase,
    GetSensorReadingsUseCase
)
from src.core.use_cases.schedule_management import (
    CreateScheduleUseCase,
    UpdateScheduleUseCase,
    ListSchedulesUseCase,
    TriggerManualWateringUseCase
)
from src.infrastructure.database.dependency import (
    get_sensor_repository,
    get_schedule_repository
)
from src.infrastructure.cache.dependency import get_redis_client
from src.shared.exceptions import SensorNotFoundError, ScheduleNotFoundError
from src.application.models import (
    SensorCreateRequest,
    SensorUpdateRequest,
    SensorResponse,
    SensorListResponse,
    SensorReadingResponse,
    ScheduleCreateRequest,
    ScheduleUpdateRequest,
    ScheduleResponse,
    ScheduleListResponse,
    ManualWateringRequest,
    SystemInfoResponse,
    PaginationParams
)

logger = structlog.get_logger(__name__)
router = APIRouter()


# Sensor Management Endpoints

@router.post("/sensors", response_model=SensorResponse, status_code=status.HTTP_201_CREATED)
async def register_sensor(
    request: SensorCreateRequest,
    sensor_repo: SensorRepository = Depends(get_sensor_repository)
) -> SensorResponse:
    """
    Register a new sensor in the system.
    
    Args:
        request: Sensor registration data
        sensor_repo: Sensor repository dependency
        
    Returns:
        Registered sensor details
        
    Raises:
        400: Invalid sensor configuration
        409: Sensor with same hardware ID already exists
    """
    logger.info(
        "Registering new sensor",
        sensor_type=request.sensor_type,
        hardware_id=request.hardware_id
    )
    
    try:
        use_case = RegisterSensorUseCase(sensor_repo)
        sensor = await use_case.execute(
            sensor_type=SensorType(request.sensor_type),
            hardware_id=request.hardware_id,
            name=request.name,
            location=request.location,
            configuration=request.configuration,
            plant_id=PlantID(request.plant_id) if request.plant_id else None
        )
        
        logger.info("Sensor registered successfully", sensor_id=str(sensor.id))
        return SensorResponse.from_domain(sensor)
        
    except Exception as e:
        logger.error("Failed to register sensor", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to register sensor: {str(e)}"
        )


@router.get("/sensors", response_model=SensorListResponse)
async def list_sensors(
    pagination: PaginationParams = Depends(),
    sensor_type: Optional[str] = Query(None, description="Filter by sensor type"),
    plant_id: Optional[UUID] = Query(None, description="Filter by plant"),
    active_only: bool = Query(True, description="Show only active sensors"),
    sensor_repo: SensorRepository = Depends(get_sensor_repository)
) -> SensorListResponse:
    """
    List all sensors with optional filtering.
    
    Args:
        pagination: Pagination parameters
        sensor_type: Optional sensor type filter
        plant_id: Optional plant filter
        active_only: Whether to show only active sensors
        sensor_repo: Sensor repository dependency
        
    Returns:
        Paginated list of sensors
    """
    logger.info(
        "Listing sensors",
        filters={
            "sensor_type": sensor_type,
            "plant_id": str(plant_id) if plant_id else None,
            "active_only": active_only
        }
    )
    
    use_case = ListSensorsUseCase(sensor_repo)
    sensors, total = await use_case.execute(
        offset=pagination.offset,
        limit=pagination.limit,
        sensor_type=SensorType(sensor_type) if sensor_type else None,
        plant_id=PlantID(plant_id) if plant_id else None,
        active_only=active_only
    )
    
    return SensorListResponse(
        items=[SensorResponse.from_domain(sensor) for sensor in sensors],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit
    )


@router.get("/sensors/{sensor_id}/readings", response_model=List[SensorReadingResponse])
async def get_sensor_readings(
    sensor_id: UUID,
    start_time: Optional[datetime] = Query(None, description="Start time for readings"),
    end_time: Optional[datetime] = Query(None, description="End time for readings"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of readings"),
    sensor_repo: SensorRepository = Depends(get_sensor_repository)
) -> List[SensorReadingResponse]:
    """
    Get sensor readings for a specific time range.
    
    Args:
        sensor_id: UUID of the sensor
        start_time: Optional start time filter
        end_time: Optional end time filter
        limit: Maximum number of readings to return
        sensor_repo: Sensor repository dependency
        
    Returns:
        List of sensor readings
        
    Raises:
        404: Sensor not found
    """
    logger.info(
        "Fetching sensor readings",
        sensor_id=str(sensor_id),
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )
    
    try:
        use_case = GetSensorReadingsUseCase(sensor_repo)
        readings = await use_case.execute(
            sensor_id=SensorID(sensor_id),
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return [
            SensorReadingResponse(
                sensor_id=reading.sensor_id,
                timestamp=reading.timestamp,
                value=reading.value,
                unit=reading.unit,
                quality=reading.quality
            )
            for reading in readings
        ]
        
    except SensorNotFoundError as e:
        logger.warning("Sensor not found", sensor_id=str(sensor_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.put("/sensors/{sensor_id}", response_model=SensorResponse)
async def update_sensor(
    sensor_id: UUID,
    request: SensorUpdateRequest,
    sensor_repo: SensorRepository = Depends(get_sensor_repository)
) -> SensorResponse:
    """
    Update sensor configuration.
    
    Args:
        sensor_id: UUID of the sensor to update
        request: Update data
        sensor_repo: Sensor repository dependency
        
    Returns:
        Updated sensor details
        
    Raises:
        404: Sensor not found
        400: Invalid update data
    """
    logger.info("Updating sensor", sensor_id=str(sensor_id))
    
    try:
        use_case = UpdateSensorUseCase(sensor_repo)
        
        # Build update data
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.location is not None:
            update_data["location"] = request.location
        if request.configuration is not None:
            update_data["configuration"] = request.configuration
        if request.active is not None:
            update_data["active"] = request.active
            
        sensor = await use_case.execute(SensorID(sensor_id), **update_data)
        
        logger.info("Sensor updated successfully", sensor_id=str(sensor_id))
        return SensorResponse.from_domain(sensor)
        
    except SensorNotFoundError as e:
        logger.warning("Sensor not found for update", sensor_id=str(sensor_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


# Watering Schedule Management Endpoints

@router.post("/schedules", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED)
async def create_schedule(
    request: ScheduleCreateRequest,
    schedule_repo: WateringScheduleRepository = Depends(get_schedule_repository)
) -> ScheduleResponse:
    """
    Create a new watering schedule.
    
    Args:
        request: Schedule creation data
        schedule_repo: Schedule repository dependency
        
    Returns:
        Created schedule details
        
    Raises:
        400: Invalid schedule configuration
    """
    logger.info(
        "Creating watering schedule",
        plant_id=str(request.plant_id),
        schedule_type=request.schedule_type
    )
    
    try:
        use_case = CreateScheduleUseCase(schedule_repo)
        schedule = await use_case.execute(
            plant_id=PlantID(request.plant_id),
            schedule_type=ScheduleType(request.schedule_type),
            start_date=request.start_date,
            time_of_day=request.time_of_day,
            interval_days=request.interval_days,
            duration_seconds=request.duration_seconds,
            amount_ml=request.amount_ml,
            enabled=request.enabled
        )
        
        logger.info("Schedule created successfully", schedule_id=str(schedule.id))
        return ScheduleResponse.from_domain(schedule)
        
    except Exception as e:
        logger.error("Failed to create schedule", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create schedule: {str(e)}"
        )


@router.get("/schedules", response_model=ScheduleListResponse)
async def list_schedules(
    pagination: PaginationParams = Depends(),
    plant_id: Optional[UUID] = Query(None, description="Filter by plant"),
    enabled_only: bool = Query(True, description="Show only enabled schedules"),
    schedule_repo: WateringScheduleRepository = Depends(get_schedule_repository)
) -> ScheduleListResponse:
    """
    List watering schedules with optional filtering.
    
    Args:
        pagination: Pagination parameters
        plant_id: Optional plant filter
        enabled_only: Whether to show only enabled schedules
        schedule_repo: Schedule repository dependency
        
    Returns:
        Paginated list of schedules
    """
    logger.info(
        "Listing schedules",
        filters={
            "plant_id": str(plant_id) if plant_id else None,
            "enabled_only": enabled_only
        }
    )
    
    use_case = ListSchedulesUseCase(schedule_repo)
    schedules, total = await use_case.execute(
        offset=pagination.offset,
        limit=pagination.limit,
        plant_id=PlantID(plant_id) if plant_id else None,
        enabled_only=enabled_only
    )
    
    return ScheduleListResponse(
        items=[ScheduleResponse.from_domain(schedule) for schedule in schedules],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit
    )


@router.put("/schedules/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    schedule_id: UUID,
    request: ScheduleUpdateRequest,
    schedule_repo: WateringScheduleRepository = Depends(get_schedule_repository)
) -> ScheduleResponse:
    """
    Update watering schedule.
    
    Args:
        schedule_id: UUID of the schedule to update
        request: Update data
        schedule_repo: Schedule repository dependency
        
    Returns:
        Updated schedule details
        
    Raises:
        404: Schedule not found
        400: Invalid update data
    """
    logger.info("Updating schedule", schedule_id=str(schedule_id))
    
    try:
        use_case = UpdateScheduleUseCase(schedule_repo)
        
        # Build update data
        update_data = {}
        if request.time_of_day is not None:
            update_data["time_of_day"] = request.time_of_day
        if request.interval_days is not None:
            update_data["interval_days"] = request.interval_days
        if request.duration_seconds is not None:
            update_data["duration_seconds"] = request.duration_seconds
        if request.amount_ml is not None:
            update_data["amount_ml"] = request.amount_ml
        if request.enabled is not None:
            update_data["enabled"] = request.enabled
            
        schedule = await use_case.execute(schedule_id, **update_data)
        
        logger.info("Schedule updated successfully", schedule_id=str(schedule_id))
        return ScheduleResponse.from_domain(schedule)
        
    except ScheduleNotFoundError as e:
        logger.warning("Schedule not found for update", schedule_id=str(schedule_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post("/watering/manual", status_code=status.HTTP_202_ACCEPTED)
async def trigger_manual_watering(
    request: ManualWateringRequest,
    background_tasks: BackgroundTasks,
    schedule_repo: WateringScheduleRepository = Depends(get_schedule_repository)
) -> Dict[str, str]:
    """
    Trigger manual watering for a plant.
    
    Args:
        request: Manual watering parameters
        background_tasks: Background task manager
        schedule_repo: Schedule repository dependency
        
    Returns:
        Confirmation message
        
    Raises:
        404: Plant not found
        400: Invalid watering parameters
    """
    logger.info(
        "Triggering manual watering",
        plant_id=str(request.plant_id),
        duration_seconds=request.duration_seconds,
        amount_ml=request.amount_ml
    )
    
    try:
        use_case = TriggerManualWateringUseCase(schedule_repo)
        
        # Add to background tasks for async execution
        background_tasks.add_task(
            use_case.execute,
            plant_id=PlantID(request.plant_id),
            duration_seconds=request.duration_seconds,
            amount_ml=request.amount_ml
        )
        
        return {
            "status": "accepted",
            "message": f"Manual watering triggered for plant {request.plant_id}"
        }
        
    except Exception as e:
        logger.error("Failed to trigger manual watering", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to trigger watering: {str(e)}"
        )


# System Information Endpoints

@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info() -> SystemInfoResponse:
    """
    Get system information and configuration.
    
    Returns:
        System configuration and version information
    """
    logger.info("Fetching system information")
    
    return SystemInfoResponse(
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
        features={
            "timescaledb": True,
            "redis_cache": True,
            "prometheus_metrics": True,
            "websocket_support": True,
            "hardware_simulation": os.getenv("HARDWARE_SIMULATION", "true").lower() == "true"
        },
        limits={
            "max_plants": 100,
            "max_sensors_per_plant": 10,
            "max_schedules_per_plant": 5,
            "sensor_reading_retention_days": 90,
            "min_watering_interval_hours": 1
        },
        supported_sensors=[
            "moisture",
            "temperature",
            "humidity",
            "light",
            "ph",
            "nutrient"
        ]
    )


@router.post("/cache/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_cache(
    redis: aioredis.Redis = Depends(get_redis_client)
) -> None:
    """
    Clear all cached data.
    
    Args:
        redis: Redis client dependency
        
    Note:
        This is an administrative endpoint that should be protected
    """
    logger.warning("Clearing all cache data")
    
    try:
        await redis.flushdb()
        logger.info("Cache cleared successfully")
        
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )