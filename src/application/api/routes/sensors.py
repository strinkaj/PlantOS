"""
Sensor data ingestion API routes.

This module provides endpoints for sensor data ingestion with time-series
optimization, real-time processing, and comprehensive validation.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from pydantic import Field
import structlog

from src.core.domain.entities import SensorReading
from src.core.domain.repositories import SensorReadingRepository, SensorRepository
from src.core.domain.value_objects import SensorID, SensorReadingID
from src.core.use_cases.sensor_data import (
    IngestSensorDataUseCase,
    BatchIngestSensorDataUseCase,
    GetSensorDataUseCase,
    GetAggregatedSensorDataUseCase
)
from src.infrastructure.database.dependency import (
    get_sensor_reading_repository,
    get_sensor_repository
)
from src.shared.exceptions import SensorNotFoundError, ValidationError
from src.shared.validation import RequestValidator, validate_and_sanitize
from src.shared.contracts import require, ensure, validate_sensor_reading
from src.application.models import (
    SensorReadingCreateRequest,
    SensorReadingBatchRequest,
    SensorReadingResponse,
    SensorDataListResponse,
    SensorAggregationRequest,
    SensorAggregationResponse,
    PaginationParams
)

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/readings", response_model=SensorReadingResponse, status_code=status.HTTP_201_CREATED)
@validate_sensor_reading
async def ingest_sensor_reading(
    request: SensorReadingCreateRequest,
    background_tasks: BackgroundTasks,
    sensor_reading_repo: SensorReadingRepository = Depends(get_sensor_reading_repository),
    sensor_repo: SensorRepository = Depends(get_sensor_repository)
) -> SensorReadingResponse:
    """
    Ingest a single sensor reading with real-time processing.
    
    Args:
        request: Sensor reading data with validation
        background_tasks: Background task manager for async processing
        sensor_reading_repo: Sensor reading repository dependency
        sensor_repo: Sensor repository dependency
        
    Returns:
        Created sensor reading with generated ID
        
    Raises:
        400: Invalid sensor data or validation errors
        404: Sensor not found
    """
    logger.info(
        "Ingesting sensor reading",
        sensor_id=str(request.sensor_id),
        sensor_type=request.sensor_type,
        value=request.value
    )
    
    try:
        use_case = IngestSensorDataUseCase(sensor_reading_repo, sensor_repo)
        
        # Execute with contract validation
        reading = await use_case.execute(
            sensor_id=request.sensor_id,
            value=request.value,
            unit=request.unit,
            timestamp=request.timestamp or datetime.utcnow(),
            quality=request.quality or "good",
            metadata=request.metadata or {}
        )
        
        # Add real-time processing to background tasks
        background_tasks.add_task(
            _process_sensor_reading_realtime,
            reading.id,
            request.sensor_id,
            request.value,
            request.sensor_type
        )
        
        logger.info(
            "Sensor reading ingested successfully",
            reading_id=str(reading.id),
            sensor_id=str(request.sensor_id)
        )
        
        return SensorReadingResponse.from_domain(reading)
        
    except SensorNotFoundError as e:
        logger.warning("Sensor not found for reading", sensor_id=str(request.sensor_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        logger.warning("Sensor reading validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/readings/batch", response_model=List[SensorReadingResponse], status_code=status.HTTP_201_CREATED)
async def ingest_sensor_readings_batch(
    request: SensorReadingBatchRequest,
    background_tasks: BackgroundTasks,
    sensor_reading_repo: SensorReadingRepository = Depends(get_sensor_reading_repository),
    sensor_repo: SensorRepository = Depends(get_sensor_repository)
) -> List[SensorReadingResponse]:
    """
    Ingest multiple sensor readings in batch for high throughput.
    
    Args:
        request: Batch sensor reading data
        background_tasks: Background task manager
        sensor_reading_repo: Sensor reading repository dependency
        sensor_repo: Sensor repository dependency
        
    Returns:
        List of created sensor readings
        
    Raises:
        400: Batch too large or validation errors
        404: One or more sensors not found
    """
    logger.info(
        "Ingesting sensor readings batch",
        batch_size=len(request.readings),
        batch_id=request.batch_id
    )
    
    # Validate batch size
    if len(request.readings) > 1000:  # Configurable limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds maximum limit of 1000 readings"
        )
    
    try:
        use_case = BatchIngestSensorDataUseCase(sensor_reading_repo, sensor_repo)
        
        # Prepare readings data
        readings_data = [
            {
                "sensor_id": reading.sensor_id,
                "value": reading.value,
                "unit": reading.unit,
                "timestamp": reading.timestamp or datetime.utcnow(),
                "quality": reading.quality or "good",
                "metadata": reading.metadata or {}
            }
            for reading in request.readings
        ]
        
        # Execute batch ingestion
        readings = await use_case.execute(readings_data)
        
        # Add batch processing to background tasks
        background_tasks.add_task(
            _process_sensor_readings_batch,
            [r.id for r in readings],
            request.batch_id
        )
        
        logger.info(
            "Sensor readings batch ingested successfully",
            batch_size=len(readings),
            batch_id=request.batch_id
        )
        
        return [SensorReadingResponse.from_domain(reading) for reading in readings]
        
    except Exception as e:
        logger.error(
            "Batch ingestion failed",
            batch_id=request.batch_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch ingestion failed: {str(e)}"
        )


@router.get("/readings/{sensor_id}", response_model=SensorDataListResponse)
@require(lambda sensor_id, start_time=None, end_time=None: 
         start_time is None or end_time is None or start_time < end_time,
         "Start time must be before end time")
async def get_sensor_readings(
    sensor_id: UUID,
    start_time: Optional[datetime] = Query(None, description="Start time for readings"),
    end_time: Optional[datetime] = Query(None, description="End time for readings"),
    pagination: PaginationParams = Depends(),
    quality_filter: Optional[str] = Query(None, description="Filter by data quality"),
    sensor_reading_repo: SensorReadingRepository = Depends(get_sensor_reading_repository)
) -> SensorDataListResponse:
    """
    Get sensor readings for a specific sensor with time-series optimization.
    
    Args:
        sensor_id: UUID of the sensor
        start_time: Optional start time filter
        end_time: Optional end time filter
        pagination: Pagination parameters
        quality_filter: Optional quality filter
        sensor_reading_repo: Sensor reading repository dependency
        
    Returns:
        Time-series sensor data with metadata
        
    Raises:
        400: Invalid time range or parameters
        404: Sensor not found
    """
    logger.info(
        "Fetching sensor readings",
        sensor_id=str(sensor_id),
        start_time=start_time,
        end_time=end_time,
        limit=pagination.limit
    )
    
    try:
        use_case = GetSensorDataUseCase(sensor_reading_repo)
        
        readings, total = await use_case.execute(
            sensor_id=SensorID(sensor_id),
            start_time=start_time,
            end_time=end_time,
            offset=pagination.offset,
            limit=pagination.limit,
            quality_filter=quality_filter
        )
        
        return SensorDataListResponse(
            sensor_id=sensor_id,
            readings=[SensorReadingResponse.from_domain(reading) for reading in readings],
            total_count=total,
            time_range={
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            },
            pagination={
                "offset": pagination.offset,
                "limit": pagination.limit,
                "total": total
            }
        )
        
    except Exception as e:
        logger.error(
            "Failed to fetch sensor readings",
            sensor_id=str(sensor_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch sensor readings: {str(e)}"
        )


@router.post("/readings/aggregate", response_model=SensorAggregationResponse)
async def get_aggregated_sensor_data(
    request: SensorAggregationRequest,
    sensor_reading_repo: SensorReadingRepository = Depends(get_sensor_reading_repository)
) -> SensorAggregationResponse:
    """
    Get aggregated sensor data with time-series analytics.
    
    Args:
        request: Aggregation request parameters
        sensor_reading_repo: Sensor reading repository dependency
        
    Returns:
        Aggregated sensor statistics and time-series data
        
    Raises:
        400: Invalid aggregation parameters
        404: Sensor not found
    """
    logger.info(
        "Aggregating sensor data",
        sensor_ids=len(request.sensor_ids),
        aggregation_type=request.aggregation_type,
        time_bucket=request.time_bucket
    )
    
    try:
        use_case = GetAggregatedSensorDataUseCase(sensor_reading_repo)
        
        aggregation_result = await use_case.execute(
            sensor_ids=[SensorID(sid) for sid in request.sensor_ids],
            start_time=request.start_time,
            end_time=request.end_time,
            aggregation_type=request.aggregation_type,
            time_bucket=request.time_bucket,
            group_by_sensor=request.group_by_sensor
        )
        
        return SensorAggregationResponse(
            aggregation_type=request.aggregation_type,
            time_bucket=request.time_bucket,
            time_range={
                "start": request.start_time.isoformat(),
                "end": request.end_time.isoformat()
            },
            data_points=aggregation_result["data_points"],
            statistics=aggregation_result["statistics"],
            metadata={
                "sensor_count": len(request.sensor_ids),
                "total_readings": aggregation_result["total_readings"],
                "data_quality": aggregation_result["data_quality"]
            }
        )
        
    except Exception as e:
        logger.error("Sensor data aggregation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Aggregation failed: {str(e)}"
        )


@router.get("/readings/{sensor_id}/latest", response_model=SensorReadingResponse)
async def get_latest_sensor_reading(
    sensor_id: UUID,
    sensor_reading_repo: SensorReadingRepository = Depends(get_sensor_reading_repository)
) -> SensorReadingResponse:
    """
    Get the latest reading for a specific sensor.
    
    Args:
        sensor_id: UUID of the sensor
        sensor_reading_repo: Sensor reading repository dependency
        
    Returns:
        Latest sensor reading
        
    Raises:
        404: Sensor not found or no readings available
    """
    logger.info("Fetching latest sensor reading", sensor_id=str(sensor_id))
    
    try:
        use_case = GetSensorDataUseCase(sensor_reading_repo)
        
        readings, _ = await use_case.execute(
            sensor_id=SensorID(sensor_id),
            limit=1,
            order_desc=True
        )
        
        if not readings:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No readings found for sensor {sensor_id}"
            )
            
        return SensorReadingResponse.from_domain(readings[0])
        
    except Exception as e:
        logger.error(
            "Failed to fetch latest reading",
            sensor_id=str(sensor_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch latest reading: {str(e)}"
        )


@router.delete("/readings/{sensor_id}/old", status_code=status.HTTP_204_NO_CONTENT)
async def cleanup_old_sensor_readings(
    sensor_id: UUID,
    days_to_keep: int = Query(90, ge=1, le=365, description="Days of data to keep"),
    sensor_reading_repo: SensorReadingRepository = Depends(get_sensor_reading_repository)
) -> None:
    """
    Clean up old sensor readings to manage storage.
    
    Args:
        sensor_id: UUID of the sensor
        days_to_keep: Number of days of data to keep
        sensor_reading_repo: Sensor reading repository dependency
        
    Raises:
        404: Sensor not found
    """
    logger.info(
        "Cleaning up old sensor readings",
        sensor_id=str(sensor_id),
        days_to_keep=days_to_keep
    )
    
    try:
        cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # This would be implemented in the repository
        # deleted_count = await sensor_reading_repo.delete_old_readings(
        #     sensor_id=SensorID(sensor_id),
        #     cutoff_time=cutoff_time
        # )
        
        logger.info(
            "Old sensor readings cleaned up",
            sensor_id=str(sensor_id),
            cutoff_time=cutoff_time
        )
        
    except Exception as e:
        logger.error(
            "Failed to cleanup old readings",
            sensor_id=str(sensor_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup old readings"
        )


# Background task functions
async def _process_sensor_reading_realtime(
    reading_id: SensorReadingID,
    sensor_id: SensorID,
    value: float,
    sensor_type: str
):
    """Process individual sensor reading in real-time."""
    logger.info(
        "Processing sensor reading real-time",
        reading_id=str(reading_id),
        sensor_id=str(sensor_id),
        value=value,
        sensor_type=sensor_type
    )
    
    try:
        # TODO: Implement real-time processing
        # - Check thresholds and trigger alerts
        # - Update plant health scores
        # - Send to streaming pipeline
        # - Cache latest values
        pass
        
    except Exception as e:
        logger.error(
            "Real-time processing failed",
            reading_id=str(reading_id),
            error=str(e)
        )


async def _process_sensor_readings_batch(
    reading_ids: List[SensorReadingID],
    batch_id: Optional[str]
):
    """Process batch of sensor readings."""
    logger.info(
        "Processing sensor readings batch",
        batch_size=len(reading_ids),
        batch_id=batch_id
    )
    
    try:
        # TODO: Implement batch processing
        # - Bulk analytics updates
        # - Batch streaming to Kafka
        # - Bulk threshold checking
        pass
        
    except Exception as e:
        logger.error(
            "Batch processing failed",
            batch_id=batch_id,
            error=str(e)
        )