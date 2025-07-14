"""
Sensor data management use cases.

These use cases handle sensor data ingestion, retrieval, and analytics
with time-series optimization and real-time processing capabilities.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

import structlog

from src.core.domain.entities import SensorReading, Sensor
from src.core.domain.repositories import SensorReadingRepository, SensorRepository
from src.shared.types import SensorID, SensorReadingID, SensorType
from src.shared.exceptions import (
    SensorNotFoundError,
    ValidationError,
    DataQualityError
)
from src.shared.contracts import require, ensure

logger = structlog.get_logger(__name__)


class IngestSensorDataUseCase:
    """Use case for ingesting individual sensor readings."""
    
    def __init__(
        self,
        sensor_reading_repo: SensorReadingRepository,
        sensor_repo: SensorRepository
    ):
        self.sensor_reading_repo = sensor_reading_repo
        self.sensor_repo = sensor_repo
        
    @require(lambda sensor_id, value: sensor_id is not None, "Sensor ID cannot be None")
    @require(lambda sensor_id, value: isinstance(value, (int, float)), "Value must be numeric")
    async def execute(
        self,
        sensor_id: UUID,
        value: float,
        unit: str = "unknown",
        timestamp: Optional[datetime] = None,
        quality: str = "good",
        metadata: Optional[Dict[str, Any]] = None
    ) -> SensorReading:
        """
        Ingest a single sensor reading.
        
        Args:
            sensor_id: ID of the sensor
            value: Sensor reading value
            unit: Unit of measurement
            timestamp: Reading timestamp (defaults to now)
            quality: Data quality indicator
            metadata: Additional metadata
            
        Returns:
            Created sensor reading entity
            
        Raises:
            SensorNotFoundError: If sensor doesn't exist
            ValidationError: If data validation fails
        """
        logger.info(
            "Ingesting sensor reading",
            sensor_id=str(sensor_id),
            value=value,
            quality=quality
        )
        
        # Validate sensor exists and is active
        sensor = await self.sensor_repo.get_by_id(SensorID(sensor_id))
        if not sensor:
            raise SensorNotFoundError(f"Sensor with ID {sensor_id} not found")
            
        if not sensor.active:
            raise ValidationError(f"Sensor {sensor_id} is not active")
            
        # Validate reading based on sensor type
        self._validate_sensor_reading(sensor, value)
        
        # Create sensor reading
        reading = SensorReading(
            id=SensorReadingID(uuid4()),
            sensor_id=SensorID(sensor_id),
            timestamp=timestamp or datetime.utcnow(),
            value=value,
            unit=unit,
            quality=quality,
            metadata=metadata or {}
        )
        
        # Save reading
        created_reading = await self.sensor_reading_repo.create(reading)
        
        # Update sensor's last reading timestamp
        sensor.last_reading_at = reading.timestamp
        await self.sensor_repo.update(sensor)
        
        logger.info(
            "Sensor reading ingested successfully",
            reading_id=str(created_reading.id),
            sensor_id=str(sensor_id)
        )
        
        return created_reading
        
    def _validate_sensor_reading(self, sensor: Sensor, value: float) -> None:
        """Validate sensor reading against sensor type constraints."""
        sensor_type = sensor.sensor_type.value
        
        if sensor_type == "moisture" and (value < 0 or value > 100):
            raise ValidationError("Moisture readings must be between 0 and 100")
        elif sensor_type == "temperature" and (value < -50 or value > 80):
            raise ValidationError("Temperature readings must be between -50°C and 80°C")
        elif sensor_type == "humidity" and (value < 0 or value > 100):
            raise ValidationError("Humidity readings must be between 0 and 100")
        elif sensor_type == "ph" and (value < 0 or value > 14):
            raise ValidationError("pH readings must be between 0 and 14")
        elif sensor_type == "light" and value < 0:
            raise ValidationError("Light readings must be non-negative")


class BatchIngestSensorDataUseCase:
    """Use case for batch ingestion of sensor readings."""
    
    def __init__(
        self,
        sensor_reading_repo: SensorReadingRepository,
        sensor_repo: SensorRepository
    ):
        self.sensor_reading_repo = sensor_reading_repo
        self.sensor_repo = sensor_repo
        
    @require(lambda readings_data: isinstance(readings_data, list), "Readings data must be a list")
    @require(lambda readings_data: len(readings_data) > 0, "Batch cannot be empty")
    @require(lambda readings_data: len(readings_data) <= 1000, "Batch size cannot exceed 1000")
    async def execute(self, readings_data: List[Dict[str, Any]]) -> List[SensorReading]:
        """
        Ingest multiple sensor readings in batch.
        
        Args:
            readings_data: List of reading data dictionaries
            
        Returns:
            List of created sensor reading entities
            
        Raises:
            ValidationError: If batch validation fails
        """
        logger.info("Ingesting sensor readings batch", batch_size=len(readings_data))
        
        # Validate all sensor IDs exist and are active
        sensor_ids = {UUID(reading["sensor_id"]) for reading in readings_data}
        sensors = {}
        
        for sensor_id in sensor_ids:
            sensor = await self.sensor_repo.get_by_id(SensorID(sensor_id))
            if not sensor:
                raise SensorNotFoundError(f"Sensor with ID {sensor_id} not found")
            if not sensor.active:
                raise ValidationError(f"Sensor {sensor_id} is not active")
            sensors[sensor_id] = sensor
            
        # Create reading entities
        readings = []
        for reading_data in readings_data:
            sensor_id = UUID(reading_data["sensor_id"])
            sensor = sensors[sensor_id]
            
            # Validate reading
            self._validate_sensor_reading(sensor, reading_data["value"])
            
            reading = SensorReading(
                id=SensorReadingID(uuid4()),
                sensor_id=SensorID(sensor_id),
                timestamp=reading_data.get("timestamp", datetime.utcnow()),
                value=reading_data["value"],
                unit=reading_data.get("unit", "unknown"),
                quality=reading_data.get("quality", "good"),
                metadata=reading_data.get("metadata", {})
            )
            readings.append(reading)
            
        # Batch save readings
        created_readings = await self.sensor_reading_repo.create_batch(readings)
        
        # Update sensor last reading timestamps
        for sensor_id, sensor in sensors.items():
            latest_reading = max(
                (r for r in created_readings if r.sensor_id.value == sensor_id),
                key=lambda r: r.timestamp,
                default=None
            )
            if latest_reading:
                sensor.last_reading_at = latest_reading.timestamp
                await self.sensor_repo.update(sensor)
                
        logger.info(
            "Sensor readings batch ingested successfully",
            batch_size=len(created_readings)
        )
        
        return created_readings
        
    def _validate_sensor_reading(self, sensor: Sensor, value: float) -> None:
        """Validate sensor reading against sensor type constraints."""
        sensor_type = sensor.sensor_type.value
        
        if sensor_type == "moisture" and (value < 0 or value > 100):
            raise ValidationError("Moisture readings must be between 0 and 100")
        elif sensor_type == "temperature" and (value < -50 or value > 80):
            raise ValidationError("Temperature readings must be between -50°C and 80°C")
        elif sensor_type == "humidity" and (value < 0 or value > 100):
            raise ValidationError("Humidity readings must be between 0 and 100")
        elif sensor_type == "ph" and (value < 0 or value > 14):
            raise ValidationError("pH readings must be between 0 and 14")
        elif sensor_type == "light" and value < 0:
            raise ValidationError("Light readings must be non-negative")


class GetSensorDataUseCase:
    """Use case for retrieving sensor readings with time filtering."""
    
    def __init__(self, sensor_reading_repo: SensorReadingRepository):
        self.sensor_reading_repo = sensor_reading_repo
        
    @require(lambda sensor_id: sensor_id is not None, "Sensor ID cannot be None")
    @require(lambda sensor_id, start_time=None, end_time=None: 
             start_time is None or end_time is None or start_time < end_time,
             "Start time must be before end time")
    async def execute(
        self,
        sensor_id: SensorID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
        quality_filter: Optional[str] = None,
        order_desc: bool = False
    ) -> tuple[List[SensorReading], int]:
        """
        Get sensor readings with time-series filtering.
        
        Args:
            sensor_id: ID of the sensor
            start_time: Optional start time filter
            end_time: Optional end time filter
            offset: Pagination offset
            limit: Maximum number of readings
            quality_filter: Optional quality filter
            order_desc: Order by timestamp descending
            
        Returns:
            Tuple of (readings list, total count)
        """
        logger.info(
            "Retrieving sensor readings",
            sensor_id=str(sensor_id.value),
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Get readings from repository
        readings = await self.sensor_reading_repo.get_by_sensor_id(
            sensor_id=sensor_id,
            start_time=start_time,
            end_time=end_time,
            quality_filter=quality_filter
        )
        
        # Apply ordering
        if order_desc:
            readings.sort(key=lambda r: r.timestamp, reverse=True)
        else:
            readings.sort(key=lambda r: r.timestamp)
            
        # Apply pagination
        total_count = len(readings)
        paginated_readings = readings[offset:offset + limit]
        
        logger.info(
            "Sensor readings retrieved successfully",
            sensor_id=str(sensor_id.value),
            returned_count=len(paginated_readings),
            total_count=total_count
        )
        
        return paginated_readings, total_count


class GetAggregatedSensorDataUseCase:
    """Use case for generating aggregated sensor analytics."""
    
    def __init__(self, sensor_reading_repo: SensorReadingRepository):
        self.sensor_reading_repo = sensor_reading_repo
        
    @require(lambda sensor_ids, start_time, end_time: len(sensor_ids) > 0, "Must specify at least one sensor")
    @require(lambda sensor_ids, start_time, end_time: start_time < end_time, "Start time must be before end time")
    async def execute(
        self,
        sensor_ids: List[SensorID],
        start_time: datetime,
        end_time: datetime,
        aggregation_type: str = "avg",
        time_bucket: str = "1hour",
        group_by_sensor: bool = False
    ) -> Dict[str, Any]:
        """
        Generate aggregated sensor data analytics.
        
        Args:
            sensor_ids: List of sensor IDs to aggregate
            start_time: Start of time range
            end_time: End of time range
            aggregation_type: Type of aggregation (avg, min, max, sum, count)
            time_bucket: Time bucket size for grouping
            group_by_sensor: Whether to group results by sensor
            
        Returns:
            Dictionary with aggregation results
        """
        logger.info(
            "Aggregating sensor data",
            sensor_count=len(sensor_ids),
            aggregation_type=aggregation_type,
            time_bucket=time_bucket
        )
        
        # Get all readings for the time period
        all_readings = []
        for sensor_id in sensor_ids:
            readings = await self.sensor_reading_repo.get_by_sensor_id(
                sensor_id=sensor_id,
                start_time=start_time,
                end_time=end_time
            )
            all_readings.extend(readings)
            
        if not all_readings:
            return {
                "data_points": [],
                "statistics": {},
                "total_readings": 0,
                "data_quality": {"good": 0, "uncertain": 0, "bad": 0}
            }
            
        # Parse time bucket
        bucket_seconds = self._parse_time_bucket(time_bucket)
        
        # Group readings by time buckets
        buckets = {}
        for reading in all_readings:
            bucket_time = self._get_bucket_time(reading.timestamp, bucket_seconds)
            bucket_key = bucket_time.isoformat()
            
            if bucket_key not in buckets:
                buckets[bucket_key] = {
                    "timestamp": bucket_time,
                    "values": [],
                    "sensor_data": {} if group_by_sensor else None
                }
                
            buckets[bucket_key]["values"].append(reading.value)
            
            if group_by_sensor:
                sensor_key = str(reading.sensor_id.value)
                if sensor_key not in buckets[bucket_key]["sensor_data"]:
                    buckets[bucket_key]["sensor_data"][sensor_key] = []
                buckets[bucket_key]["sensor_data"][sensor_key].append(reading.value)
                
        # Calculate aggregations
        data_points = []
        for bucket_key in sorted(buckets.keys()):
            bucket = buckets[bucket_key]
            values = bucket["values"]
            
            aggregated_value = self._calculate_aggregation(values, aggregation_type)
            
            data_point = {
                "timestamp": bucket["timestamp"].isoformat(),
                "value": aggregated_value,
                "count": len(values)
            }
            
            if group_by_sensor and bucket["sensor_data"]:
                data_point["by_sensor"] = {}
                for sensor_key, sensor_values in bucket["sensor_data"].items():
                    data_point["by_sensor"][sensor_key] = self._calculate_aggregation(
                        sensor_values, aggregation_type
                    )
                    
            data_points.append(data_point)
            
        # Calculate overall statistics
        all_values = [reading.value for reading in all_readings]
        quality_counts = {"good": 0, "uncertain": 0, "bad": 0, "unknown": 0}
        for reading in all_readings:
            quality_counts[reading.quality] = quality_counts.get(reading.quality, 0) + 1
            
        statistics = {
            "total_readings": len(all_readings),
            "min_value": min(all_values) if all_values else None,
            "max_value": max(all_values) if all_values else None,
            "avg_value": sum(all_values) / len(all_values) if all_values else None,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_hours": (end_time - start_time).total_seconds() / 3600
            }
        }
        
        result = {
            "data_points": data_points,
            "statistics": statistics,
            "total_readings": len(all_readings),
            "data_quality": quality_counts
        }
        
        logger.info(
            "Sensor data aggregated successfully",
            data_points_count=len(data_points),
            total_readings=len(all_readings)
        )
        
        return result
        
    def _parse_time_bucket(self, time_bucket: str) -> int:
        """Parse time bucket string to seconds."""
        if time_bucket == "1min":
            return 60
        elif time_bucket == "5min":
            return 300
        elif time_bucket == "15min":
            return 900
        elif time_bucket == "30min":
            return 1800
        elif time_bucket == "1hour":
            return 3600
        elif time_bucket == "6hour":
            return 21600
        elif time_bucket == "12hour":
            return 43200
        elif time_bucket == "1day":
            return 86400
        else:
            return 3600  # Default to 1 hour
            
    def _get_bucket_time(self, timestamp: datetime, bucket_seconds: int) -> datetime:
        """Get the bucket time for a given timestamp."""
        epoch = timestamp.timestamp()
        bucket_epoch = (epoch // bucket_seconds) * bucket_seconds
        return datetime.fromtimestamp(bucket_epoch)
        
    def _calculate_aggregation(self, values: List[float], aggregation_type: str) -> float:
        """Calculate aggregation for a list of values."""
        if not values:
            return 0.0
            
        if aggregation_type == "avg":
            return sum(values) / len(values)
        elif aggregation_type == "min":
            return min(values)
        elif aggregation_type == "max":
            return max(values)
        elif aggregation_type == "sum":
            return sum(values)
        elif aggregation_type == "count":
            return float(len(values))
        elif aggregation_type == "stddev":
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return variance ** 0.5
        else:
            return sum(values) / len(values)  # Default to average