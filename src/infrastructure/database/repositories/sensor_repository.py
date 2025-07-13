"""
Sensor repository implementation using SQLAlchemy.

This module provides the concrete implementation of the SensorRepository
interface using SQLAlchemy for database operations.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from src.core.domain.entities import Sensor
from src.core.domain.repositories import SensorRepository
from src.core.domain.value_objects import SensorID, PlantID, SensorType
from src.infrastructure.database.models import SensorModel
from src.infrastructure.database.repositories.base import BaseRepository
from src.shared.exceptions import SensorNotFoundError

logger = structlog.get_logger(__name__)


class SQLAlchemySensorRepository(BaseRepository[SensorModel, Sensor, SensorID], SensorRepository):
    """SQLAlchemy implementation of SensorRepository."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        super().__init__(session, SensorModel)
        
    def _to_entity(self, model: SensorModel) -> Sensor:
        """Convert database model to domain entity."""
        return Sensor(
            id=SensorID(model.id),
            sensor_type=SensorType(model.sensor_type),
            hardware_id=model.hardware_id,
            name=model.name,
            location=model.location,
            plant_id=PlantID(model.plant_id) if model.plant_id else None,
            configuration=model.configuration or {},
            calibration_data=model.calibration_data or {},
            last_reading_at=model.last_reading_at,
            status=model.status,
            active=model.active,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        
    def _to_model(self, entity: Sensor) -> SensorModel:
        """Convert domain entity to database model."""
        return SensorModel(
            id=entity.id.value,
            sensor_type=entity.sensor_type.value,
            hardware_id=entity.hardware_id,
            name=entity.name,
            location=entity.location,
            plant_id=entity.plant_id.value if entity.plant_id else None,
            configuration=entity.configuration,
            calibration_data=entity.calibration_data,
            last_reading_at=entity.last_reading_at,
            status=entity.status,
            active=entity.active
        )
        
    def _get_id_value(self, entity_id: SensorID) -> UUID:
        """Extract UUID from SensorID."""
        return entity_id.value
        
    async def get_by_id(self, sensor_id: SensorID) -> Optional[Sensor]:
        """Get sensor by ID."""
        logger.info("Fetching sensor by ID", sensor_id=str(sensor_id))
        
        sensor = await super().get_by_id(sensor_id)
        
        if not sensor:
            logger.warning("Sensor not found", sensor_id=str(sensor_id))
            
        return sensor
        
    async def get_by_hardware_id(self, hardware_id: str) -> Optional[Sensor]:
        """Get sensor by hardware ID."""
        logger.info("Fetching sensor by hardware ID", hardware_id=hardware_id)
        
        try:
            stmt = select(SensorModel).where(
                SensorModel.hardware_id == hardware_id
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                return self._to_entity(model)
                
            return None
            
        except Exception as e:
            logger.error(
                "Failed to fetch sensor by hardware ID",
                hardware_id=hardware_id,
                error=str(e)
            )
            raise
            
    async def get_by_plant(
        self,
        plant_id: PlantID,
        sensor_type: Optional[SensorType] = None,
        active_only: bool = True
    ) -> List[Sensor]:
        """Get sensors for a specific plant."""
        logger.info(
            "Fetching sensors by plant",
            plant_id=str(plant_id),
            sensor_type=sensor_type,
            active_only=active_only
        )
        
        try:
            stmt = select(SensorModel).where(
                SensorModel.plant_id == plant_id.value
            )
            
            if sensor_type:
                stmt = stmt.where(SensorModel.sensor_type == sensor_type.value)
                
            if active_only:
                stmt = stmt.where(SensorModel.active == True)
                
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._to_entity(model) for model in models]
            
        except Exception as e:
            logger.error(
                "Failed to fetch sensors by plant",
                plant_id=str(plant_id),
                error=str(e)
            )
            raise
            
    async def get_active_sensors(self, sensor_type: Optional[SensorType] = None) -> List[Sensor]:
        """Get all active sensors."""
        logger.info("Fetching active sensors", sensor_type=sensor_type)
        
        try:
            stmt = select(SensorModel).where(SensorModel.active == True)
            
            if sensor_type:
                stmt = stmt.where(SensorModel.sensor_type == sensor_type.value)
                
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._to_entity(model) for model in models]
            
        except Exception as e:
            logger.error("Failed to fetch active sensors", error=str(e))
            raise
            
    async def update_last_reading(self, sensor_id: SensorID, timestamp: datetime) -> None:
        """Update last reading timestamp for a sensor."""
        logger.info(
            "Updating last reading timestamp",
            sensor_id=str(sensor_id),
            timestamp=timestamp
        )
        
        try:
            stmt = (
                update(SensorModel)
                .where(SensorModel.id == sensor_id.value)
                .values(last_reading_at=timestamp, updated_at=datetime.utcnow())
            )
            
            await self.session.execute(stmt)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            logger.error(
                "Failed to update last reading timestamp",
                sensor_id=str(sensor_id),
                error=str(e)
            )
            raise
            
    async def update_status(self, sensor_id: SensorID, status: str) -> None:
        """Update sensor status."""
        logger.info(
            "Updating sensor status",
            sensor_id=str(sensor_id),
            status=status
        )
        
        try:
            stmt = (
                update(SensorModel)
                .where(SensorModel.id == sensor_id.value)
                .values(status=status, updated_at=datetime.utcnow())
            )
            
            await self.session.execute(stmt)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            logger.error(
                "Failed to update sensor status",
                sensor_id=str(sensor_id),
                error=str(e)
            )
            raise