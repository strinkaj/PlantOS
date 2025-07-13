"""
Database dependency injection.

This module provides FastAPI dependency injection for database sessions
and repository instances with proper lifecycle management.
"""

from typing import AsyncGenerator
import os

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
import aioredis
import structlog

from src.core.domain.repositories import (
    PlantRepository,
    PlantSpeciesRepository,
    SensorRepository,
    SensorReadingRepository,
    CareEventRepository,
    ActuatorRepository,
    PlantHealthRepository,
    WateringScheduleRepository
)
from src.infrastructure.database.config import get_database_manager
from src.infrastructure.database.repositories.plant_repository import SQLAlchemyPlantRepository
from src.infrastructure.database.repositories.plant_species_repository import SQLAlchemyPlantSpeciesRepository
from src.infrastructure.database.repositories.sensor_repository import SQLAlchemySensorRepository

logger = structlog.get_logger(__name__)


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session for dependency injection.
    
    Yields:
        AsyncSession for database operations
    """
    db_manager = get_database_manager()
    
    if not db_manager:
        raise RuntimeError("Database not initialized")
        
    async with db_manager.session() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_redis_client() -> aioredis.Redis:
    """
    Get Redis client for dependency injection.
    
    Returns:
        Redis client instance
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    redis = aioredis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30
    )
    
    return redis


# Repository Dependencies

def get_plant_repository(
    session: AsyncSession = Depends(get_database_session)
) -> PlantRepository:
    """Get plant repository instance."""
    return SQLAlchemyPlantRepository(session)


def get_plant_species_repository(
    session: AsyncSession = Depends(get_database_session)
) -> PlantSpeciesRepository:
    """Get plant species repository instance."""
    return SQLAlchemyPlantSpeciesRepository(session)


def get_sensor_repository(
    session: AsyncSession = Depends(get_database_session)
) -> SensorRepository:
    """Get sensor repository instance."""
    return SQLAlchemySensorRepository(session)


# TODO: Implement these repositories
def get_sensor_reading_repository(
    session: AsyncSession = Depends(get_database_session)
) -> SensorReadingRepository:
    """Get sensor reading repository instance."""
    # TODO: Implement SQLAlchemySensorReadingRepository
    raise NotImplementedError("SensorReadingRepository not implemented yet")


def get_care_event_repository(
    session: AsyncSession = Depends(get_database_session)
) -> CareEventRepository:
    """Get care event repository instance."""
    # TODO: Implement SQLAlchemyCareEventRepository
    raise NotImplementedError("CareEventRepository not implemented yet")


def get_actuator_repository(
    session: AsyncSession = Depends(get_database_session)
) -> ActuatorRepository:
    """Get actuator repository instance."""
    # TODO: Implement SQLAlchemyActuatorRepository
    raise NotImplementedError("ActuatorRepository not implemented yet")


def get_plant_health_repository(
    session: AsyncSession = Depends(get_database_session)
) -> PlantHealthRepository:
    """Get plant health repository instance."""
    # TODO: Implement SQLAlchemyPlantHealthRepository
    raise NotImplementedError("PlantHealthRepository not implemented yet")


def get_schedule_repository(
    session: AsyncSession = Depends(get_database_session)
) -> WateringScheduleRepository:
    """Get watering schedule repository instance."""
    # TODO: Implement SQLAlchemyWateringScheduleRepository
    raise NotImplementedError("WateringScheduleRepository not implemented yet")


# Use Case Dependencies

def get_database_health_check(
    session: AsyncSession = Depends(get_database_session)
) -> AsyncSession:
    """Get database session for health checks."""
    return session