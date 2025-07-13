"""
SQLAlchemy implementation of PlantRepository.

This module provides concrete repository implementation for Plant entities
using async SQLAlchemy with proper error handling and performance optimization.
"""

import logging
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update, delete, func, or_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.domain.entities import Plant, PlantStatus
from src.core.domain.repositories import PlantRepository
from src.infrastructure.database.models import PlantModel, PlantSpeciesModel
from src.shared.types import PlantID, SpeciesID
from src.shared.exceptions import DatabaseError, PlantNotFoundError, DuplicateResourceError

logger = logging.getLogger(__name__)


class SQLAlchemyPlantRepository(PlantRepository):
    """SQLAlchemy implementation of PlantRepository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, plant: Plant) -> Plant:
        """Create a new plant in the database."""
        try:
            # Check for duplicate name
            existing = await self._get_by_name_internal(plant.name)
            if existing:
                raise DuplicateResourceError("plant", plant.name)
            
            # Create database model
            plant_model = PlantModel(
                id=plant.id,
                name=plant.name,
                species_id=plant.species_id,
                location=plant.location,
                status=plant.status.value,
                health_score=plant.health_score,
                last_watered_at=plant.last_watered_at,
                last_fertilized_at=plant.last_fertilized_at,
                notes=plant.notes,
                metadata=plant.metadata,
                created_at=plant.created_at,
                updated_at=plant.updated_at
            )
            
            self.session.add(plant_model)
            await self.session.flush()  # Get the ID without committing
            
            logger.info(f"Created plant: {plant.name} (ID: {plant.id})")
            return plant
            
        except DuplicateResourceError:
            raise
        except Exception as e:
            logger.error(f"Failed to create plant {plant.name}: {e}")
            raise DatabaseError(f"Failed to create plant: {e}")
    
    async def get_by_id(self, plant_id: PlantID) -> Optional[Plant]:
        """Retrieve a plant by its ID."""
        try:
            stmt = (
                select(PlantModel)
                .options(selectinload(PlantModel.species))
                .where(PlantModel.id == plant_id)
            )
            result = await self.session.execute(stmt)
            plant_model = result.scalar_one_or_none()
            
            if plant_model:
                return self._model_to_entity(plant_model)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get plant by ID {plant_id}: {e}")
            raise DatabaseError(f"Failed to retrieve plant: {e}")
    
    async def get_by_name(self, name: str) -> Optional[Plant]:
        """Retrieve a plant by its name."""
        try:
            plant_model = await self._get_by_name_internal(name)
            if plant_model:
                return self._model_to_entity(plant_model)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get plant by name {name}: {e}")
            raise DatabaseError(f"Failed to retrieve plant: {e}")
    
    async def _get_by_name_internal(self, name: str) -> Optional[PlantModel]:
        """Internal method to get plant model by name."""
        stmt = (
            select(PlantModel)
            .options(selectinload(PlantModel.species))
            .where(PlantModel.name == name)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all(self, status: Optional[PlantStatus] = None) -> List[Plant]:
        """Retrieve all plants, optionally filtered by status."""
        try:
            stmt = (
                select(PlantModel)
                .options(selectinload(PlantModel.species))
                .order_by(PlantModel.created_at.desc())
            )
            
            if status:
                stmt = stmt.where(PlantModel.status == status.value)
            
            result = await self.session.execute(stmt)
            plant_models = result.scalars().all()
            
            return [self._model_to_entity(model) for model in plant_models]
            
        except Exception as e:
            logger.error(f"Failed to get all plants: {e}")
            raise DatabaseError(f"Failed to retrieve plants: {e}")
    
    async def get_by_species(self, species_id: SpeciesID) -> List[Plant]:
        """Retrieve all plants of a specific species."""
        try:
            stmt = (
                select(PlantModel)
                .options(selectinload(PlantModel.species))
                .where(PlantModel.species_id == species_id)
                .order_by(PlantModel.created_at.desc())
            )
            
            result = await self.session.execute(stmt)
            plant_models = result.scalars().all()
            
            return [self._model_to_entity(model) for model in plant_models]
            
        except Exception as e:
            logger.error(f"Failed to get plants by species {species_id}: {e}")
            raise DatabaseError(f"Failed to retrieve plants by species: {e}")
    
    async def update(self, plant: Plant) -> Plant:
        """Update an existing plant."""
        try:
            # Check if plant exists
            existing = await self.get_by_id(plant.id)
            if not existing:
                raise PlantNotFoundError(plant.id)
            
            # Check for name conflicts (excluding current plant)
            if plant.name != existing.name:
                name_conflict = await self._get_by_name_internal(plant.name)
                if name_conflict and name_conflict.id != plant.id:
                    raise DuplicateResourceError("plant", plant.name)
            
            # Update the plant
            stmt = (
                update(PlantModel)
                .where(PlantModel.id == plant.id)
                .values(
                    name=plant.name,
                    species_id=plant.species_id,
                    location=plant.location,
                    status=plant.status.value,
                    health_score=plant.health_score,
                    last_watered_at=plant.last_watered_at,
                    last_fertilized_at=plant.last_fertilized_at,
                    notes=plant.notes,
                    metadata=plant.metadata,
                    updated_at=datetime.utcnow()
                )
            )
            
            await self.session.execute(stmt)
            
            logger.info(f"Updated plant: {plant.name} (ID: {plant.id})")
            return plant
            
        except (PlantNotFoundError, DuplicateResourceError):
            raise
        except Exception as e:
            logger.error(f"Failed to update plant {plant.id}: {e}")
            raise DatabaseError(f"Failed to update plant: {e}")
    
    async def delete(self, plant_id: PlantID) -> bool:
        """Delete a plant by ID."""
        try:
            # Check if plant exists
            existing = await self.get_by_id(plant_id)
            if not existing:
                return False
            
            stmt = delete(PlantModel).where(PlantModel.id == plant_id)
            result = await self.session.execute(stmt)
            
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted plant: {plant_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete plant {plant_id}: {e}")
            raise DatabaseError(f"Failed to delete plant: {e}")
    
    async def count(self, status: Optional[PlantStatus] = None) -> int:
        """Count total plants, optionally filtered by status."""
        try:
            stmt = select(func.count(PlantModel.id))
            
            if status:
                stmt = stmt.where(PlantModel.status == status.value)
            
            result = await self.session.execute(stmt)
            return result.scalar() or 0
            
        except Exception as e:
            logger.error(f"Failed to count plants: {e}")
            raise DatabaseError(f"Failed to count plants: {e}")
    
    async def search(self, query: str) -> List[Plant]:
        """Search plants by name or notes."""
        try:
            search_term = f"%{query.lower()}%"
            
            stmt = (
                select(PlantModel)
                .options(selectinload(PlantModel.species))
                .where(
                    or_(
                        func.lower(PlantModel.name).like(search_term),
                        func.lower(PlantModel.notes).like(search_term),
                        func.lower(PlantModel.location).like(search_term)
                    )
                )
                .order_by(PlantModel.name)
            )
            
            result = await self.session.execute(stmt)
            plant_models = result.scalars().all()
            
            return [self._model_to_entity(model) for model in plant_models]
            
        except Exception as e:
            logger.error(f"Failed to search plants with query '{query}': {e}")
            raise DatabaseError(f"Failed to search plants: {e}")
    
    def _model_to_entity(self, model: PlantModel) -> Plant:
        """Convert database model to domain entity."""
        return Plant(
            id=PlantID(model.id),
            name=model.name,
            species_id=SpeciesID(model.species_id) if model.species_id else None,
            location=model.location,
            status=PlantStatus(model.status),
            health_score=float(model.health_score) if model.health_score else None,
            last_watered_at=model.last_watered_at,
            last_fertilized_at=model.last_fertilized_at,
            notes=model.notes,
            metadata=model.metadata,
            created_at=model.created_at,
            updated_at=model.updated_at
        )