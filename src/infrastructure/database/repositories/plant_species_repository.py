"""
Plant species repository implementation using SQLAlchemy.

This module provides the concrete implementation of the PlantSpeciesRepository
interface using SQLAlchemy for database operations.
"""

from typing import Optional, List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from src.core.domain.entities import PlantSpecies
from src.core.domain.repositories import PlantSpeciesRepository
from src.core.domain.value_objects import SpeciesID
from src.infrastructure.database.models import PlantSpeciesModel
from src.infrastructure.database.repositories.base import BaseRepository
from src.shared.exceptions import SpeciesNotFoundError

logger = structlog.get_logger(__name__)


class SQLAlchemyPlantSpeciesRepository(BaseRepository[PlantSpeciesModel, PlantSpecies, SpeciesID], PlantSpeciesRepository):
    """SQLAlchemy implementation of PlantSpeciesRepository."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        super().__init__(session, PlantSpeciesModel)
        
    def _to_entity(self, model: PlantSpeciesModel) -> PlantSpecies:
        """Convert database model to domain entity."""
        return PlantSpecies(
            id=SpeciesID(model.id),
            scientific_name=model.scientific_name,
            common_name=model.common_name,
            family=model.family,
            care_instructions={
                "watering_frequency_days": model.watering_frequency_days,
                "min_temperature": model.min_temperature,
                "max_temperature": model.max_temperature,
                "min_humidity": model.min_humidity,
                "max_humidity": model.max_humidity,
                "min_light_hours": model.min_light_hours,
                "max_light_hours": model.max_light_hours,
                "soil_type": model.soil_type,
                "fertilizer_frequency_days": model.fertilizer_frequency_days,
                "pruning_frequency_days": model.pruning_frequency_days,
                "notes": model.notes
            },
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        
    def _to_model(self, entity: PlantSpecies) -> PlantSpeciesModel:
        """Convert domain entity to database model."""
        return PlantSpeciesModel(
            id=entity.id.value,
            scientific_name=entity.scientific_name,
            common_name=entity.common_name,
            family=entity.family,
            watering_frequency_days=entity.care_instructions.get("watering_frequency_days"),
            min_temperature=entity.care_instructions.get("min_temperature"),
            max_temperature=entity.care_instructions.get("max_temperature"),
            min_humidity=entity.care_instructions.get("min_humidity"),
            max_humidity=entity.care_instructions.get("max_humidity"),
            min_light_hours=entity.care_instructions.get("min_light_hours"),
            max_light_hours=entity.care_instructions.get("max_light_hours"),
            soil_type=entity.care_instructions.get("soil_type"),
            fertilizer_frequency_days=entity.care_instructions.get("fertilizer_frequency_days"),
            pruning_frequency_days=entity.care_instructions.get("pruning_frequency_days"),
            notes=entity.care_instructions.get("notes")
        )
        
    def _get_id_value(self, entity_id: SpeciesID) -> UUID:
        """Extract UUID from SpeciesID."""
        return entity_id.value
        
    async def get_by_id(self, species_id: SpeciesID) -> Optional[PlantSpecies]:
        """Get species by ID."""
        logger.info("Fetching species by ID", species_id=str(species_id))
        
        species = await super().get_by_id(species_id)
        
        if not species:
            logger.warning("Species not found", species_id=str(species_id))
            
        return species
        
    async def get_by_scientific_name(self, scientific_name: str) -> Optional[PlantSpecies]:
        """Get species by scientific name."""
        logger.info("Fetching species by scientific name", name=scientific_name)
        
        try:
            stmt = select(PlantSpeciesModel).where(
                PlantSpeciesModel.scientific_name == scientific_name
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                return self._to_entity(model)
                
            return None
            
        except Exception as e:
            logger.error(
                "Failed to fetch species by scientific name",
                name=scientific_name,
                error=str(e)
            )
            raise
            
    async def get_by_common_name(self, common_name: str) -> List[PlantSpecies]:
        """Get species by common name (can be multiple)."""
        logger.info("Fetching species by common name", name=common_name)
        
        try:
            stmt = select(PlantSpeciesModel).where(
                PlantSpeciesModel.common_name.ilike(f"%{common_name}%")
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._to_entity(model) for model in models]
            
        except Exception as e:
            logger.error(
                "Failed to fetch species by common name",
                name=common_name,
                error=str(e)
            )
            raise
            
    async def search(self, query: str) -> List[PlantSpecies]:
        """Search species by name."""
        logger.info("Searching species", query=query)
        
        try:
            stmt = select(PlantSpeciesModel).where(
                PlantSpeciesModel.scientific_name.ilike(f"%{query}%") |
                PlantSpeciesModel.common_name.ilike(f"%{query}%") |
                PlantSpeciesModel.family.ilike(f"%{query}%")
            ).limit(20)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._to_entity(model) for model in models]
            
        except Exception as e:
            logger.error("Failed to search species", query=query, error=str(e))
            raise
            
    async def list_all(self, offset: int = 0, limit: int = 100) -> List[PlantSpecies]:
        """List all species with pagination."""
        logger.info("Listing all species", offset=offset, limit=limit)
        
        return await super().get_all(offset=offset, limit=limit)