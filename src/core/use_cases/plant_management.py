"""
Plant management use cases.

These use cases handle all plant lifecycle operations including creation,
updates, health monitoring, and removal from the system.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

import structlog

from src.core.domain.entities import Plant, PlantSpecies
from src.core.domain.repositories import PlantRepository, PlantSpeciesRepository
from src.shared.types import PlantID, SpeciesID, HealthScore, Location
from src.shared.exceptions import (
    PlantNotFoundError,
    PlantAlreadyExistsError, 
    SpeciesNotFoundError,
    ValidationError
)
from src.shared.contracts import require, ensure

logger = structlog.get_logger(__name__)


class CreatePlantUseCase:
    """Use case for creating a new plant in the system."""
    
    def __init__(
        self,
        plant_repo: PlantRepository,
        species_repo: PlantSpeciesRepository
    ):
        self.plant_repo = plant_repo
        self.species_repo = species_repo
        
    @require(lambda name, species_id: name and name.strip(), "Plant name cannot be empty")
    @require(lambda name, species_id: len(name.strip()) >= 2, "Plant name must be at least 2 characters")
    async def execute(
        self,
        name: str,
        species_id: UUID,
        location: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> Plant:
        """
        Create a new plant.
        
        Args:
            name: Plant name (must be unique)
            species_id: ID of the plant species
            location: Optional location data
            notes: Optional notes about the plant
            
        Returns:
            Created plant entity
            
        Raises:
            PlantAlreadyExistsError: If plant name already exists
            SpeciesNotFoundError: If species doesn't exist
            ValidationError: If input validation fails
        """
        logger.info("Creating new plant", name=name, species_id=str(species_id))
        
        # Validate species exists
        species = await self.species_repo.get_by_id(SpeciesID(species_id))
        if not species:
            raise SpeciesNotFoundError(f"Species with ID {species_id} not found")
            
        # Check if plant name already exists
        existing_plant = await self.plant_repo.get_by_name(name.strip())
        if existing_plant:
            raise PlantAlreadyExistsError(f"Plant with name '{name}' already exists")
            
        # Create location if provided
        plant_location = None
        if location:
            plant_location = Location(
                room=location.get("room"),
                position=location.get("position"),
                light_level=location.get("light_level")
            )
            
        # Create plant entity
        plant = Plant(
            id=PlantID(uuid4()),
            name=name.strip(),
            species_id=SpeciesID(species_id),
            location=plant_location,
            notes=notes,
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Save to repository
        created_plant = await self.plant_repo.create(plant)
        
        logger.info(
            "Plant created successfully",
            plant_id=str(created_plant.id),
            name=created_plant.name
        )
        
        return created_plant


class GetPlantUseCase:
    """Use case for retrieving a plant by ID."""
    
    def __init__(self, plant_repo: PlantRepository):
        self.plant_repo = plant_repo
        
    @require(lambda plant_id: plant_id is not None, "Plant ID cannot be None")
    async def execute(self, plant_id: UUID) -> Plant:
        """
        Get a plant by its ID.
        
        Args:
            plant_id: ID of the plant to retrieve
            
        Returns:
            Plant entity
            
        Raises:
            PlantNotFoundError: If plant doesn't exist
        """
        logger.info("Retrieving plant", plant_id=str(plant_id))
        
        plant = await self.plant_repo.get_by_id(PlantID(plant_id))
        if not plant:
            raise PlantNotFoundError(f"Plant with ID {plant_id} not found")
            
        logger.info("Plant retrieved successfully", plant_id=str(plant_id))
        return plant


class ListPlantsUseCase:
    """Use case for listing plants with pagination and filtering."""
    
    def __init__(self, plant_repo: PlantRepository):
        self.plant_repo = plant_repo
        
    @require(lambda offset, limit: offset >= 0, "Offset must be non-negative")
    @require(lambda offset, limit: limit > 0, "Limit must be positive")
    @require(lambda offset, limit: limit <= 1000, "Limit cannot exceed 1000")
    async def execute(
        self,
        offset: int = 0,
        limit: int = 50,
        status: Optional[str] = None,
        species_id: Optional[UUID] = None
    ) -> tuple[List[Plant], int]:
        """
        List plants with pagination and filtering.
        
        Args:
            offset: Number of plants to skip
            limit: Maximum number of plants to return
            status: Optional status filter
            species_id: Optional species filter
            
        Returns:
            Tuple of (plants list, total count)
        """
        logger.info(
            "Listing plants",
            offset=offset,
            limit=limit,
            status=status,
            species_id=str(species_id) if species_id else None
        )
        
        if species_id:
            plants = await self.plant_repo.get_by_species(SpeciesID(species_id))
        else:
            plants = await self.plant_repo.get_all(status=status)
            
        # Apply pagination
        total_count = len(plants)
        paginated_plants = plants[offset:offset + limit]
        
        logger.info(
            "Plants listed successfully",
            returned_count=len(paginated_plants),
            total_count=total_count
        )
        
        return paginated_plants, total_count


class UpdatePlantUseCase:
    """Use case for updating an existing plant."""
    
    def __init__(self, plant_repo: PlantRepository):
        self.plant_repo = plant_repo
        
    @require(lambda plant_id: plant_id is not None, "Plant ID cannot be None")
    async def execute(
        self,
        plant_id: UUID,
        name: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> Plant:
        """
        Update an existing plant.
        
        Args:
            plant_id: ID of the plant to update
            name: Optional new name
            location: Optional new location data
            notes: Optional new notes
            
        Returns:
            Updated plant entity
            
        Raises:
            PlantNotFoundError: If plant doesn't exist
            PlantAlreadyExistsError: If new name conflicts with existing plant
        """
        logger.info("Updating plant", plant_id=str(plant_id))
        
        # Get existing plant
        plant = await self.plant_repo.get_by_id(PlantID(plant_id))
        if not plant:
            raise PlantNotFoundError(f"Plant with ID {plant_id} not found")
            
        # Check name uniqueness if changing name
        if name and name.strip() != plant.name:
            existing_plant = await self.plant_repo.get_by_name(name.strip())
            if existing_plant and existing_plant.id != plant.id:
                raise PlantAlreadyExistsError(f"Plant with name '{name}' already exists")
                
        # Update fields
        if name:
            plant.name = name.strip()
            
        if location is not None:
            if location:
                plant.location = Location(
                    room=location.get("room"),
                    position=location.get("position"),
                    light_level=location.get("light_level")
                )
            else:
                plant.location = None
                
        if notes is not None:
            plant.notes = notes
            
        plant.updated_at = datetime.utcnow()
        
        # Save changes
        updated_plant = await self.plant_repo.update(plant)
        
        logger.info("Plant updated successfully", plant_id=str(plant_id))
        return updated_plant


class DeletePlantUseCase:
    """Use case for removing a plant from the system."""
    
    def __init__(self, plant_repo: PlantRepository):
        self.plant_repo = plant_repo
        
    @require(lambda plant_id: plant_id is not None, "Plant ID cannot be None")
    async def execute(self, plant_id: UUID) -> None:
        """
        Delete a plant from the system.
        
        Args:
            plant_id: ID of the plant to delete
            
        Raises:
            PlantNotFoundError: If plant doesn't exist
        """
        logger.info("Deleting plant", plant_id=str(plant_id))
        
        # Verify plant exists
        plant = await self.plant_repo.get_by_id(PlantID(plant_id))
        if not plant:
            raise PlantNotFoundError(f"Plant with ID {plant_id} not found")
            
        # Delete the plant
        await self.plant_repo.delete(PlantID(plant_id))
        
        logger.info("Plant deleted successfully", plant_id=str(plant_id))


class CalculatePlantHealthUseCase:
    """Use case for calculating and updating plant health scores."""
    
    def __init__(self, plant_repo: PlantRepository):
        self.plant_repo = plant_repo
        
    @require(lambda plant_id: plant_id is not None, "Plant ID cannot be None")
    async def execute(self, plant_id: UUID) -> Dict[str, Any]:
        """
        Calculate plant health score based on sensor data and care history.
        
        Args:
            plant_id: ID of the plant to analyze
            
        Returns:
            Dictionary with health analysis results
            
        Raises:
            PlantNotFoundError: If plant doesn't exist
        """
        logger.info("Calculating plant health", plant_id=str(plant_id))
        
        # Get plant
        plant = await self.plant_repo.get_by_id(PlantID(plant_id))
        if not plant:
            raise PlantNotFoundError(f"Plant with ID {plant_id} not found")
            
        # TODO: Implement actual health calculation algorithm
        # This would analyze:
        # - Recent sensor readings (moisture, temperature, pH, light)
        # - Watering frequency and timing
        # - Plant species requirements
        # - Growth patterns and history
        
        # For now, return a basic health assessment
        health_data = {
            "plant_id": plant_id,
            "health_score": 85.0,  # Placeholder score
            "status": "healthy",
            "last_watered": None,  # Would come from care history
            "moisture_level": None,  # Would come from latest sensor reading
            "light_level": None,
            "temperature": None,
            "humidity": None,
            "recommendations": [
                "Monitor soil moisture levels",
                "Ensure adequate lighting"
            ],
            "alerts": []
        }
        
        # Update plant health score
        if plant.health_score is None or abs(plant.health_score.value - health_data["health_score"]) > 5:
            plant.health_score = HealthScore(health_data["health_score"])
            plant.updated_at = datetime.utcnow()
            await self.plant_repo.update(plant)
            
        logger.info(
            "Plant health calculated",
            plant_id=str(plant_id),
            health_score=health_data["health_score"]
        )
        
        return health_data