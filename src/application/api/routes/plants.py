"""
Plant management API routes.

This module provides endpoints for plant CRUD operations, health monitoring,
and care scheduling with comprehensive type safety and validation.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import Field
import structlog

from src.core.domain.entities import Plant, PlantSpecies
from src.core.domain.repositories import PlantRepository, PlantSpeciesRepository
from src.core.domain.value_objects import PlantID, Location, HealthScore
from src.core.use_cases.plant_management import (
    CreatePlantUseCase,
    UpdatePlantUseCase,
    GetPlantUseCase,
    ListPlantsUseCase,
    DeletePlantUseCase,
    CalculatePlantHealthUseCase
)
from src.infrastructure.database.dependency import get_plant_repository, get_plant_species_repository
from src.shared.exceptions import PlantNotFoundError, PlantAlreadyExistsError
from src.application.models import (
    PlantCreateRequest,
    PlantUpdateRequest,
    PlantResponse,
    PlantListResponse,
    PlantHealthResponse,
    PaginationParams
)

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/", response_model=PlantResponse, status_code=status.HTTP_201_CREATED)
async def create_plant(
    request: PlantCreateRequest,
    plant_repo: PlantRepository = Depends(get_plant_repository),
    species_repo: PlantSpeciesRepository = Depends(get_plant_species_repository)
) -> PlantResponse:
    """
    Create a new plant.
    
    Args:
        request: Plant creation data with validation
        plant_repo: Plant repository dependency
        species_repo: Species repository dependency
        
    Returns:
        Created plant with generated ID
        
    Raises:
        400: Invalid species ID or plant data
        409: Plant with same name already exists
    """
    logger.info("Creating new plant", name=request.name, species_id=str(request.species_id))
    
    try:
        use_case = CreatePlantUseCase(plant_repo, species_repo)
        plant = await use_case.execute(
            name=request.name,
            species_id=request.species_id,
            location=Location(
                room=request.location.room,
                position=request.location.position,
                light_level=request.location.light_level
            ),
            notes=request.notes
        )
        
        logger.info("Plant created successfully", plant_id=str(plant.id))
        return PlantResponse.from_domain(plant)
        
    except PlantAlreadyExistsError as e:
        logger.warning("Plant creation failed - duplicate name", name=request.name)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to create plant", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create plant: {str(e)}"
        )


@router.get("/{plant_id}", response_model=PlantResponse)
async def get_plant(
    plant_id: UUID,
    plant_repo: PlantRepository = Depends(get_plant_repository)
) -> PlantResponse:
    """
    Get plant by ID.
    
    Args:
        plant_id: UUID of the plant
        plant_repo: Plant repository dependency
        
    Returns:
        Plant details with current health status
        
    Raises:
        404: Plant not found
    """
    logger.info("Fetching plant", plant_id=str(plant_id))
    
    try:
        use_case = GetPlantUseCase(plant_repo)
        plant = await use_case.execute(PlantID(plant_id))
        
        return PlantResponse.from_domain(plant)
        
    except PlantNotFoundError as e:
        logger.warning("Plant not found", plant_id=str(plant_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/", response_model=PlantListResponse)
async def list_plants(
    pagination: PaginationParams = Depends(),
    room: Optional[str] = Query(None, description="Filter by room"),
    species_id: Optional[UUID] = Query(None, description="Filter by species"),
    health_status: Optional[str] = Query(None, description="Filter by health status"),
    plant_repo: PlantRepository = Depends(get_plant_repository)
) -> PlantListResponse:
    """
    List plants with optional filtering.
    
    Args:
        pagination: Pagination parameters
        room: Optional room filter
        species_id: Optional species filter
        health_status: Optional health status filter
        plant_repo: Plant repository dependency
        
    Returns:
        Paginated list of plants
    """
    logger.info(
        "Listing plants",
        offset=pagination.offset,
        limit=pagination.limit,
        filters={
            "room": room,
            "species_id": str(species_id) if species_id else None,
            "health_status": health_status
        }
    )
    
    use_case = ListPlantsUseCase(plant_repo)
    plants, total = await use_case.execute(
        offset=pagination.offset,
        limit=pagination.limit,
        room=room,
        species_id=species_id,
        health_status=health_status
    )
    
    return PlantListResponse(
        items=[PlantResponse.from_domain(plant) for plant in plants],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit
    )


@router.put("/{plant_id}", response_model=PlantResponse)
async def update_plant(
    plant_id: UUID,
    request: PlantUpdateRequest,
    plant_repo: PlantRepository = Depends(get_plant_repository)
) -> PlantResponse:
    """
    Update plant details.
    
    Args:
        plant_id: UUID of the plant to update
        request: Update data (only provided fields will be updated)
        plant_repo: Plant repository dependency
        
    Returns:
        Updated plant details
        
    Raises:
        404: Plant not found
        400: Invalid update data
    """
    logger.info("Updating plant", plant_id=str(plant_id))
    
    try:
        use_case = UpdatePlantUseCase(plant_repo)
        
        # Build update data from request
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.location is not None:
            update_data["location"] = Location(
                room=request.location.room,
                position=request.location.position,
                light_level=request.location.light_level
            )
        if request.notes is not None:
            update_data["notes"] = request.notes
            
        plant = await use_case.execute(PlantID(plant_id), **update_data)
        
        logger.info("Plant updated successfully", plant_id=str(plant_id))
        return PlantResponse.from_domain(plant)
        
    except PlantNotFoundError as e:
        logger.warning("Plant not found for update", plant_id=str(plant_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to update plant", plant_id=str(plant_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update plant: {str(e)}"
        )


@router.delete("/{plant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_plant(
    plant_id: UUID,
    plant_repo: PlantRepository = Depends(get_plant_repository)
) -> None:
    """
    Delete a plant.
    
    Args:
        plant_id: UUID of the plant to delete
        plant_repo: Plant repository dependency
        
    Raises:
        404: Plant not found
    """
    logger.info("Deleting plant", plant_id=str(plant_id))
    
    try:
        use_case = DeletePlantUseCase(plant_repo)
        await use_case.execute(PlantID(plant_id))
        
        logger.info("Plant deleted successfully", plant_id=str(plant_id))
        
    except PlantNotFoundError as e:
        logger.warning("Plant not found for deletion", plant_id=str(plant_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/{plant_id}/health", response_model=PlantHealthResponse)
async def get_plant_health(
    plant_id: UUID,
    plant_repo: PlantRepository = Depends(get_plant_repository)
) -> PlantHealthResponse:
    """
    Get detailed plant health analysis.
    
    Args:
        plant_id: UUID of the plant
        plant_repo: Plant repository dependency
        
    Returns:
        Comprehensive health metrics and recommendations
        
    Raises:
        404: Plant not found
    """
    logger.info("Calculating plant health", plant_id=str(plant_id))
    
    try:
        use_case = CalculatePlantHealthUseCase(plant_repo)
        health_data = await use_case.execute(PlantID(plant_id))
        
        return PlantHealthResponse(
            plant_id=plant_id,
            health_score=health_data["score"],
            status=health_data["status"],
            last_watered=health_data["last_watered"],
            moisture_level=health_data["moisture_level"],
            light_level=health_data["light_level"],
            temperature=health_data["temperature"],
            humidity=health_data["humidity"],
            recommendations=health_data["recommendations"],
            alerts=health_data["alerts"]
        )
        
    except PlantNotFoundError as e:
        logger.warning("Plant not found for health check", plant_id=str(plant_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )