"""
Base repository implementation with common functionality.

This module provides a generic base repository class that implements
common CRUD operations to reduce code duplication across repositories.
"""

from typing import TypeVar, Generic, Type, Optional, List, Dict, Any
from uuid import UUID
from abc import abstractmethod

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import structlog

from src.infrastructure.database.models.base import BaseModel
from src.shared.exceptions import DatabaseError, DuplicateResourceError

logger = structlog.get_logger(__name__)

# Type variables for generic repository
TModel = TypeVar("TModel", bound=BaseModel)
TEntity = TypeVar("TEntity")
TID = TypeVar("TID")


class BaseRepository(Generic[TModel, TEntity, TID]):
    """
    Base repository with common CRUD operations.
    
    Provides generic implementation for standard database operations
    that can be inherited by specific repositories.
    """
    
    def __init__(self, session: AsyncSession, model_class: Type[TModel]):
        """
        Initialize base repository.
        
        Args:
            session: AsyncSession for database operations
            model_class: SQLAlchemy model class
        """
        self.session = session
        self.model_class = model_class
        
    @abstractmethod
    def _to_entity(self, model: TModel) -> TEntity:
        """Convert database model to domain entity."""
        pass
        
    @abstractmethod
    def _to_model(self, entity: TEntity) -> TModel:
        """Convert domain entity to database model."""
        pass
        
    @abstractmethod
    def _get_id_value(self, entity_id: TID) -> UUID:
        """Extract UUID value from domain ID type."""
        pass
        
    async def get_by_id(self, entity_id: TID) -> Optional[TEntity]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Domain ID type
            
        Returns:
            Entity if found, None otherwise
        """
        try:
            id_value = self._get_id_value(entity_id)
            
            stmt = select(self.model_class).where(
                self.model_class.id == id_value
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                return self._to_entity(model)
                
            return None
            
        except SQLAlchemyError as e:
            logger.error(
                "Database error in get_by_id",
                entity_id=str(entity_id),
                error=str(e)
            )
            raise DatabaseError(f"Failed to fetch entity: {str(e)}")
            
    async def get_all(
        self,
        offset: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[TEntity]:
        """
        Get all entities with pagination.
        
        Args:
            offset: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filter conditions
            
        Returns:
            List of entities
        """
        try:
            stmt = select(self.model_class)
            
            # Apply filters if provided
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field) and value is not None:
                        stmt = stmt.where(
                            getattr(self.model_class, field) == value
                        )
            
            # Apply pagination
            stmt = stmt.offset(offset).limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._to_entity(model) for model in models]
            
        except SQLAlchemyError as e:
            logger.error(
                "Database error in get_all",
                error=str(e),
                filters=filters
            )
            raise DatabaseError(f"Failed to fetch entities: {str(e)}")
            
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities matching filters.
        
        Args:
            filters: Optional filter conditions
            
        Returns:
            Count of matching entities
        """
        try:
            stmt = select(func.count()).select_from(self.model_class)
            
            # Apply filters if provided
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field) and value is not None:
                        stmt = stmt.where(
                            getattr(self.model_class, field) == value
                        )
            
            result = await self.session.execute(stmt)
            return result.scalar() or 0
            
        except SQLAlchemyError as e:
            logger.error("Database error in count", error=str(e))
            raise DatabaseError(f"Failed to count entities: {str(e)}")
            
    async def create(self, entity: TEntity) -> TEntity:
        """
        Create a new entity.
        
        Args:
            entity: Domain entity to create
            
        Returns:
            Created entity with generated ID
            
        Raises:
            DuplicateResourceError: If entity violates uniqueness constraint
        """
        try:
            model = self._to_model(entity)
            
            self.session.add(model)
            await self.session.commit()
            await self.session.refresh(model)
            
            return self._to_entity(model)
            
        except IntegrityError as e:
            await self.session.rollback()
            logger.warning("Integrity error during creation", error=str(e))
            raise DuplicateResourceError(
                "Entity violates uniqueness constraint"
            )
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database error in create", error=str(e))
            raise DatabaseError(f"Failed to create entity: {str(e)}")
            
    async def update(self, entity: TEntity) -> TEntity:
        """
        Update an existing entity.
        
        Args:
            entity: Domain entity with updates
            
        Returns:
            Updated entity
        """
        try:
            model = self._to_model(entity)
            
            # Use merge to update existing entity
            merged = await self.session.merge(model)
            await self.session.commit()
            await self.session.refresh(merged)
            
            return self._to_entity(merged)
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database error in update", error=str(e))
            raise DatabaseError(f"Failed to update entity: {str(e)}")
            
    async def delete(self, entity_id: TID) -> bool:
        """
        Delete an entity.
        
        Args:
            entity_id: ID of entity to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            id_value = self._get_id_value(entity_id)
            
            stmt = delete(self.model_class).where(
                self.model_class.id == id_value
            )
            
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            return result.rowcount > 0
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                "Database error in delete",
                entity_id=str(entity_id),
                error=str(e)
            )
            raise DatabaseError(f"Failed to delete entity: {str(e)}")
            
    async def exists(self, entity_id: TID) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: ID to check
            
        Returns:
            True if exists, False otherwise
        """
        try:
            id_value = self._get_id_value(entity_id)
            
            stmt = select(
                select(self.model_class.id).where(
                    self.model_class.id == id_value
                ).exists()
            )
            
            result = await self.session.execute(stmt)
            return result.scalar() or False
            
        except SQLAlchemyError as e:
            logger.error(
                "Database error in exists",
                entity_id=str(entity_id),
                error=str(e)
            )
            raise DatabaseError(f"Failed to check existence: {str(e)}")