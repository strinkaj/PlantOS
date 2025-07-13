"""
Dependency Injection Container.

This module provides a comprehensive dependency injection container
implementing the Dependency Inversion Principle for loose coupling.
"""

from typing import TypeVar, Type, Callable, Dict, Any, Optional
from abc import ABC, abstractmethod
import inspect
from contextlib import asynccontextmanager

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class DIContainer:
    """
    Dependency Injection Container with lifecycle management.
    
    Provides registration and resolution of dependencies with support for:
    - Singleton and transient lifecycles
    - Factory functions
    - Interface to implementation mapping
    - Async dependency resolution
    """
    
    def __init__(self):
        """Initialize container."""
        self._singletons: Dict[Type, Any] = {}
        self._transients: Dict[Type, Callable] = {}
        self._factories: Dict[Type, Callable] = {}
        self._interfaces: Dict[Type, Type] = {}
        
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """
        Register a singleton dependency.
        
        Args:
            interface: Abstract interface type
            implementation: Concrete implementation type
        """
        logger.debug("Registering singleton", interface=interface.__name__, implementation=implementation.__name__)
        self._interfaces[interface] = implementation
        
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """
        Register a transient dependency (new instance each time).
        
        Args:
            interface: Abstract interface type
            implementation: Concrete implementation type
        """
        logger.debug("Registering transient", interface=interface.__name__, implementation=implementation.__name__)
        self._transients[interface] = implementation
        
    def register_factory(self, interface: Type[T], factory: Callable[..., T]) -> None:
        """
        Register a factory function for dependency creation.
        
        Args:
            interface: Interface type
            factory: Factory function that creates instances
        """
        logger.debug("Registering factory", interface=interface.__name__)
        self._factories[interface] = factory
        
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        Register a specific instance as singleton.
        
        Args:
            interface: Interface type
            instance: Pre-created instance
        """
        logger.debug("Registering instance", interface=interface.__name__)
        self._singletons[interface] = instance
        
    async def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a dependency by interface type.
        
        Args:
            interface: Interface type to resolve
            
        Returns:
            Instance of the requested type
            
        Raises:
            ValueError: If dependency is not registered
        """
        logger.debug("Resolving dependency", interface=interface.__name__)
        
        # Check if we have a pre-created singleton instance
        if interface in self._singletons:
            return self._singletons[interface]
            
        # Check if we have a factory
        if interface in self._factories:
            factory = self._factories[interface]
            instance = await self._create_with_dependencies(factory)
            return instance
            
        # Check if it's registered as singleton
        if interface in self._interfaces:
            implementation = self._interfaces[interface]
            instance = await self._create_with_dependencies(implementation)
            self._singletons[interface] = instance
            return instance
            
        # Check if it's registered as transient
        if interface in self._transients:
            implementation = self._transients[interface]
            instance = await self._create_with_dependencies(implementation)
            return instance
            
        raise ValueError(f"Dependency {interface.__name__} is not registered")
        
    async def _create_with_dependencies(self, cls_or_func: Callable) -> Any:
        """
        Create instance with automatic dependency injection.
        
        Args:
            cls_or_func: Class or function to instantiate
            
        Returns:
            Created instance with injected dependencies
        """
        # Get the signature of the constructor or function
        sig = inspect.signature(cls_or_func)
        
        # Resolve dependencies for each parameter
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param.annotation and param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = await self.resolve(param.annotation)
                except ValueError:
                    # Skip parameters we can't resolve (optional dependencies)
                    if param.default == inspect.Parameter.empty:
                        logger.warning(
                            "Cannot resolve required dependency",
                            parameter=param_name,
                            type=param.annotation.__name__
                        )
                        
        # Create the instance
        if inspect.iscoroutinefunction(cls_or_func):
            return await cls_or_func(**kwargs)
        else:
            return cls_or_func(**kwargs)
            
    async def cleanup(self) -> None:
        """Cleanup all singletons that support cleanup."""
        logger.info("Cleaning up DI container")
        
        for instance in self._singletons.values():
            if hasattr(instance, 'cleanup') and callable(instance.cleanup):
                try:
                    if inspect.iscoroutinefunction(instance.cleanup):
                        await instance.cleanup()
                    else:
                        instance.cleanup()
                except Exception as e:
                    logger.error("Error during cleanup", error=str(e))
                    
        self._singletons.clear()


class ServiceProvider(ABC):
    """Abstract base class for service providers."""
    
    @abstractmethod
    async def configure(self, container: DIContainer) -> None:
        """Configure services in the container."""
        pass


class DatabaseServiceProvider(ServiceProvider):
    """Service provider for database-related dependencies."""
    
    async def configure(self, container: DIContainer) -> None:
        """Configure database services."""
        from src.core.domain.repositories import (
            PlantRepository,
            PlantSpeciesRepository,
            SensorRepository
        )
        from src.infrastructure.database.repositories import (
            SQLAlchemyPlantRepository,
            SQLAlchemyPlantSpeciesRepository,
            SQLAlchemySensorRepository
        )
        
        # Register repository implementations
        container.register_transient(PlantRepository, SQLAlchemyPlantRepository)
        container.register_transient(PlantSpeciesRepository, SQLAlchemyPlantSpeciesRepository)
        container.register_transient(SensorRepository, SQLAlchemySensorRepository)
        
        logger.info("Database services configured")


class CacheServiceProvider(ServiceProvider):
    """Service provider for cache-related dependencies."""
    
    async def configure(self, container: DIContainer) -> None:
        """Configure cache services."""
        from src.infrastructure.cache.dependency import get_redis_client
        
        # Register Redis client factory
        import aioredis
        container.register_factory(aioredis.Redis, get_redis_client)
        
        logger.info("Cache services configured")


# Global container instance
_container: Optional[DIContainer] = None


async def get_container() -> DIContainer:
    """
    Get the global DI container.
    
    Returns:
        Configured DI container instance
    """
    global _container
    
    if _container is None:
        _container = DIContainer()
        
        # Configure all service providers
        providers = [
            DatabaseServiceProvider(),
            CacheServiceProvider(),
        ]
        
        for provider in providers:
            await provider.configure(_container)
            
        logger.info("DI container initialized")
        
    return _container


async def cleanup_container() -> None:
    """Cleanup the global container."""
    global _container
    
    if _container:
        await _container.cleanup()
        _container = None
        logger.info("DI container cleaned up")


@asynccontextmanager
async def container_scope():
    """Context manager for container lifecycle."""
    try:
        container = await get_container()
        yield container
    finally:
        await cleanup_container()