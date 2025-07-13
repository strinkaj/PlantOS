"""
Dependency Injection infrastructure.

This module provides dependency injection capabilities following the
Dependency Inversion Principle for loose coupling and testability.
"""

from .container import (
    DIContainer,
    ServiceProvider,
    DatabaseServiceProvider,
    CacheServiceProvider,
    get_container,
    cleanup_container,
    container_scope
)

__all__ = [
    "DIContainer",
    "ServiceProvider", 
    "DatabaseServiceProvider",
    "CacheServiceProvider",
    "get_container",
    "cleanup_container",
    "container_scope"
]