"""
Database configuration and connection management for PlantOS.

This module provides async SQLAlchemy setup with connection pooling,
health checks, and production-grade connection management.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from urllib.parse import urlparse

from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncEngine, 
    async_sessionmaker, 
    AsyncSession
)
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base

from src.shared.exceptions import DatabaseError, ConfigurationError

logger = logging.getLogger(__name__)

# SQLAlchemy base class for all models
Base = declarative_base()


class DatabaseConfig:
    """Database configuration with validation and connection pooling settings."""
    
    def __init__(
        self,
        url: str,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        connect_timeout: int = 10,
        command_timeout: int = 60,
        isolation_level: Optional[str] = None
    ):
        self.url = self._validate_url(url)
        self.echo = echo
        self.pool_size = self._validate_positive_int(pool_size, "pool_size")
        self.max_overflow = self._validate_positive_int(max_overflow, "max_overflow")
        self.pool_timeout = self._validate_positive_int(pool_timeout, "pool_timeout")
        self.pool_recycle = self._validate_positive_int(pool_recycle, "pool_recycle")
        self.pool_pre_ping = pool_pre_ping
        self.connect_timeout = self._validate_positive_int(connect_timeout, "connect_timeout")
        self.command_timeout = self._validate_positive_int(command_timeout, "command_timeout")
        self.isolation_level = isolation_level
    
    @staticmethod
    def _validate_url(url: str) -> str:
        """Validate database URL format."""
        if not url:
            raise ConfigurationError("Database URL cannot be empty")
        
        try:
            parsed = urlparse(url)
            if not parsed.scheme:
                raise ConfigurationError("Database URL must include a scheme (e.g., postgresql://)")
            if parsed.scheme not in ['postgresql', 'postgresql+asyncpg', 'sqlite', 'sqlite+aiosqlite']:
                raise ConfigurationError(f"Unsupported database scheme: {parsed.scheme}")
        except Exception as e:
            raise ConfigurationError(f"Invalid database URL: {e}")
        
        return url
    
    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        """Validate that a value is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive integer, got: {value}")
        return value
    
    def to_engine_kwargs(self) -> Dict[str, Any]:
        """Convert config to SQLAlchemy engine kwargs."""
        kwargs = {
            "echo": self.echo,
            "pool_pre_ping": self.pool_pre_ping,
            "connect_args": {
                "command_timeout": self.command_timeout,
                "server_settings": {
                    "application_name": "PlantOS",
                    "jit": "off"  # Disable JIT for faster startup
                }
            }
        }
        
        # Add connection pooling for non-SQLite databases
        if not self.url.startswith('sqlite'):
            kwargs.update({
                "poolclass": QueuePool,
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_timeout": self.pool_timeout,
                "pool_recycle": self.pool_recycle,
            })
            
            # Add connection timeout for PostgreSQL
            if 'postgresql' in self.url:
                kwargs["connect_args"]["connect_timeout"] = self.connect_timeout
        else:
            # SQLite doesn't support connection pooling
            kwargs["poolclass"] = NullPool
        
        if self.isolation_level:
            kwargs["isolation_level"] = self.isolation_level
        
        return kwargs


class DatabaseManager:
    """Manages database connections, health checks, and lifecycle."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._health_check_query = text("SELECT 1")
        
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            # Create async engine
            engine_kwargs = self.config.to_engine_kwargs()
            self.engine = create_async_engine(self.config.url, **engine_kwargs)
            
            # Set up session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Add event listeners for connection management
            self._setup_event_listeners()
            
            # Test connection
            await self.health_check()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for connection management."""
        if not self.engine:
            return
        
        @event.listens_for(self.engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Configure connection on connect."""
            logger.debug(f"New database connection established: {id(dbapi_connection)}")
            
            # Set connection-specific settings for PostgreSQL
            if 'postgresql' in self.config.url:
                with dbapi_connection.cursor() as cursor:
                    # Set timezone to UTC
                    cursor.execute("SET timezone = 'UTC'")
                    # Optimize for OLTP workload
                    cursor.execute("SET synchronous_commit = on")
                    cursor.execute("SET wal_buffers = '16MB'")
        
        @event.listens_for(self.engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            logger.debug(f"Connection checked out from pool: {id(dbapi_connection)}")
        
        @event.listens_for(self.engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Handle connection checkin to pool."""
            logger.debug(f"Connection returned to pool: {id(dbapi_connection)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        if not self.engine:
            raise DatabaseError("Database not initialized")
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(self._health_check_query)
                await result.fetchone()
            
            # Get pool status
            pool_status = self._get_pool_status()
            
            return {
                "status": "healthy",
                "url": self._sanitize_url(self.config.url),
                "pool": pool_status,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    def _get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status."""
        if not self.engine or not hasattr(self.engine, 'pool'):
            return {"type": "no_pool"}
        
        pool = self.engine.pool
        return {
            "type": pool.__class__.__name__,
            "size": getattr(pool, 'size', lambda: 0)(),
            "checked_in": getattr(pool, 'checkedin', lambda: 0)(),
            "checked_out": getattr(pool, 'checkedout', lambda: 0)(),
            "overflow": getattr(pool, 'overflow', lambda: 0)(),
            "invalid": getattr(pool, 'invalid', lambda: 0)()
        }
    
    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Remove sensitive information from database URL."""
        try:
            parsed = urlparse(url)
            if parsed.password:
                sanitized = url.replace(parsed.password, "***")
            else:
                sanitized = url
            return sanitized
        except Exception:
            return "***"
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup."""
        if not self.session_factory:
            raise DatabaseError("Database not initialized")
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            await session.close()
    
    async def execute_raw_sql(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute raw SQL query with parameters."""
        if not self.engine:
            raise DatabaseError("Database not initialized")
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text(query), params or {})
                return result
        except Exception as e:
            logger.error(f"Raw SQL execution failed: {e}")
            raise DatabaseError(f"SQL execution failed: {e}")
    
    async def close(self) -> None:
        """Close database connections and cleanup."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            raise DatabaseError("Database not initialized")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(f"Table creation failed: {e}")
    
    async def drop_tables(self) -> None:
        """Drop all database tables (use with caution!)."""
        if not self.engine:
            raise DatabaseError("Database not initialized")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise DatabaseError(f"Table drop failed: {e}")


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


async def initialize_database(config: DatabaseConfig) -> DatabaseManager:
    """Initialize global database manager."""
    global db_manager
    db_manager = DatabaseManager(config)
    await db_manager.initialize()
    return db_manager


async def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    if db_manager is None:
        raise DatabaseError("Database not initialized. Call initialize_database() first.")
    return db_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection function for FastAPI to get database session."""
    manager = await get_database_manager()
    async with manager.get_session() as session:
        yield session


async def close_database() -> None:
    """Close database connections."""
    global db_manager
    if db_manager:
        await db_manager.close()
        db_manager = None