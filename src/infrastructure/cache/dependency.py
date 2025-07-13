"""
Cache dependency injection.

This module provides FastAPI dependency injection for Redis cache
with proper connection management.
"""

import os
from typing import AsyncGenerator

import aioredis
import structlog

logger = structlog.get_logger(__name__)


class RedisManager:
    """Redis connection manager with connection pooling."""
    
    def __init__(self):
        """Initialize Redis manager."""
        self._redis: aioredis.Redis = None
        self._url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
    async def get_redis(self) -> aioredis.Redis:
        """
        Get Redis client with connection pooling.
        
        Returns:
            Redis client instance
        """
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
                max_connections=20
            )
            
            # Test connection
            try:
                await self._redis.ping()
                logger.info("Redis connection established", url=self._url)
            except Exception as e:
                logger.error("Failed to connect to Redis", url=self._url, error=str(e))
                raise
                
        return self._redis
        
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")


# Global Redis manager instance
_redis_manager = RedisManager()


async def get_redis_client() -> aioredis.Redis:
    """
    FastAPI dependency for Redis client.
    
    Returns:
        Redis client instance with connection pooling
    """
    return await _redis_manager.get_redis()


async def close_redis_connections():
    """Close all Redis connections - called during app shutdown."""
    await _redis_manager.close()