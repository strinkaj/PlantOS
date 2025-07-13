"""
Health check and monitoring API routes.

This module provides endpoints for system health checks, dependency status,
and monitoring information.
"""

from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import structlog
import psutil
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.infrastructure.database.dependency import get_database_session
from src.infrastructure.cache.dependency import get_redis_client
from src.application.models import HealthResponse, HealthStatus, DependencyStatus

logger = structlog.get_logger(__name__)
router = APIRouter()


class SystemMetrics(BaseModel):
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_connections: int


async def check_database_health(session: AsyncSession) -> DependencyStatus:
    """Check database connectivity and performance."""
    try:
        start_time = datetime.utcnow()
        
        # Execute a simple query to check connectivity
        result = await session.execute(text("SELECT 1"))
        await session.commit()
        
        # Calculate response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Check TimescaleDB extension
        ts_result = await session.execute(
            text("SELECT default_version FROM pg_available_extensions WHERE name = 'timescaledb'")
        )
        timescaledb_available = ts_result.scalar() is not None
        
        return DependencyStatus(
            name="PostgreSQL + TimescaleDB",
            status=HealthStatus.HEALTHY,
            response_time_ms=round(response_time_ms, 2),
            details={
                "timescaledb_available": timescaledb_available,
                "connection_pool_size": session.bind.pool.size() if hasattr(session.bind, 'pool') else "N/A"
            }
        )
        
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return DependencyStatus(
            name="PostgreSQL + TimescaleDB",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=0,
            error=str(e)
        )


async def check_redis_health(redis: aioredis.Redis) -> DependencyStatus:
    """Check Redis connectivity and performance."""
    try:
        start_time = datetime.utcnow()
        
        # Ping Redis
        await redis.ping()
        
        # Get Redis info
        info = await redis.info()
        
        # Calculate response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return DependencyStatus(
            name="Redis Cache",
            status=HealthStatus.HEALTHY,
            response_time_ms=round(response_time_ms, 2),
            details={
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown")
            }
        )
        
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        return DependencyStatus(
            name="Redis Cache",
            status=HealthStatus.UNHEALTHY,
            response_time_ms=0,
            error=str(e)
        )


def get_system_metrics() -> SystemMetrics:
    """Get current system resource metrics."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Active connections
        connections = len(psutil.net_connections())
        
        return SystemMetrics(
            cpu_percent=round(cpu_percent, 2),
            memory_percent=round(memory_percent, 2),
            disk_usage_percent=round(disk_usage_percent, 2),
            active_connections=connections
        )
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        return SystemMetrics(
            cpu_percent=0,
            memory_percent=0,
            disk_usage_percent=0,
            active_connections=0
        )


@router.get("/", response_model=HealthResponse)
async def health_check(
    session: AsyncSession = Depends(get_database_session),
    redis: aioredis.Redis = Depends(get_redis_client)
) -> HealthResponse:
    """
    Comprehensive health check endpoint.
    
    Returns:
        Overall system health status with dependency checks
    """
    logger.info("Performing health check")
    
    # Check all dependencies
    dependencies = []
    
    # Database health
    db_status = await check_database_health(session)
    dependencies.append(db_status)
    
    # Redis health
    redis_status = await check_redis_health(redis)
    dependencies.append(redis_status)
    
    # Determine overall status
    unhealthy_deps = [d for d in dependencies if d.status == HealthStatus.UNHEALTHY]
    degraded_deps = [d for d in dependencies if d.status == HealthStatus.DEGRADED]
    
    if unhealthy_deps:
        overall_status = HealthStatus.UNHEALTHY
    elif degraded_deps:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY
    
    # Get system metrics
    metrics = get_system_metrics()
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="1.0.0",
        dependencies=dependencies,
        metrics=metrics.dict()
    )


@router.get("/liveness", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, str]:
    """
    Simple liveness check for container orchestration.
    
    Returns:
        200 OK if the service is alive
    """
    return {"status": "alive"}


@router.get("/readiness", response_model=Dict[str, Any])
async def readiness_check(
    session: AsyncSession = Depends(get_database_session),
    redis: aioredis.Redis = Depends(get_redis_client)
) -> Dict[str, Any]:
    """
    Readiness check for container orchestration.
    
    Returns:
        200 OK if the service is ready to accept traffic
        503 Service Unavailable if dependencies are not ready
    """
    logger.info("Performing readiness check")
    
    # Quick connectivity checks
    ready = True
    errors = []
    
    # Check database
    try:
        await session.execute(text("SELECT 1"))
        await session.commit()
    except Exception as e:
        ready = False
        errors.append(f"Database not ready: {str(e)}")
    
    # Check Redis
    try:
        await redis.ping()
    except Exception as e:
        ready = False
        errors.append(f"Redis not ready: {str(e)}")
    
    if not ready:
        logger.warning("Service not ready", errors=errors)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not ready", "errors": errors}
        )
    
    return {"status": "ready"}


@router.get("/metrics/summary", response_model=Dict[str, Any])
async def metrics_summary() -> Dict[str, Any]:
    """
    Get summary of application metrics.
    
    Returns:
        Key performance indicators and metrics
    """
    logger.info("Fetching metrics summary")
    
    # Get system metrics
    system_metrics = get_system_metrics()
    
    # TODO: Add application-specific metrics
    # - Active plant count
    # - Recent sensor readings count
    # - Watering events today
    # - Alert count
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system": system_metrics.dict(),
        "application": {
            "uptime_seconds": 0,  # TODO: Track actual uptime
            "request_count": 0,   # TODO: Track from Prometheus
            "error_rate": 0.0,    # TODO: Calculate from metrics
            "average_response_time_ms": 0  # TODO: Calculate from metrics
        }
    }