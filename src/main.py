"""
Main FastAPI application for PlantOS.

This module creates and configures the FastAPI application with all
necessary middleware, dependencies, and route handlers.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

from src.application.models import APIConfig, ErrorResponse
from src.infrastructure.database.config import DatabaseConfig, initialize_database, close_database
from src.infrastructure.database.models import setup_timescaledb_features
from src.shared.exceptions import PlantOSError, get_http_status_code, should_log_error
from src.application.api.routes import plants, health, system
from src.infrastructure.logging.config import configure_logging
from src.infrastructure.monitoring.metrics import setup_metrics

# Configure structured logging
configure_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting PlantOS application...")
    
    try:
        # Initialize database
        db_config = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "postgresql+asyncpg://plantos_user:plantos_dev_password@localhost:5432/plantos_dev"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DATABASE_POOL_RECYCLE", "3600")),
        )
        
        db_manager = await initialize_database(db_config)
        
        # Create tables if they don't exist
        await db_manager.create_tables()
        
        # Setup TimescaleDB features
        await setup_timescaledb_features(db_manager)
        
        # Initialize metrics
        setup_metrics()
        
        logger.info("PlantOS application started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    
    # Shutdown
    logger.info("Shutting down PlantOS application...")
    
    try:
        await close_database()
        logger.info("PlantOS application shut down successfully")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Load configuration
    config = APIConfig()
    
    # Create FastAPI app
    app = FastAPI(
        title=config.title,
        version=config.version,
        description=config.description,
        docs_url=config.docs_url,
        redoc_url=config.redoc_url,
        openapi_url=config.openapi_url,
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app, config)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    setup_routes(app)
    
    # Setup metrics
    setup_prometheus_metrics(app)
    
    return app


def setup_middleware(app: FastAPI, config: APIConfig) -> None:
    """Configure application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=config.cors_methods,
        allow_headers=config.cors_headers,
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all HTTP requests with correlation IDs."""
        
        # Generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or f"req_{id(request)}"
        
        # Add correlation ID to structlog context
        with structlog.contextvars.bound_contextvars(
            correlation_id=correlation_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent", ""),
        ):
            start_time = asyncio.get_event_loop().time()
            
            logger.info("Request started")
            
            try:
                response = await call_next(request)
                
                # Add correlation ID to response headers
                response.headers["X-Correlation-ID"] = correlation_id
                
                # Log response
                duration = asyncio.get_event_loop().time() - start_time
                logger.info(
                    "Request completed",
                    status_code=response.status_code,
                    duration_ms=round(duration * 1000, 2)
                )
                
                return response
                
            except Exception as e:
                duration = asyncio.get_event_loop().time() - start_time
                logger.error(
                    "Request failed",
                    error=str(e),
                    duration_ms=round(duration * 1000, 2)
                )
                raise


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers."""
    
    @app.exception_handler(PlantOSError)
    async def plantos_exception_handler(request: Request, exc: PlantOSError):
        """Handle PlantOS custom exceptions."""
        
        status_code = get_http_status_code(exc)
        
        # Log error if appropriate
        if should_log_error(exc):
            logger.error(
                "PlantOS error occurred",
                error_type=type(exc).__name__,
                error_message=exc.message,
                error_code=exc.error_code,
                details=exc.details,
                status_code=status_code
            )
        
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=exc.message,
                detail=str(exc.details) if exc.details else None,
                code=exc.error_code
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        
        logger.error(
            "Unexpected error occurred",
            error_type=type(exc).__name__,
            error_message=str(exc),
            request_method=request.method,
            request_url=str(request.url)
        )
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail="An unexpected error occurred"
            ).dict()
        )


def setup_routes(app: FastAPI) -> None:
    """Setup application routes."""
    
    # Include API routers
    app.include_router(
        plants.router,
        prefix="/api/v1/plants",
        tags=["Plants"]
    )
    
    app.include_router(
        health.router,
        prefix="/api/v1/health",
        tags=["Health"]
    )
    
    app.include_router(
        system.router,
        prefix="/api/v1/system",
        tags=["System"]
    )
    
    # Root endpoint
    @app.get("/", response_model=Dict[str, Any])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "PlantOS API",
            "version": "1.0.0",
            "description": "Production-grade plant care automation system",
            "status": "healthy",
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
                "health": "/api/v1/health",
                "plants": "/api/v1/plants",
                "metrics": "/metrics"
            }
        }


def setup_prometheus_metrics(app: FastAPI) -> None:
    """Setup Prometheus metrics collection."""
    
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="plantos_requests_inprogress",
        inprogress_labels=True,
    )
    
    # Add custom metrics
    instrumentator.add(
        lambda info: info.modified_duration if info.modified_duration else 0,
        metric_name="plantos_request_duration_seconds",
        metric_doc="Request duration in seconds",
    )
    
    instrumentator.instrument(app).expose(app, endpoint="/metrics")


# Create the application instance
app = create_application()


# Development server entry point
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_config=None,  # Use our custom logging configuration
        access_log=False,  # Handled by our middleware
    )