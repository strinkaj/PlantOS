# PlantOS Development Guide

## Development Environment Setup

### Prerequisites
- Python 3.12+
- PostgreSQL 15+ with TimescaleDB
- Redis 7+
- GCC/Clang for C development
- Docker & Docker Compose
- Make

### Initial Setup
```bash
# Clone repository
git clone https://github.com/yourusername/PlantOS.git
cd PlantOS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Initialize database
docker-compose up -d postgres redis
alembic upgrade head

# Run tests
pytest
```

## Code Quality Standards

### Python Code Style

#### Type Hints
```python
from typing import Optional, List, Dict, Union
from datetime import datetime

def calculate_water_needed(
    plant: Plant,
    last_watered: datetime,
    soil_moisture: float,
    weather_data: Optional[WeatherData] = None
) -> tuple[float, WateringPriority]:
    """
    Calculate water amount needed for a plant.
    
    Args:
        plant: Plant entity with care requirements
        last_watered: Last watering timestamp
        soil_moisture: Current soil moisture percentage (0-100)
        weather_data: Optional weather forecast data
        
    Returns:
        Tuple of (water_amount_ml, priority)
        
    Raises:
        InvalidSoilMoistureError: If moisture reading is invalid
    """
    pass
```

#### Error Handling
```python
from src.shared.exceptions import PlantOSError, SensorReadError

class SoilMoistureSensor:
    def read(self) -> float:
        try:
            raw_value = self._hardware.read_analog()
            return self._calibrate(raw_value)
        except HardwareError as e:
            logger.error(f"Sensor read failed: {e}")
            raise SensorReadError(
                sensor_type="soil_moisture",
                details=str(e)
            ) from e
```

#### Dependency Injection
```python
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Infrastructure
    database = providers.Singleton(
        Database,
        connection_string=config.database.url
    )
    
    # Repositories
    plant_repository = providers.Factory(
        PostgresPlantRepository,
        session_factory=database.provided.session_factory
    )
    
    # Use cases
    water_plant_use_case = providers.Factory(
        WaterPlantUseCase,
        plant_repository=plant_repository,
        watering_service=watering_service
    )
```

### C Code Standards

#### Header Guards and Documentation
```c
#ifndef PLANTOS_MOISTURE_SENSOR_H
#define PLANTOS_MOISTURE_SENSOR_H

/**
 * @file moisture_sensor.h
 * @brief Capacitive soil moisture sensor interface
 * @author Your Name
 * @date 2024-01-01
 */

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Initialize moisture sensor
 * @param pin ADC pin number
 * @return 0 on success, error code on failure
 */
int moisture_sensor_init(uint8_t pin);

/**
 * @brief Read calibrated moisture percentage
 * @param moisture Output moisture percentage (0-100)
 * @return 0 on success, error code on failure
 */
int moisture_sensor_read(float *moisture);

#endif /* PLANTOS_MOISTURE_SENSOR_H */
```

#### Error Handling in C
```c
#include "moisture_sensor.h"
#include "hardware/adc.h"
#include "utils/logger.h"

#define MOISTURE_SENSOR_OK 0
#define MOISTURE_SENSOR_ERROR_INIT -1
#define MOISTURE_SENSOR_ERROR_READ -2
#define MOISTURE_SENSOR_ERROR_CALIBRATION -3

typedef struct {
    uint8_t pin;
    uint16_t dry_value;
    uint16_t wet_value;
    bool initialized;
} moisture_sensor_t;

static moisture_sensor_t sensor = {0};

int moisture_sensor_read(float *moisture) {
    if (!sensor.initialized) {
        log_error("Moisture sensor not initialized");
        return MOISTURE_SENSOR_ERROR_INIT;
    }
    
    if (moisture == NULL) {
        log_error("NULL pointer provided for moisture output");
        return MOISTURE_SENSOR_ERROR_READ;
    }
    
    uint16_t raw_value;
    if (adc_read(sensor.pin, &raw_value) != 0) {
        log_error("Failed to read ADC pin %d", sensor.pin);
        return MOISTURE_SENSOR_ERROR_READ;
    }
    
    // Calibrate and convert to percentage
    *moisture = calibrate_moisture(raw_value);
    return MOISTURE_SENSOR_OK;
}
```

### SQL Best Practices

#### Migrations
```sql
-- migrations/001_create_plants_table.sql
BEGIN;

CREATE TABLE IF NOT EXISTS plants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    scientific_name VARCHAR(255),
    plant_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT plants_name_length CHECK (LENGTH(name) >= 1)
);

CREATE INDEX idx_plants_type ON plants(plant_type);
CREATE INDEX idx_plants_created_at ON plants(created_at DESC);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_plants_updated_at 
    BEFORE UPDATE ON plants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMIT;
```

#### Stored Procedures
```sql
-- sql/functions/get_watering_schedule.sql
CREATE OR REPLACE FUNCTION get_watering_schedule(
    p_plant_id UUID,
    p_days_ahead INTEGER DEFAULT 7
) 
RETURNS TABLE (
    scheduled_date DATE,
    water_amount_ml INTEGER,
    priority VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY
    WITH plant_requirements AS (
        SELECT 
            p.id,
            ps.water_frequency_days,
            ps.water_amount_ml_base,
            ps.season_modifier
        FROM plants p
        JOIN plant_species ps ON p.species_id = ps.id
        WHERE p.id = p_plant_id
    )
    SELECT 
        date_series.date AS scheduled_date,
        (pr.water_amount_ml_base * pr.season_modifier)::INTEGER AS water_amount_ml,
        CASE 
            WHEN date_series.date <= CURRENT_DATE + INTERVAL '1 day' THEN 'high'
            WHEN date_series.date <= CURRENT_DATE + INTERVAL '3 days' THEN 'medium'
            ELSE 'low'
        END AS priority
    FROM plant_requirements pr
    CROSS JOIN LATERAL (
        SELECT generate_series(
            CURRENT_DATE,
            CURRENT_DATE + (p_days_ahead || ' days')::INTERVAL,
            (pr.water_frequency_days || ' days')::INTERVAL
        )::DATE AS date
    ) date_series;
END;
$$ LANGUAGE plpgsql;
```

## Testing Strategy

### Unit Testing (Python)
```python
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.core.use_cases.plant.water_plant import WaterPlantUseCase
from src.core.domain.entities import Plant, WateringEvent

class TestWaterPlantUseCase:
    @pytest.fixture
    def mock_plant_repo(self):
        return Mock()
    
    @pytest.fixture
    def mock_watering_service(self):
        return Mock()
    
    @pytest.fixture
    def use_case(self, mock_plant_repo, mock_watering_service):
        return WaterPlantUseCase(
            plant_repository=mock_plant_repo,
            watering_service=mock_watering_service
        )
    
    @pytest.mark.asyncio
    async def test_water_plant_success(self, use_case, mock_plant_repo):
        # Arrange
        plant_id = "123e4567-e89b-12d3-a456-426614174000"
        plant = Plant(
            id=plant_id,
            name="Tomato Plant",
            last_watered=datetime.now() - timedelta(days=3)
        )
        mock_plant_repo.find_by_id.return_value = plant
        
        # Act
        result = await use_case.execute(plant_id, water_amount_ml=250)
        
        # Assert
        assert result.success is True
        assert result.water_amount_ml == 250
        mock_plant_repo.save.assert_called_once()
```

### Integration Testing
```python
import pytest
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine

@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("postgres:15-alpine") as postgres:
        yield postgres

@pytest.fixture
def database_url(postgres_container):
    return postgres_container.get_connection_url()

@pytest.mark.integration
class TestPlantRepository:
    def test_save_and_retrieve_plant(self, database_url):
        # Setup
        engine = create_engine(database_url)
        repo = PostgresPlantRepository(engine)
        
        # Create plant
        plant = Plant(name="Test Plant", species="Tomato")
        repo.save(plant)
        
        # Retrieve plant
        retrieved = repo.find_by_id(plant.id)
        assert retrieved.name == "Test Plant"
```

### Hardware Testing (C)
```c
#include "unity.h"
#include "moisture_sensor.h"
#include "mock_adc.h"

void setUp(void) {
    mock_adc_reset();
}

void test_moisture_sensor_init_success(void) {
    // Arrange
    uint8_t pin = 34;
    
    // Act
    int result = moisture_sensor_init(pin);
    
    // Assert
    TEST_ASSERT_EQUAL(0, result);
}

void test_moisture_sensor_read_returns_valid_percentage(void) {
    // Arrange
    moisture_sensor_init(34);
    mock_adc_set_value(2048);  // Mid-range value
    
    // Act
    float moisture;
    int result = moisture_sensor_read(&moisture);
    
    // Assert
    TEST_ASSERT_EQUAL(0, result);
    TEST_ASSERT_FLOAT_WITHIN(0.1, 50.0, moisture);
}
```

## Performance Optimization

### Database Query Optimization
```python
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

class PostgresPlantRepository:
    async def find_plants_needing_water(self) -> List[Plant]:
        # Optimized query with eager loading
        stmt = (
            select(Plant)
            .options(selectinload(Plant.species))
            .join(Plant.last_watering_event)
            .where(
                and_(
                    Plant.is_active == True,
                    WateringEvent.timestamp < datetime.now() - timedelta(days=2)
                )
            )
            .order_by(WateringEvent.timestamp.asc())
            .limit(100)
        )
        
        async with self.session() as session:
            result = await session.execute(stmt)
            return result.scalars().all()
```

### Caching Strategy
```python
from functools import lru_cache
from redis import Redis
import json

class PlantCareCache:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour
    
    async def get_plant_care_requirements(self, species_id: str) -> Optional[Dict]:
        # Try cache first
        cached = self.redis.get(f"species:{species_id}:care")
        if cached:
            return json.loads(cached)
        
        # Load from database
        requirements = await self._load_from_db(species_id)
        if requirements:
            self.redis.setex(
                f"species:{species_id}:care",
                self.ttl,
                json.dumps(requirements)
            )
        
        return requirements
```

## Monitoring and Observability

### Structured Logging
```python
import structlog
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

logger = structlog.get_logger()

def configure_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            add_request_id,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def add_request_id(logger, log_method, event_dict):
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
watering_events_total = Counter(
    'plantos_watering_events_total',
    'Total number of watering events',
    ['plant_id', 'trigger_type']
)

sensor_read_duration = Histogram(
    'plantos_sensor_read_duration_seconds',
    'Time spent reading sensor values',
    ['sensor_type']
)

soil_moisture_gauge = Gauge(
    'plantos_soil_moisture_percentage',
    'Current soil moisture reading',
    ['plant_id', 'sensor_id']
)
```

## Security Best Practices

### Input Validation
```python
from pydantic import BaseModel, validator, Field
from typing import Optional
import re

class CreatePlantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    species_id: Optional[str] = None
    location: Optional[str] = None
    
    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', v):
            raise ValueError('Plant name contains invalid characters')
        return v.strip()
    
    @validator('species_id')
    def validate_species_id(cls, v):
        if v and not is_valid_uuid(v):
            raise ValueError('Invalid species ID format')
        return v
```

### SQL Injection Prevention
```python
from sqlalchemy import text
from sqlalchemy.sql import literal_column

class SecurePlantRepository:
    async def search_plants(self, search_term: str) -> List[Plant]:
        # Safe parameterized query
        stmt = text("""
            SELECT * FROM plants 
            WHERE name ILIKE :search_term 
            OR scientific_name ILIKE :search_term
            ORDER BY name
            LIMIT 100
        """).bindparams(search_term=f"%{search_term}%")
        
        # Never do this:
        # bad_query = f"SELECT * FROM plants WHERE name LIKE '%{search_term}%'"
        
        async with self.session() as session:
            result = await session.execute(stmt)
            return [Plant(**row) for row in result]
```