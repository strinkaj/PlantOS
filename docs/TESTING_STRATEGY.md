# PlantOS Testing Strategy

## Testing Philosophy
- **Test Pyramid**: Unit > Integration > E2E tests
- **TDD Approach**: Write tests first for critical business logic
- **Coverage Goals**: 90% for core domain, 80% for infrastructure
- **Performance**: All tests must run in <5 minutes locally

## Test Structure

### Unit Tests

#### Python Unit Test Example
```python
# tests/unit/core/domain/test_plant.py
import pytest
from datetime import datetime, timedelta
from src.core.domain.entities import Plant, PlantSpecies
from src.core.domain.value_objects import WateringRequirement

class TestPlant:
    @pytest.fixture
    def tomato_species(self):
        return PlantSpecies(
            id="species-123",
            name="Tomato",
            scientific_name="Solanum lycopersicum",
            watering_requirement=WateringRequirement(
                frequency_days=2,
                amount_ml=250,
                drought_tolerance="low"
            )
        )
    
    def test_plant_needs_water_when_dry_too_long(self, tomato_species):
        # Arrange
        plant = Plant(
            name="My Tomato",
            species=tomato_species,
            last_watered=datetime.now() - timedelta(days=3)
        )
        
        # Act
        needs_water = plant.needs_water()
        
        # Assert
        assert needs_water is True
    
    def test_plant_calculates_days_until_next_watering(self, tomato_species):
        # Arrange
        last_watered = datetime.now() - timedelta(days=1)
        plant = Plant(
            name="My Tomato",
            species=tomato_species,
            last_watered=last_watered
        )
        
        # Act
        days_until = plant.days_until_next_watering()
        
        # Assert
        assert days_until == 1
    
    @pytest.mark.parametrize("moisture,expected", [
        (20, True),   # Very dry
        (40, True),   # Dry
        (60, False),  # Adequate
        (80, False),  # Wet
    ])
    def test_plant_needs_water_based_on_moisture(
        self, tomato_species, moisture, expected
    ):
        plant = Plant(name="Test", species=tomato_species)
        assert plant.needs_water_by_moisture(moisture) == expected
```

#### C Unit Test Example
```c
// tests/unit/hardware/test_moisture_sensor.c
#include "unity.h"
#include "moisture_sensor.h"
#include "mock_adc.h"

void setUp(void) {
    mock_adc_init();
    moisture_sensor_init(34);
}

void tearDown(void) {
    mock_adc_cleanup();
}

void test_moisture_reading_maps_correctly(void) {
    // Test dry condition
    mock_adc_set_value(3000);  // Dry value
    float moisture;
    TEST_ASSERT_EQUAL(0, moisture_sensor_read(&moisture));
    TEST_ASSERT_FLOAT_WITHIN(1.0, 0.0, moisture);
    
    // Test wet condition
    mock_adc_set_value(1200);  // Wet value
    TEST_ASSERT_EQUAL(0, moisture_sensor_read(&moisture));
    TEST_ASSERT_FLOAT_WITHIN(1.0, 100.0, moisture);
    
    // Test mid-range
    mock_adc_set_value(2100);  // Mid value
    TEST_ASSERT_EQUAL(0, moisture_sensor_read(&moisture));
    TEST_ASSERT_FLOAT_WITHIN(1.0, 50.0, moisture);
}

void test_moisture_sensor_handles_out_of_range_values(void) {
    float moisture;
    
    // Test below range
    mock_adc_set_value(1000);
    TEST_ASSERT_EQUAL(0, moisture_sensor_read(&moisture));
    TEST_ASSERT_EQUAL_FLOAT(100.0, moisture);
    
    // Test above range
    mock_adc_set_value(4000);
    TEST_ASSERT_EQUAL(0, moisture_sensor_read(&moisture));
    TEST_ASSERT_EQUAL_FLOAT(0.0, moisture);
}
```

### Integration Tests

#### Database Integration Test
```python
# tests/integration/infrastructure/test_plant_repository.py
import pytest
from uuid import uuid4
from datetime import datetime
from testcontainers.postgres import PostgresContainer
from sqlalchemy.ext.asyncio import create_async_engine

from src.infrastructure.database.repositories import PostgresPlantRepository
from src.core.domain.entities import Plant

@pytest.fixture(scope="module")
async def postgres_container():
    with PostgresContainer("timescale/timescaledb:latest-pg15") as postgres:
        yield postgres.get_connection_url()

@pytest.fixture
async def repository(postgres_container):
    engine = create_async_engine(postgres_container.replace("postgresql://", "postgresql+asyncpg://"))
    
    # Run migrations
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield PostgresPlantRepository(engine)
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.mark.asyncio
class TestPostgresPlantRepository:
    async def test_save_and_retrieve_plant(self, repository):
        # Arrange
        plant = Plant(
            id=str(uuid4()),
            name="Integration Test Plant",
            species_id="species-123",
            created_at=datetime.now()
        )
        
        # Act
        await repository.save(plant)
        retrieved = await repository.find_by_id(plant.id)
        
        # Assert
        assert retrieved is not None
        assert retrieved.name == plant.name
        assert retrieved.species_id == plant.species_id
    
    async def test_find_plants_needing_water(self, repository):
        # Arrange - Create plants with different watering needs
        plants = [
            Plant(name="Dry Plant", last_watered=datetime.now() - timedelta(days=5)),
            Plant(name="Recent Plant", last_watered=datetime.now() - timedelta(hours=12)),
            Plant(name="Old Plant", last_watered=datetime.now() - timedelta(days=10))
        ]
        
        for plant in plants:
            await repository.save(plant)
        
        # Act
        needing_water = await repository.find_plants_needing_water(days_threshold=3)
        
        # Assert
        assert len(needing_water) == 2
        assert all(p.name in ["Dry Plant", "Old Plant"] for p in needing_water)
```

#### Hardware Integration Test
```python
# tests/integration/hardware/test_sensor_integration.py
import pytest
import ctypes
from pathlib import Path

from src.infrastructure.hardware.python_bindings import MoistureSensorBinding

@pytest.mark.hardware  # Skip in CI without hardware
class TestHardwareSensorIntegration:
    @pytest.fixture
    def sensor_lib(self):
        lib_path = Path(__file__).parent / "../../../build/libsensors.so"
        return ctypes.CDLL(str(lib_path))
    
    def test_real_moisture_sensor_reading(self, sensor_lib):
        # Initialize sensor
        assert sensor_lib.moisture_sensor_init(34) == 0
        
        # Read multiple times to ensure stability
        readings = []
        for _ in range(10):
            moisture = ctypes.c_float()
            assert sensor_lib.moisture_sensor_read(ctypes.byref(moisture)) == 0
            readings.append(moisture.value)
        
        # Check readings are stable
        avg = sum(readings) / len(readings)
        assert all(abs(r - avg) < 5.0 for r in readings)
```

### End-to-End Tests

#### API E2E Test
```python
# tests/e2e/test_plant_care_workflow.py
import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta

from src.main import app

@pytest.mark.asyncio
class TestPlantCareWorkflow:
    @pytest.fixture
    async def client(self):
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    async def test_complete_plant_care_workflow(self, client):
        # 1. Create a plant
        create_response = await client.post("/api/plants", json={
            "name": "E2E Test Tomato",
            "species_id": "tomato-species-id",
            "location": "Garden Bed A"
        })
        assert create_response.status_code == 201
        plant_id = create_response.json()["id"]
        
        # 2. Simulate sensor readings
        sensor_response = await client.post(f"/api/plants/{plant_id}/readings", json={
            "sensor_type": "moisture",
            "value": 25.0,  # Low moisture
            "timestamp": datetime.now().isoformat()
        })
        assert sensor_response.status_code == 201
        
        # 3. Check watering recommendations
        recommendations = await client.get(f"/api/plants/{plant_id}/care-recommendations")
        assert recommendations.status_code == 200
        assert recommendations.json()["needs_water"] is True
        assert recommendations.json()["urgency"] == "high"
        
        # 4. Trigger watering
        water_response = await client.post(f"/api/plants/{plant_id}/water", json={
            "amount_ml": 250,
            "trigger": "automated"
        })
        assert water_response.status_code == 200
        
        # 5. Verify plant status updated
        status = await client.get(f"/api/plants/{plant_id}")
        assert status.status_code == 200
        assert status.json()["last_watered"] is not None
```

### Performance Tests

#### Load Testing with Locust
```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between
import random

class PlantOSUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Create a plant for this user
        response = self.client.post("/api/plants", json={
            "name": f"Load Test Plant {random.randint(1000, 9999)}",
            "species_id": "tomato-species-id"
        })
        self.plant_id = response.json()["id"]
    
    @task(5)
    def check_plant_status(self):
        self.client.get(f"/api/plants/{self.plant_id}")
    
    @task(3)
    def submit_sensor_reading(self):
        self.client.post(f"/api/plants/{self.plant_id}/readings", json={
            "sensor_type": "moisture",
            "value": random.uniform(20, 80)
        })
    
    @task(1)
    def get_care_recommendations(self):
        self.client.get(f"/api/plants/{self.plant_id}/care-recommendations")
```

### Test Data Management

#### Fixtures and Factories
```python
# tests/fixtures/factories.py
import factory
from factory.alchemy import SQLAlchemyModelFactory
from faker import Faker

from src.infrastructure.database.models import PlantModel, SensorReadingModel

fake = Faker()

class PlantFactory(SQLAlchemyModelFactory):
    class Meta:
        model = PlantModel
        sqlalchemy_session_persistence = "commit"
    
    id = factory.LazyFunction(lambda: str(uuid4()))
    name = factory.LazyAttribute(lambda obj: f"{fake.first_name()}'s {obj.species.common_name}")
    species_id = factory.SubFactory(SpeciesFactory)
    location = factory.Faker("city")
    created_at = factory.Faker("date_time_this_year")
    
    @factory.post_generation
    def sensor_readings(obj, create, extracted, **kwargs):
        if not create:
            return
        
        if extracted:
            for reading in extracted:
                obj.readings.append(reading)

class SensorReadingFactory(SQLAlchemyModelFactory):
    class Meta:
        model = SensorReadingModel
    
    sensor_type = factory.Iterator(["moisture", "temperature", "light", "ph"])
    value = factory.LazyAttribute(
        lambda obj: {
            "moisture": random.uniform(20, 80),
            "temperature": random.uniform(15, 30),
            "light": random.uniform(100, 10000),
            "ph": random.uniform(5.5, 7.5)
        }[obj.sensor_type]
    )
    timestamp = factory.Faker("date_time_between", start_date="-7d", end_date="now")
```

#### Test Database Seeding
```python
# tests/fixtures/seed_test_data.py
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

async def seed_test_database(session: AsyncSession):
    """Seed database with test data for integration tests."""
    
    # Create species
    species_data = [
        {"name": "Tomato", "water_frequency_days": 2, "optimal_moisture": 60},
        {"name": "Cactus", "water_frequency_days": 14, "optimal_moisture": 20},
        {"name": "Fern", "water_frequency_days": 1, "optimal_moisture": 80},
    ]
    
    species = []
    for data in species_data:
        species.append(Species(**data))
    
    session.add_all(species)
    await session.flush()
    
    # Create plants with varied conditions
    plants = []
    for i in range(10):
        plant = Plant(
            name=f"Test Plant {i}",
            species_id=species[i % 3].id,
            last_watered=datetime.now() - timedelta(days=random.randint(0, 10))
        )
        plants.append(plant)
    
    session.add_all(plants)
    await session.commit()
```

### Continuous Integration

#### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: timescale/timescaledb:latest-pg15
        env:
          POSTGRES_PASSWORD: testpass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install C dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y gcc make libunity-dev
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cache/pip
          ~/.cache/pre-commit
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    
    - name: Install Python dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Build C libraries
      run: make -C hardware/
    
    - name: Run linters
      run: |
        ruff check src/ tests/
        mypy src/
        clang-tidy hardware/drivers/*.c
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
        cd hardware && ./run_tests.sh
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://postgres:testpass@localhost/plantos_test
        REDIS_URL: redis://localhost:6379
      run: pytest tests/integration/ -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### Test Reporting

#### Coverage Report Configuration
```ini
# .coveragerc
[run]
source = src
omit = 
    */tests/*
    */migrations/*
    */__init__.py
    */config/*

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

#### Test Report Generation
```python
# scripts/generate_test_report.py
#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from datetime import datetime

def generate_test_report():
    """Generate comprehensive test report from pytest JSON output."""
    
    report_path = Path("test-results.json")
    if not report_path.exists():
        print("No test results found. Run: pytest --json-report")
        sys.exit(1)
    
    with open(report_path) as f:
        data = json.load(f)
    
    summary = data["summary"]
    
    report = f"""
# PlantOS Test Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Tests: {summary['total']}
- Passed: {summary['passed']}
- Failed: {summary['failed']}
- Skipped: {summary['skipped']}
- Duration: {summary['duration']:.2f}s

## Coverage
- Line Coverage: {data['coverage']['percent_covered']:.1f}%
- Branch Coverage: {data['coverage']['percent_branch_covered']:.1f}%

## Failed Tests
"""
    
    for test in data["tests"]:
        if test["outcome"] == "failed":
            report += f"\n### {test['nodeid']}\n"
            report += f"```\n{test['call']['longrepr']}\n```\n"
    
    with open("TEST_REPORT.md", "w") as f:
        f.write(report)
    
    print("Test report generated: TEST_REPORT.md")

if __name__ == "__main__":
    generate_test_report()
```