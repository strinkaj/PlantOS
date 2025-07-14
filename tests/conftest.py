"""
Global pytest configuration and fixtures for PlantOS tests.
"""
import pytest
from datetime import datetime, timezone
from uuid import UUID, uuid4
from decimal import Decimal

from src.core.domain.entities import (
    PlantSpecies,
    Plant,
    Sensor,
    SensorReading,
    create_plant_species,
    create_plant,
    create_sensor,
    create_sensor_reading
)
from src.shared.types import (
    PlantSpeciesID,
    PlantID,
    SensorID,
    SensorType,
    CareEventType,
    PlantStatus,
    ReadingID
)


@pytest.fixture
def sample_plant_species_id() -> PlantSpeciesID:
    """Sample plant species ID for testing."""
    return PlantSpeciesID(uuid4())


@pytest.fixture
def sample_plant_id() -> PlantID:
    """Sample plant ID for testing."""
    return PlantID(uuid4())


@pytest.fixture
def sample_sensor_id() -> SensorID:
    """Sample sensor ID for testing."""
    return SensorID(uuid4())


@pytest.fixture
def sample_reading_id() -> ReadingID:
    """Sample reading ID for testing."""
    return ReadingID(uuid4())


@pytest.fixture
def sample_plant_species(sample_plant_species_id: PlantSpeciesID) -> PlantSpecies:
    """Sample plant species for testing."""
    return create_plant_species(
        id=sample_plant_species_id,
        name="Tomato",
        scientific_name="Solanum lycopersicum",
        optimal_moisture_min=Decimal("40.0"),
        optimal_moisture_max=Decimal("60.0"),
        optimal_temperature_min=Decimal("18.0"),
        optimal_temperature_max=Decimal("24.0"),
        optimal_light_hours=Decimal("8.0"),
        watering_frequency_hours=48,
        fertilizing_frequency_days=14,
        notes="Test tomato plant species"
    )


@pytest.fixture
def sample_plant(sample_plant_id: PlantID, sample_plant_species_id: PlantSpeciesID) -> Plant:
    """Sample plant for testing."""
    return create_plant(
        id=sample_plant_id,
        species_id=sample_plant_species_id,
        name="Test Tomato",
        location="Greenhouse A",
        notes="Test plant"
    )


@pytest.fixture
def sample_sensor(sample_sensor_id: SensorID, sample_plant_id: PlantID) -> Sensor:
    """Sample sensor for testing."""
    return create_sensor(
        id=sample_sensor_id,
        type=SensorType.MOISTURE,
        name="Moisture Sensor 1",
        model="MS-100",
        pin=17,
        plant_id=sample_plant_id,
        calibration_offset=Decimal("0.5"),
        min_value=Decimal("0.0"),
        max_value=Decimal("100.0"),
        unit="%",
        update_interval_seconds=300
    )


@pytest.fixture
def sample_sensor_reading(
    sample_reading_id: ReadingID, 
    sample_sensor_id: SensorID,
    sample_plant_id: PlantID
) -> SensorReading:
    """Sample sensor reading for testing."""
    return create_sensor_reading(
        id=sample_reading_id,
        sensor_id=sample_sensor_id,
        plant_id=sample_plant_id,
        value=Decimal("55.5"),
        unit="%",
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_datetime(monkeypatch):
    """Mock datetime for consistent testing."""
    class MockDatetime:
        @staticmethod
        def now(tz=None):
            return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    monkeypatch.setattr("src.core.domain.entities.datetime", MockDatetime)
    return MockDatetime.now(timezone.utc)