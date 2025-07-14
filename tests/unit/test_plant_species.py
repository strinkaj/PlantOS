"""
Unit tests for PlantSpecies entity.
"""
import pytest
from decimal import Decimal
from uuid import uuid4

from src.core.domain.entities import PlantSpecies, create_plant_species
from src.shared.types import PlantSpeciesID
from src.shared.exceptions import ValidationError


class TestPlantSpecies:
    """Test cases for PlantSpecies entity."""
    
    def test_create_plant_species_valid(self):
        """Test creating a valid plant species."""
        species_id = PlantSpeciesID(uuid4())
        species = create_plant_species(
            id=species_id,
            name="Basil",
            scientific_name="Ocimum basilicum",
            optimal_moisture_min=Decimal("50.0"),
            optimal_moisture_max=Decimal("70.0"),
            optimal_temperature_min=Decimal("20.0"),
            optimal_temperature_max=Decimal("30.0"),
            optimal_light_hours=Decimal("6.0"),
            watering_frequency_hours=24,
            fertilizing_frequency_days=7
        )
        
        assert species.id == species_id
        assert species.name == "Basil"
        assert species.scientific_name == "Ocimum basilicum"
        assert species.optimal_moisture_min == Decimal("50.0")
        assert species.optimal_moisture_max == Decimal("70.0")
        assert species.optimal_temperature_min == Decimal("20.0")
        assert species.optimal_temperature_max == Decimal("30.0")
        assert species.optimal_light_hours == Decimal("6.0")
        assert species.watering_frequency_hours == 24
        assert species.fertilizing_frequency_days == 7
        assert species.notes is None
    
    def test_create_plant_species_with_notes(self):
        """Test creating a plant species with notes."""
        species_id = PlantSpeciesID(uuid4())
        species = create_plant_species(
            id=species_id,
            name="Mint",
            scientific_name="Mentha",
            optimal_moisture_min=Decimal("60.0"),
            optimal_moisture_max=Decimal("80.0"),
            optimal_temperature_min=Decimal("15.0"),
            optimal_temperature_max=Decimal("25.0"),
            optimal_light_hours=Decimal("4.0"),
            watering_frequency_hours=12,
            fertilizing_frequency_days=21,
            notes="Prefers partial shade"
        )
        
        assert species.notes == "Prefers partial shade"
    
    def test_plant_species_invalid_moisture_range(self):
        """Test that moisture min must be less than max."""
        with pytest.raises(ValidationError, match="Moisture min must be less than max"):
            create_plant_species(
                id=PlantSpeciesID(uuid4()),
                name="Invalid Species",
                scientific_name="Invalid species",
                optimal_moisture_min=Decimal("70.0"),  # Greater than max
                optimal_moisture_max=Decimal("50.0"),
                optimal_temperature_min=Decimal("20.0"),
                optimal_temperature_max=Decimal("30.0"),
                optimal_light_hours=Decimal("6.0"),
                watering_frequency_hours=24,
                fertilizing_frequency_days=7
            )
    
    def test_plant_species_invalid_temperature_range(self):
        """Test that temperature min must be less than max."""
        with pytest.raises(ValidationError, match="Temperature min must be less than max"):
            create_plant_species(
                id=PlantSpeciesID(uuid4()),
                name="Invalid Species",
                scientific_name="Invalid species",
                optimal_moisture_min=Decimal("50.0"),
                optimal_moisture_max=Decimal("70.0"),
                optimal_temperature_min=Decimal("30.0"),  # Greater than max
                optimal_temperature_max=Decimal("20.0"),
                optimal_light_hours=Decimal("6.0"),
                watering_frequency_hours=24,
                fertilizing_frequency_days=7
            )
    
    def test_plant_species_negative_values(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError, match="Moisture values must be non-negative"):
            create_plant_species(
                id=PlantSpeciesID(uuid4()),
                name="Invalid Species",
                scientific_name="Invalid species",
                optimal_moisture_min=Decimal("-10.0"),
                optimal_moisture_max=Decimal("70.0"),
                optimal_temperature_min=Decimal("20.0"),
                optimal_temperature_max=Decimal("30.0"),
                optimal_light_hours=Decimal("6.0"),
                watering_frequency_hours=24,
                fertilizing_frequency_days=7
            )
    
    def test_plant_species_invalid_light_hours(self):
        """Test that light hours must be between 0 and 24."""
        with pytest.raises(ValidationError, match="Light hours must be between 0 and 24"):
            create_plant_species(
                id=PlantSpeciesID(uuid4()),
                name="Invalid Species",
                scientific_name="Invalid species",
                optimal_moisture_min=Decimal("50.0"),
                optimal_moisture_max=Decimal("70.0"),
                optimal_temperature_min=Decimal("20.0"),
                optimal_temperature_max=Decimal("30.0"),
                optimal_light_hours=Decimal("25.0"),  # More than 24
                watering_frequency_hours=24,
                fertilizing_frequency_days=7
            )
    
    def test_plant_species_invalid_frequencies(self):
        """Test that frequencies must be positive."""
        with pytest.raises(ValidationError, match="Watering frequency must be positive"):
            create_plant_species(
                id=PlantSpeciesID(uuid4()),
                name="Invalid Species",
                scientific_name="Invalid species",
                optimal_moisture_min=Decimal("50.0"),
                optimal_moisture_max=Decimal("70.0"),
                optimal_temperature_min=Decimal("20.0"),
                optimal_temperature_max=Decimal("30.0"),
                optimal_light_hours=Decimal("6.0"),
                watering_frequency_hours=0,  # Zero is invalid
                fertilizing_frequency_days=7
            )
    
    def test_plant_species_empty_name(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError, match="Name cannot be empty"):
            create_plant_species(
                id=PlantSpeciesID(uuid4()),
                name="",  # Empty name
                scientific_name="Invalid species",
                optimal_moisture_min=Decimal("50.0"),
                optimal_moisture_max=Decimal("70.0"),
                optimal_temperature_min=Decimal("20.0"),
                optimal_temperature_max=Decimal("30.0"),
                optimal_light_hours=Decimal("6.0"),
                watering_frequency_hours=24,
                fertilizing_frequency_days=7
            )
    
    def test_plant_species_moisture_range_methods(self, sample_plant_species):
        """Test methods for checking moisture values."""
        # Within range
        assert sample_plant_species.is_moisture_in_range(Decimal("50.0")) is True
        assert sample_plant_species.is_moisture_in_range(Decimal("40.0")) is True  # Min
        assert sample_plant_species.is_moisture_in_range(Decimal("60.0")) is True  # Max
        
        # Outside range
        assert sample_plant_species.is_moisture_in_range(Decimal("30.0")) is False
        assert sample_plant_species.is_moisture_in_range(Decimal("70.0")) is False
        
        # Critical (below 20% of min)
        assert sample_plant_species.is_moisture_critical(Decimal("5.0")) is True
        assert sample_plant_species.is_moisture_critical(Decimal("10.0")) is False
    
    def test_plant_species_temperature_range_methods(self, sample_plant_species):
        """Test methods for checking temperature values."""
        # Within range
        assert sample_plant_species.is_temperature_in_range(Decimal("20.0")) is True
        assert sample_plant_species.is_temperature_in_range(Decimal("18.0")) is True  # Min
        assert sample_plant_species.is_temperature_in_range(Decimal("24.0")) is True  # Max
        
        # Outside range
        assert sample_plant_species.is_temperature_in_range(Decimal("15.0")) is False
        assert sample_plant_species.is_temperature_in_range(Decimal("30.0")) is False