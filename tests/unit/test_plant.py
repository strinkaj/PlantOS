"""
Unit tests for Plant entity.
"""
import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from src.core.domain.entities import Plant, create_plant, PlantHealthAssessment
from src.shared.types import PlantID, PlantSpeciesID, PlantStatus, ReadingID
from src.shared.exceptions import ValidationError, BusinessRuleViolation


class TestPlant:
    """Test cases for Plant entity."""
    
    def test_create_plant_valid(self):
        """Test creating a valid plant."""
        plant_id = PlantID(uuid4())
        species_id = PlantSpeciesID(uuid4())
        
        plant = create_plant(
            id=plant_id,
            species_id=species_id,
            name="My Tomato Plant",
            location="Balcony",
            notes="Planted from seed"
        )
        
        assert plant.id == plant_id
        assert plant.species_id == species_id
        assert plant.name == "My Tomato Plant"
        assert plant.location == "Balcony"
        assert plant.notes == "Planted from seed"
        assert plant.status == PlantStatus.HEALTHY
        assert plant.last_watered is None
        assert plant.health_score is None
        assert isinstance(plant.planted_date, datetime)
        assert plant.monitoring_started is None
    
    def test_create_plant_minimal(self):
        """Test creating a plant with minimal required fields."""
        plant_id = PlantID(uuid4())
        species_id = PlantSpeciesID(uuid4())
        
        plant = create_plant(
            id=plant_id,
            species_id=species_id,
            name="Basic Plant"
        )
        
        assert plant.name == "Basic Plant"
        assert plant.location is None
        assert plant.notes is None
        assert plant.status == PlantStatus.HEALTHY
    
    def test_plant_empty_name(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError, match="Plant name cannot be empty"):
            create_plant(
                id=PlantID(uuid4()),
                species_id=PlantSpeciesID(uuid4()),
                name=""
            )
    
    def test_plant_calculate_health_score_no_data(self, sample_plant, sample_plant_species):
        """Test health score calculation with no sensor data."""
        score = sample_plant.calculate_health_score(
            recent_readings=[],
            species=sample_plant_species
        )
        
        # No data should return neutral score
        assert score == Decimal("50.0")
    
    def test_plant_calculate_health_score_optimal(self, sample_plant, sample_plant_species):
        """Test health score calculation with optimal readings."""
        from src.core.domain.entities import create_sensor_reading
        from src.shared.types import SensorID, SensorType
        
        # Create readings within optimal range
        readings = [
            create_sensor_reading(
                id=ReadingID(uuid4()),
                sensor_id=SensorID(uuid4()),
                plant_id=sample_plant.id,
                value=Decimal("50.0"),  # Within 40-60 range
                unit="%",
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        score = sample_plant.calculate_health_score(
            recent_readings=readings,
            species=sample_plant_species
        )
        
        # Optimal readings should give high score
        assert score > Decimal("80.0")
    
    def test_plant_calculate_health_score_suboptimal(self, sample_plant, sample_plant_species):
        """Test health score calculation with suboptimal readings."""
        from src.core.domain.entities import create_sensor_reading
        from src.shared.types import SensorID
        
        # Create readings outside optimal range
        readings = [
            create_sensor_reading(
                id=ReadingID(uuid4()),
                sensor_id=SensorID(uuid4()),
                plant_id=sample_plant.id,
                value=Decimal("25.0"),  # Below optimal 40-60 range
                unit="%",
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        score = sample_plant.calculate_health_score(
            recent_readings=readings,
            species=sample_plant_species
        )
        
        # Suboptimal readings should give lower score
        assert score < Decimal("50.0")
    
    def test_plant_needs_watering_never_watered(self, sample_plant, sample_plant_species):
        """Test that plant needs watering if never watered."""
        needs_water, reason = sample_plant.needs_watering(
            species=sample_plant_species,
            current_moisture=None
        )
        
        assert needs_water is True
        assert "Never watered" in reason
    
    def test_plant_needs_watering_schedule_based(self, sample_plant, sample_plant_species):
        """Test schedule-based watering need."""
        # Set last watered to 3 days ago (species requires every 48 hours)
        sample_plant.last_watered = datetime.now(timezone.utc) - timedelta(days=3)
        
        needs_water, reason = sample_plant.needs_watering(
            species=sample_plant_species,
            current_moisture=None
        )
        
        assert needs_water is True
        assert "Schedule" in reason
    
    def test_plant_needs_watering_moisture_based(self, sample_plant, sample_plant_species):
        """Test moisture-based watering need."""
        # Set last watered recently
        sample_plant.last_watered = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Low moisture should trigger watering
        needs_water, reason = sample_plant.needs_watering(
            species=sample_plant_species,
            current_moisture=Decimal("30.0")  # Below optimal 40-60
        )
        
        assert needs_water is True
        assert "Low moisture" in reason
    
    def test_plant_needs_watering_critical_moisture(self, sample_plant, sample_plant_species):
        """Test critical moisture overrides schedule."""
        # Set last watered recently
        sample_plant.last_watered = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Critical moisture should always trigger watering
        needs_water, reason = sample_plant.needs_watering(
            species=sample_plant_species,
            current_moisture=Decimal("5.0")  # Critical level
        )
        
        assert needs_water is True
        assert "Critical moisture" in reason
    
    def test_plant_does_not_need_watering(self, sample_plant, sample_plant_species):
        """Test when plant doesn't need watering."""
        # Set last watered recently
        sample_plant.last_watered = datetime.now(timezone.utc) - timedelta(hours=12)
        
        # Good moisture level
        needs_water, reason = sample_plant.needs_watering(
            species=sample_plant_species,
            current_moisture=Decimal("50.0")  # Optimal level
        )
        
        assert needs_water is False
        assert reason == "Adequate moisture and within schedule"
    
    def test_plant_mark_as_watered(self, sample_plant, mock_datetime):
        """Test marking plant as watered."""
        assert sample_plant.last_watered is None
        
        sample_plant.mark_as_watered()
        
        assert sample_plant.last_watered == mock_datetime
    
    def test_plant_update_health_assessment(self, sample_plant):
        """Test updating plant health assessment."""
        assessment = PlantHealthAssessment(
            id=uuid4(),
            plant_id=sample_plant.id,
            timestamp=datetime.now(timezone.utc),
            overall_score=Decimal("85.0"),
            moisture_score=Decimal("90.0"),
            temperature_score=Decimal("80.0"),
            light_score=Decimal("85.0"),
            recommendations=["All parameters within optimal range"],
            issues_detected=[]
        )
        
        sample_plant.update_health_assessment(assessment)
        
        assert sample_plant.health_score == Decimal("85.0")
        assert sample_plant.status == PlantStatus.HEALTHY
    
    def test_plant_update_health_assessment_warning_status(self, sample_plant):
        """Test health assessment updates status to warning."""
        assessment = PlantHealthAssessment(
            id=uuid4(),
            plant_id=sample_plant.id,
            timestamp=datetime.now(timezone.utc),
            overall_score=Decimal("65.0"),  # Between 60-80
            moisture_score=Decimal("60.0"),
            temperature_score=Decimal("70.0"),
            light_score=Decimal("65.0"),
            recommendations=["Increase watering frequency"],
            issues_detected=["Moisture slightly below optimal"]
        )
        
        sample_plant.update_health_assessment(assessment)
        
        assert sample_plant.health_score == Decimal("65.0")
        assert sample_plant.status == PlantStatus.WARNING
    
    def test_plant_update_health_assessment_critical_status(self, sample_plant):
        """Test health assessment updates status to critical."""
        assessment = PlantHealthAssessment(
            id=uuid4(),
            plant_id=sample_plant.id,
            timestamp=datetime.now(timezone.utc),
            overall_score=Decimal("35.0"),  # Below 60
            moisture_score=Decimal("20.0"),
            temperature_score=Decimal("50.0"),
            light_score=Decimal("35.0"),
            recommendations=["Immediate watering required"],
            issues_detected=["Critical moisture level", "Low light exposure"]
        )
        
        sample_plant.update_health_assessment(assessment)
        
        assert sample_plant.health_score == Decimal("35.0")
        assert sample_plant.status == PlantStatus.CRITICAL
    
    def test_plant_start_monitoring(self, sample_plant, mock_datetime):
        """Test starting monitoring for a plant."""
        assert sample_plant.monitoring_started is None
        
        sample_plant.start_monitoring()
        
        assert sample_plant.monitoring_started == mock_datetime
    
    def test_plant_start_monitoring_already_started(self, sample_plant):
        """Test that monitoring can't be started twice."""
        sample_plant.start_monitoring()
        
        with pytest.raises(BusinessRuleViolation, match="Monitoring already started"):
            sample_plant.start_monitoring()
    
    def test_plant_stop_monitoring(self, sample_plant):
        """Test stopping monitoring for a plant."""
        sample_plant.start_monitoring()
        assert sample_plant.monitoring_started is not None
        
        sample_plant.stop_monitoring()
        
        assert sample_plant.monitoring_started is None
    
    def test_plant_is_monitoring_active(self, sample_plant):
        """Test checking if monitoring is active."""
        assert sample_plant.is_monitoring_active() is False
        
        sample_plant.start_monitoring()
        assert sample_plant.is_monitoring_active() is True
        
        sample_plant.stop_monitoring()
        assert sample_plant.is_monitoring_active() is False