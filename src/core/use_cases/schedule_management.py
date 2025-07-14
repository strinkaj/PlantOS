"""
Schedule management use cases.

These use cases handle watering schedule creation, updates, and 
manual watering operations for automated plant care.
"""

from datetime import datetime, timedelta, time
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

import structlog

from src.core.domain.entities import WateringSchedule, Plant
from src.core.domain.repositories import WateringScheduleRepository, PlantRepository
from src.shared.types import PlantID, ScheduleID, ScheduleType
from src.shared.exceptions import (
    PlantNotFoundError,
    ScheduleNotFoundError,
    ValidationError,
    ScheduleConflictError
)
from src.shared.contracts import require, ensure

logger = structlog.get_logger(__name__)


class CreateScheduleUseCase:
    """Use case for creating watering schedules for plants."""
    
    def __init__(
        self,
        schedule_repo: WateringScheduleRepository,
        plant_repo: PlantRepository
    ):
        self.schedule_repo = schedule_repo
        self.plant_repo = plant_repo
        
    @require(lambda plant_id: plant_id is not None, "Plant ID cannot be None")
    @require(lambda plant_id, interval_days: interval_days > 0, "Interval days must be positive")
    @require(lambda plant_id, interval_days, duration_seconds: duration_seconds > 0,
             "Duration seconds must be positive")
    @require(lambda plant_id, interval_days, duration_seconds: duration_seconds <= 600,
             "Duration cannot exceed 600 seconds (10 minutes)")
    async def execute(
        self,
        plant_id: UUID,
        schedule_type: str = "automatic",
        start_date: Optional[datetime] = None,
        time_of_day: str = "08:00",
        interval_days: int = 1,
        duration_seconds: int = 30,
        amount_ml: Optional[float] = None,
        enabled: bool = True
    ) -> WateringSchedule:
        """
        Create a new watering schedule for a plant.
        
        Args:
            plant_id: ID of the plant to schedule watering for
            schedule_type: Type of schedule (automatic, manual, conditional)
            start_date: When the schedule should start (defaults to now)
            time_of_day: Time of day to water (HH:MM format)
            interval_days: Days between watering
            duration_seconds: How long to water for
            amount_ml: Optional specific amount in milliliters
            enabled: Whether the schedule is active
            
        Returns:
            Created watering schedule entity
            
        Raises:
            PlantNotFoundError: If plant doesn't exist
            ValidationError: If input validation fails
            ScheduleConflictError: If schedule conflicts with existing ones
        """
        logger.info(
            "Creating watering schedule",
            plant_id=str(plant_id),
            schedule_type=schedule_type,
            interval_days=interval_days
        )
        
        # Validate plant exists
        plant = await self.plant_repo.get_by_id(PlantID(plant_id))
        if not plant:
            raise PlantNotFoundError(f"Plant with ID {plant_id} not found")
            
        # Validate schedule type
        allowed_types = ['automatic', 'manual', 'conditional']
        if schedule_type not in allowed_types:
            raise ValidationError(f"Schedule type must be one of: {', '.join(allowed_types)}")
            
        # Validate time format
        try:
            hour, minute = map(int, time_of_day.split(':'))
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                raise ValueError()
        except (ValueError, AttributeError):
            raise ValidationError("Time must be in HH:MM format")
            
        # Check for schedule conflicts if automatic
        if schedule_type == "automatic" and enabled:
            existing_schedules = await self.schedule_repo.get_by_plant_id(PlantID(plant_id))
            active_schedules = [s for s in existing_schedules if s.enabled and s.schedule_type.value == "automatic"]
            
            if active_schedules:
                # Check for time conflicts (within 1 hour of each other)
                schedule_time = time(hour, minute)
                for existing in active_schedules:
                    existing_time = time(*map(int, existing.time_of_day.split(':')))
                    time_diff = abs((datetime.combine(datetime.today(), schedule_time) - 
                                   datetime.combine(datetime.today(), existing_time)).total_seconds())
                    if time_diff < 3600:  # Less than 1 hour apart
                        raise ScheduleConflictError(
                            f"Schedule conflicts with existing schedule at {existing.time_of_day}"
                        )
                        
        # Calculate next execution time
        start_datetime = start_date or datetime.utcnow()
        schedule_time = time(hour, minute)
        next_execution = datetime.combine(start_datetime.date(), schedule_time)
        
        # If the time has already passed today, schedule for tomorrow
        if next_execution <= datetime.utcnow():
            next_execution += timedelta(days=1)
            
        # Create schedule entity
        schedule = WateringSchedule(
            id=ScheduleID(uuid4()),
            plant_id=PlantID(plant_id),
            schedule_type=ScheduleType(schedule_type),
            start_date=start_datetime,
            time_of_day=time_of_day,
            interval_days=interval_days,
            duration_seconds=duration_seconds,
            amount_ml=amount_ml,
            enabled=enabled,
            last_executed=None,
            next_execution=next_execution if enabled else None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Save to repository
        created_schedule = await self.schedule_repo.create(schedule)
        
        logger.info(
            "Watering schedule created successfully",
            schedule_id=str(created_schedule.id),
            plant_id=str(plant_id),
            next_execution=next_execution.isoformat() if next_execution else None
        )
        
        return created_schedule


class UpdateScheduleUseCase:
    """Use case for updating existing watering schedules."""
    
    def __init__(self, schedule_repo: WateringScheduleRepository):
        self.schedule_repo = schedule_repo
        
    @require(lambda schedule_id: schedule_id is not None, "Schedule ID cannot be None")
    async def execute(
        self,
        schedule_id: UUID,
        time_of_day: Optional[str] = None,
        interval_days: Optional[int] = None,
        duration_seconds: Optional[int] = None,
        amount_ml: Optional[float] = None,
        enabled: Optional[bool] = None
    ) -> WateringSchedule:
        """
        Update an existing watering schedule.
        
        Args:
            schedule_id: ID of the schedule to update
            time_of_day: Optional new time of day
            interval_days: Optional new interval
            duration_seconds: Optional new duration
            amount_ml: Optional new amount
            enabled: Optional new enabled status
            
        Returns:
            Updated watering schedule entity
            
        Raises:
            ScheduleNotFoundError: If schedule doesn't exist
            ValidationError: If input validation fails
        """
        logger.info("Updating watering schedule", schedule_id=str(schedule_id))
        
        # Get existing schedule
        schedule = await self.schedule_repo.get_by_id(ScheduleID(schedule_id))
        if not schedule:
            raise ScheduleNotFoundError(f"Schedule with ID {schedule_id} not found")
            
        # Update fields with validation
        if time_of_day is not None:
            try:
                hour, minute = map(int, time_of_day.split(':'))
                if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                    raise ValueError()
                schedule.time_of_day = time_of_day
            except (ValueError, AttributeError):
                raise ValidationError("Time must be in HH:MM format")
                
        if interval_days is not None:
            if interval_days <= 0:
                raise ValidationError("Interval days must be positive")
            schedule.interval_days = interval_days
            
        if duration_seconds is not None:
            if duration_seconds <= 0 or duration_seconds > 600:
                raise ValidationError("Duration must be between 1 and 600 seconds")
            schedule.duration_seconds = duration_seconds
            
        if amount_ml is not None:
            if amount_ml <= 0:
                raise ValidationError("Amount must be positive")
            schedule.amount_ml = amount_ml
            
        if enabled is not None:
            schedule.enabled = enabled
            
        # Recalculate next execution if schedule was modified
        if any(param is not None for param in [time_of_day, interval_days, enabled]):
            if schedule.enabled:
                # Calculate next execution based on updated parameters
                hour, minute = map(int, schedule.time_of_day.split(':'))
                schedule_time = time(hour, minute)
                today = datetime.utcnow().date()
                next_execution = datetime.combine(today, schedule_time)
                
                # If time has passed today, schedule for tomorrow
                if next_execution <= datetime.utcnow():
                    next_execution += timedelta(days=1)
                    
                schedule.next_execution = next_execution
            else:
                schedule.next_execution = None
                
        schedule.updated_at = datetime.utcnow()
        
        # Save changes
        updated_schedule = await self.schedule_repo.update(schedule)
        
        logger.info("Watering schedule updated successfully", schedule_id=str(schedule_id))
        return updated_schedule


class ListSchedulesUseCase:
    """Use case for listing watering schedules with filtering."""
    
    def __init__(self, schedule_repo: WateringScheduleRepository):
        self.schedule_repo = schedule_repo
        
    @require(lambda offset, limit: offset >= 0, "Offset must be non-negative")
    @require(lambda offset, limit: limit > 0, "Limit must be positive")
    @require(lambda offset, limit: limit <= 1000, "Limit cannot exceed 1000")
    async def execute(
        self,
        offset: int = 0,
        limit: int = 50,
        plant_id: Optional[UUID] = None,
        enabled_only: bool = False
    ) -> tuple[List[WateringSchedule], int]:
        """
        List watering schedules with pagination and filtering.
        
        Args:
            offset: Number of schedules to skip
            limit: Maximum number of schedules to return
            plant_id: Optional plant ID filter
            enabled_only: Whether to return only enabled schedules
            
        Returns:
            Tuple of (schedules list, total count)
        """
        logger.info(
            "Listing watering schedules",
            offset=offset,
            limit=limit,
            plant_id=str(plant_id) if plant_id else None,
            enabled_only=enabled_only
        )
        
        if plant_id:
            schedules = await self.schedule_repo.get_by_plant_id(PlantID(plant_id))
        else:
            schedules = await self.schedule_repo.get_all()
            
        # Apply enabled filter
        if enabled_only:
            schedules = [s for s in schedules if s.enabled]
            
        # Apply pagination
        total_count = len(schedules)
        paginated_schedules = schedules[offset:offset + limit]
        
        logger.info(
            "Watering schedules listed successfully",
            returned_count=len(paginated_schedules),
            total_count=total_count
        )
        
        return paginated_schedules, total_count


class TriggerManualWateringUseCase:
    """Use case for triggering immediate manual watering."""
    
    def __init__(
        self,
        plant_repo: PlantRepository,
        schedule_repo: WateringScheduleRepository
    ):
        self.plant_repo = plant_repo
        self.schedule_repo = schedule_repo
        
    @require(lambda plant_id: plant_id is not None, "Plant ID cannot be None")
    @require(lambda plant_id, duration_seconds: duration_seconds > 0,
             "Duration seconds must be positive")
    @require(lambda plant_id, duration_seconds: duration_seconds <= 600,
             "Duration cannot exceed 600 seconds (10 minutes)")
    async def execute(
        self,
        plant_id: UUID,
        duration_seconds: int,
        amount_ml: Optional[float] = None,
        reason: str = "manual_trigger"
    ) -> Dict[str, Any]:
        """
        Trigger immediate manual watering for a plant.
        
        Args:
            plant_id: ID of the plant to water
            duration_seconds: How long to water for
            amount_ml: Optional specific amount in milliliters
            reason: Reason for manual watering
            
        Returns:
            Dictionary with watering operation details
            
        Raises:
            PlantNotFoundError: If plant doesn't exist
            ValidationError: If input validation fails
        """
        logger.info(
            "Triggering manual watering",
            plant_id=str(plant_id),
            duration_seconds=duration_seconds,
            reason=reason
        )
        
        # Validate plant exists
        plant = await self.plant_repo.get_by_id(PlantID(plant_id))
        if not plant:
            raise PlantNotFoundError(f"Plant with ID {plant_id} not found")
            
        # Validate amount if provided
        if amount_ml is not None and amount_ml <= 0:
            raise ValidationError("Amount must be positive")
            
        # Create manual watering record
        watering_operation = {
            "operation_id": str(uuid4()),
            "plant_id": str(plant_id),
            "plant_name": plant.name,
            "type": "manual",
            "duration_seconds": duration_seconds,
            "amount_ml": amount_ml,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "initiated"
        }
        
        # TODO: Trigger actual hardware watering operation
        # This would interface with the hardware abstraction layer to:
        # 1. Check pump availability
        # 2. Start watering operation
        # 3. Monitor operation progress
        # 4. Record completion/failure
        
        # For now, simulate successful operation
        watering_operation["status"] = "completed"
        watering_operation["completed_at"] = datetime.utcnow().isoformat()
        
        # Update any existing automatic schedules to prevent immediate re-watering
        schedules = await self.schedule_repo.get_by_plant_id(PlantID(plant_id))
        for schedule in schedules:
            if schedule.enabled and schedule.schedule_type.value == "automatic":
                # Update last executed time to prevent duplicate watering
                schedule.last_executed = datetime.utcnow()
                
                # Calculate next execution
                next_execution = datetime.utcnow() + timedelta(days=schedule.interval_days)
                hour, minute = map(int, schedule.time_of_day.split(':'))
                schedule_time = time(hour, minute)
                schedule.next_execution = datetime.combine(next_execution.date(), schedule_time)
                
                await self.schedule_repo.update(schedule)
                
        logger.info(
            "Manual watering completed successfully",
            plant_id=str(plant_id),
            operation_id=watering_operation["operation_id"]
        )
        
        return watering_operation