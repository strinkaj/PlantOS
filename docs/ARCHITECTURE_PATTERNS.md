# PlantOS - Comprehensive Architecture & Engineering Patterns

This document outlines the complete architectural patterns, principles, and advanced engineering practices that PlantOS implements for optimal resilience, reusability, maintainability, and operational excellence.

## Table of Contents

1. [The Pragmatic Programmer Principles](#the-pragmatic-programmer-principles)
2. [Classic Design Patterns](#classic-design-patterns-implementation)
3. [SOLID Principles](#solid-principles-in-plantos)
4. [Domain-Driven Design](#domain-driven-design-ddd-implementation)
5. [Resilience Engineering](#resilience-engineering-principles)
6. [Advanced Microservices Patterns](#advanced-microservices-patterns)
7. [Security-First Principles](#security-first-principles)
8. [Performance Engineering](#performance-engineering)
9. [Polyglot Architecture Patterns](#polyglot-architecture-patterns)
10. [Testing Patterns](#testing-patterns)
11. [Configuration Management](#configuration-management-excellence)

## The Pragmatic Programmer Principles

### 1. DRY (Don't Repeat Yourself)
**Implementation in PlantOS:**
- **Shared Type System**: `src/shared/types.py` provides type definitions used across all services
- **Hardware Abstraction**: Single interface definition used by all hardware implementations
- **Configuration Templates**: Reusable environment and compose configurations
- **Cross-Language Standards**: Consistent logging, error handling, and metrics patterns

```python
# Example: Reusable sensor reading pattern
@dataclass
class SensorReading:
    """Single definition used across Python, Go (via gRPC), and Rust (via FFI)"""
    sensor_id: SensorID
    value: Decimal
    timestamp: datetime
    quality_score: float
```

### 2. Orthogonality - Independent, Loosely Coupled Components
**Implementation:**
- **Microservices**: Python API, Go streaming, Rust drivers, Julia analytics
- **Message Queues**: Kafka decouples producers from consumers
- **Hardware Abstraction**: Sensor changes don't affect business logic
- **Database Layer**: Repository pattern isolates data access

### 3. Reversibility - Avoid Irreversible Decisions
**Implementation:**
- **Multiple Container Runtimes**: Podman (preferred) or Docker fallback
- **Database Agnostic**: SQLAlchemy abstracts PostgreSQL specifics
- **Cloud Agnostic**: MinIO provides S3-compatible interface
- **Hardware Abstraction**: Easy switch between real/mock hardware

### 4. Tracer Bullets - Build End-to-End Skeleton First
**Implementation Strategy:**
```
Phase 1: Minimal viable API → Database → Hardware mock → Response
Phase 2: Add real hardware drivers
Phase 3: Add streaming and analytics
```

### 5. Contract Programming - Preconditions, Postconditions, Invariants
**Implementation:**
```python
class PlantCareService:
    async def water_plant(self, plant_id: PlantID, amount: WaterAmount) -> bool:
        # Preconditions
        assert amount > 0, "Water amount must be positive"
        assert plant_id is not None, "Plant ID required"
        
        # Business logic with invariants
        plant = await self.get_plant(plant_id)
        assert plant.status == "active", "Cannot water inactive plant"
        
        # Postconditions enforced by return type and validation
        success = await self.hardware.water(plant_id, amount)
        if success:
            await self.log_watering_event(plant_id, amount)
        return success
```

### 6. Crash Early - Fail Fast Principle
**Implementation:**
- **Type Safety**: mypy catches type errors at development time
- **Pydantic Validation**: API inputs validated immediately
- **Hardware Safety**: Rust drivers fail fast on invalid operations
- **Configuration Validation**: Environment variables validated at startup

### 7. Decoupling and Law of Demeter
**Implementation:**
- **Repository Pattern**: Services only know about repositories, not database details
- **Event-Driven**: Components communicate via events, not direct calls
- **Dependency Injection**: Components receive dependencies, don't create them

## Classic Design Patterns Implementation

### 1. Strategy Pattern - Plant Care Algorithms
```python
class WateringStrategy(Protocol):
    def should_water(self, plant: Plant, sensor_data: SensorReading) -> bool: ...
    def calculate_amount(self, plant: Plant) -> WaterAmount: ...

class MoistureBasedStrategy:
    def should_water(self, plant: Plant, sensor_data: SensorReading) -> bool:
        return sensor_data.value < plant.optimal_moisture_min

class WeatherAwareStrategy:
    def should_water(self, plant: Plant, sensor_data: SensorReading) -> bool:
        # Consider weather forecast in watering decision
        weather = self.weather_service.get_forecast()
        return self.base_strategy.should_water(plant, sensor_data) and not weather.rain_expected

class PlantCareService:
    def __init__(self, strategy: WateringStrategy):
        self.strategy = strategy
```

### 2. Observer Pattern - Sensor Data Updates
```python
class SensorSubject:
    def __init__(self):
        self._observers: List[SensorObserver] = []
    
    def attach(self, observer: SensorObserver) -> None:
        self._observers.append(observer)
    
    def notify(self, reading: SensorReading) -> None:
        for observer in self._observers:
            await observer.update(reading)

class PlantHealthMonitor(SensorObserver):
    async def update(self, reading: SensorReading) -> None:
        if reading.value < CRITICAL_THRESHOLD:
            await self.alert_service.send_alert(f"Critical moisture: {reading.value}")

class DataLogger(SensorObserver):
    async def update(self, reading: SensorReading) -> None:
        await self.repository.save_reading(reading)
```

### 3. Command Pattern - Hardware Operations
```python
class Command(Protocol):
    async def execute(self) -> bool:
    async def undo(self) -> bool:

class WaterPlantCommand:
    def __init__(self, plant_id: PlantID, amount: WaterAmount, hardware: HardwareManager):
        self.plant_id = plant_id
        self.amount = amount
        self.hardware = hardware
        self.executed_at: Optional[datetime] = None
    
    async def execute(self) -> bool:
        success = await self.hardware.water(self.plant_id, self.amount)
        if success:
            self.executed_at = datetime.utcnow()
        return success
    
    async def undo(self) -> bool:
        # Log compensation action (can't un-water, but can adjust schedule)
        await self.hardware.adjust_next_watering(self.plant_id, delay_hours=24)
        return True

class CommandInvoker:
    def __init__(self):
        self.history: List[Command] = []
    
    async def execute_command(self, command: Command) -> bool:
        success = await command.execute()
        if success:
            self.history.append(command)
        return success
```

### 4. Factory Pattern - Hardware Creation
```python
class SensorFactory:
    _sensor_types = {
        SensorType.MOISTURE: MoistureSensor,
        SensorType.TEMPERATURE: TemperatureSensor,
        SensorType.HUMIDITY: HumiditySensor,
    }
    
    @classmethod
    def create_sensor(cls, sensor_type: SensorType, config: Dict[str, Any]) -> SensorInterface:
        if sensor_type not in cls._sensor_types:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        
        sensor_class = cls._sensor_types[sensor_type]
        return sensor_class(config)

# Environment-specific factories
class DevelopmentHardwareFactory(HardwareFactory):
    def create_sensor(self, sensor_type: SensorType, config: Dict) -> SensorInterface:
        return MockSensor(sensor_type, config)

class ProductionHardwareFactory(HardwareFactory):
    def create_sensor(self, sensor_type: SensorType, config: Dict) -> SensorInterface:
        return RustSensorBinding(sensor_type, config)
```

### 5. Adapter Pattern - Hardware Abstraction
```python
class RaspberryPiGPIOAdapter:
    """Adapts RPi.GPIO to our standard GPIO interface"""
    
    def __init__(self):
        import RPi.GPIO as GPIO
        self.gpio = GPIO
        self.gpio.setmode(GPIO.BCM)
    
    async def read_digital(self, pin: PinNumber) -> bool:
        return bool(self.gpio.input(pin))
    
    async def write_digital(self, pin: PinNumber, value: bool) -> bool:
        self.gpio.output(pin, value)
        return True

class MockGPIOAdapter:
    """Mock GPIO for development environment"""
    
    def __init__(self):
        self.pin_states: Dict[PinNumber, bool] = {}
    
    async def read_digital(self, pin: PinNumber) -> bool:
        return self.pin_states.get(pin, False)
    
    async def write_digital(self, pin: PinNumber, value: bool) -> bool:
        self.pin_states[pin] = value
        return True
```

### 6. Template Method Pattern - Sensor Reading Procedure
```python
class SensorReader(ABC):
    """Template for all sensor reading procedures"""
    
    async def read_sensor(self, sensor_id: SensorID) -> SensorReading:
        # Template method defines the algorithm
        await self.pre_read_setup(sensor_id)
        raw_value = await self.perform_read(sensor_id)
        calibrated_value = await self.calibrate_reading(raw_value, sensor_id)
        quality_score = await self.assess_quality(raw_value, calibrated_value)
        await self.post_read_cleanup(sensor_id)
        
        return SensorReading(
            sensor_id=sensor_id,
            value=calibrated_value,
            quality_score=quality_score,
            timestamp=datetime.utcnow()
        )
    
    @abstractmethod
    async def perform_read(self, sensor_id: SensorID) -> Decimal:
        """Subclasses implement specific reading logic"""
        pass
    
    async def pre_read_setup(self, sensor_id: SensorID) -> None:
        """Default implementation, can be overridden"""
        pass
    
    async def calibrate_reading(self, raw_value: Decimal, sensor_id: SensorID) -> Decimal:
        """Apply sensor-specific calibration"""
        calibration = await self.get_calibration_data(sensor_id)
        return raw_value * calibration.slope + calibration.offset
```

### 7. Chain of Responsibility - Error Handling
```python
class ErrorHandler(ABC):
    def __init__(self):
        self._next_handler: Optional[ErrorHandler] = None
    
    def set_next(self, handler: 'ErrorHandler') -> 'ErrorHandler':
        self._next_handler = handler
        return handler
    
    async def handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        if await self.can_handle(error, context):
            return await self.do_handle(error, context)
        elif self._next_handler:
            return await self._next_handler.handle(error, context)
        return False
    
    @abstractmethod
    async def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def do_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        pass

class SensorErrorHandler(ErrorHandler):
    async def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        return isinstance(error, SensorError)
    
    async def do_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        # Try to read from backup sensor
        backup_sensor_id = context.get('backup_sensor_id')
        if backup_sensor_id:
            reading = await self.sensor_service.read_sensor(backup_sensor_id)
            context['reading'] = reading
            return True
        return False

class NetworkErrorHandler(ErrorHandler):
    async def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        return isinstance(error, (ConnectionError, TimeoutError))
    
    async def do_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        # Retry with exponential backoff
        return await self.retry_with_backoff(context['operation'])
```

### 8. Builder Pattern - Complex Configuration
```python
class PlantConfigurationBuilder:
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'PlantConfigurationBuilder':
        self._config = PlantConfiguration()
        return self
    
    def with_species(self, species_id: SpeciesID) -> 'PlantConfigurationBuilder':
        self._config.species_id = species_id
        return self
    
    def with_sensors(self, *sensor_configs: SensorConfig) -> 'PlantConfigurationBuilder':
        self._config.sensors.extend(sensor_configs)
        return self
    
    def with_watering_strategy(self, strategy: WateringStrategy) -> 'PlantConfigurationBuilder':
        self._config.watering_strategy = strategy
        return self
    
    def with_care_schedule(self, schedule: CareSchedule) -> 'PlantConfigurationBuilder':
        self._config.care_schedule = schedule
        return self
    
    def build(self) -> PlantConfiguration:
        # Validate configuration before returning
        self._validate_configuration()
        result = self._config
        self.reset()  # Prepare for next build
        return result

# Usage
plant_config = (PlantConfigurationBuilder()
    .with_species(SpeciesID("spider-plant"))
    .with_sensors(
        SensorConfig(SensorType.MOISTURE, pin=21),
        SensorConfig(SensorType.TEMPERATURE, pin=22)
    )
    .with_watering_strategy(WeatherAwareStrategy())
    .with_care_schedule(CareSchedule(water_frequency_hours=48))
    .build())
```

## SOLID Principles in PlantOS

### 1. Single Responsibility Principle (SRP)

```python
# ❌ Bad: Class doing too many things
class PlantManager:
    def water_plant(self, plant_id): pass
    def read_sensors(self, plant_id): pass
    def send_notifications(self, message): pass
    def generate_reports(self, plant_id): pass
    def control_lights(self, plant_id): pass

# ✅ Good: Separated responsibilities
class PlantCareService:
    """Single responsibility: Plant care logic"""
    def determine_watering_needs(self, plant: Plant, sensor_data: SensorReading) -> WateringDecision: pass

class SensorService:
    """Single responsibility: Sensor data management"""
    def read_all_sensors(self, plant_id: PlantID) -> List[SensorReading]: pass

class NotificationService:
    """Single responsibility: User notifications"""
    def send_care_alert(self, plant_id: PlantID, alert_type: AlertType) -> bool: pass

class ReportingService:
    """Single responsibility: Data analysis and reporting"""
    def generate_plant_health_report(self, plant_id: PlantID) -> HealthReport: pass
```

### 2. Open/Closed Principle (OCP)

```python
# Extensible watering strategies without modifying existing code
class WateringStrategy(Protocol):
    def calculate_water_amount(self, plant: Plant, sensor_data: SensorReading) -> WaterAmount: pass

class BasicMoistureStrategy:
    def calculate_water_amount(self, plant: Plant, sensor_data: SensorReading) -> WaterAmount:
        if sensor_data.value < plant.optimal_moisture_min:
            return WaterAmount(100)  # Basic 100ml
        return WaterAmount(0)

class WeatherAwareStrategy:
    def __init__(self, weather_service: WeatherService):
        self.weather_service = weather_service
    
    def calculate_water_amount(self, plant: Plant, sensor_data: SensorReading) -> WaterAmount:
        base_amount = BasicMoistureStrategy().calculate_water_amount(plant, sensor_data)
        if base_amount > 0:
            weather = await self.weather_service.get_forecast()
            if weather.rain_probability > 0.7:
                return WaterAmount(int(base_amount * 0.3))  # Reduce watering
        return base_amount

class MLPredictiveStrategy:
    def __init__(self, ml_model: PlantCareModel):
        self.ml_model = ml_model
    
    def calculate_water_amount(self, plant: Plant, sensor_data: SensorReading) -> WaterAmount:
        features = self.extract_features(plant, sensor_data)
        prediction = self.ml_model.predict(features)
        return WaterAmount(int(prediction.recommended_water_ml))
```

### 3. Dependency Inversion Principle (DIP)

```python
# High-level modules depend on abstractions, not concretions
class PlantCareOrchestrator:
    def __init__(
        self,
        sensor_service: SensorService,
        watering_service: WateringService,
        notification_service: NotificationService,
        plant_repository: PlantRepository
    ):
        # Depend on abstractions (interfaces), not concrete implementations
        self.sensor_service = sensor_service
        self.watering_service = watering_service
        self.notification_service = notification_service
        self.plant_repository = plant_repository
    
    async def perform_care_cycle(self, plant_id: PlantID) -> None:
        plant = await self.plant_repository.get_by_id(plant_id)
        sensor_data = await self.sensor_service.read_latest(plant_id)
        
        if await self.needs_care(plant, sensor_data):
            success = await self.watering_service.water_plant(plant_id)
            if success:
                await self.notification_service.send_care_completion(plant_id)

# Dependency injection container
class DIContainer:
    def __init__(self, config: Config):
        # Concrete implementations chosen based on environment
        if config.environment == "development":
            self.sensor_service = MockSensorService()
            self.hardware_service = SimulatedHardwareService()
        else:
            self.sensor_service = RaspberryPiSensorService()
            self.hardware_service = RustHardwareService()
        
        self.plant_repository = SQLAlchemyPlantRepository(config.database_url)
        self.notification_service = EmailNotificationService(config.smtp_settings)
```

## Domain-Driven Design (DDD) Implementation

### 1. Bounded Contexts

```python
# Plant Care Context
class PlantCareContext:
    """Bounded context for plant care business logic"""
    
    class Plant(Entity):
        def __init__(self, plant_id: PlantID, species: PlantSpecies):
            self.id = plant_id
            self.species = species
            self.care_history: List[CareEvent] = []
            self.current_status = PlantStatus.HEALTHY
        
        def needs_watering(self, moisture_level: MoistureLevel) -> bool:
            return moisture_level < self.species.optimal_moisture_range.minimum
        
        def apply_care(self, care_event: CareEvent) -> None:
            self.care_history.append(care_event)
            self._update_status()

# Hardware Control Context  
class HardwareContext:
    """Bounded context for hardware operations"""
    
    class Device(Entity):
        def __init__(self, device_id: DeviceID, device_type: DeviceType):
            self.id = device_id
            self.type = device_type
            self.status = DeviceStatus.IDLE
            self.last_operation: Optional[datetime] = None
        
        def can_operate(self) -> bool:
            return self.status == DeviceStatus.IDLE

# Analytics Context
class AnalyticsContext:
    """Bounded context for data analysis and insights"""
    
    class PlantHealthAnalysis(ValueObject):
        def __init__(self, plant_id: PlantID, health_score: HealthScore, recommendations: List[str]):
            self.plant_id = plant_id
            self.health_score = health_score
            self.recommendations = recommendations
            self.analyzed_at = datetime.utcnow()
```

### 2. Domain Events

```python
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import List

class DomainEvent(ABC):
    """Base class for all domain events"""
    occurred_at: datetime
    event_id: UUID
    
    def __post_init__(self):
        if not hasattr(self, 'occurred_at'):
            self.occurred_at = datetime.utcnow()
        if not hasattr(self, 'event_id'):
            self.event_id = uuid4()

@dataclass
class PlantWateredEvent(DomainEvent):
    plant_id: PlantID
    water_amount: WaterAmount
    triggered_by: TriggerType
    moisture_before: MoistureLevel
    moisture_after: Optional[MoistureLevel] = None

@dataclass
class SensorReadingRecordedEvent(DomainEvent):
    sensor_id: SensorID
    plant_id: PlantID
    reading: SensorReading
    quality_assessment: QualityAssessment

@dataclass
class PlantHealthDeterioratedEvent(DomainEvent):
    plant_id: PlantID
    previous_health_score: HealthScore
    current_health_score: HealthScore
    deterioration_factors: List[str]

class DomainEventPublisher:
    def __init__(self):
        self._subscribers: Dict[type, List[Callable]] = {}
    
    def subscribe(self, event_type: type, handler: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    async def publish(self, event: DomainEvent):
        event_type = type(event)
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                await handler(event)

# Event handlers
class PlantCareEventHandlers:
    def __init__(self, notification_service: NotificationService, analytics_service: AnalyticsService):
        self.notification_service = notification_service
        self.analytics_service = analytics_service
    
    async def handle_plant_watered(self, event: PlantWateredEvent):
        # Update analytics
        await self.analytics_service.record_watering_event(event)
        
        # Send notification if manually triggered
        if event.triggered_by == TriggerType.MANUAL:
            await self.notification_service.send_watering_confirmation(event.plant_id)
    
    async def handle_health_deteriorated(self, event: PlantHealthDeterioratedEvent):
        # Send immediate alert
        await self.notification_service.send_health_alert(event.plant_id, event.current_health_score)
        
        # Trigger emergency care assessment
        await self.analytics_service.assess_emergency_care_needs(event.plant_id)
```

## Resilience Engineering Principles

### 1. Circuit Breaker Pattern - Prevent Cascading Failures

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        return (datetime.utcnow() - self.last_failure_time) >= timedelta(seconds=self.recovery_timeout)

# Usage in PlantOS
class ResilientSensorService:
    def __init__(self):
        self.sensor_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        self.backup_sensors: Dict[SensorID, SensorID] = {}
    
    async def read_sensor_with_fallback(self, sensor_id: SensorID) -> SensorReading:
        try:
            return await self.sensor_circuit.call(self.hardware.read_sensor, sensor_id)
        except CircuitBreakerOpenError:
            # Fallback to backup sensor or cached value
            backup_id = self.backup_sensors.get(sensor_id)
            if backup_id:
                logger.warning(f"Primary sensor {sensor_id} failed, using backup {backup_id}")
                return await self.hardware.read_sensor(backup_id)
            else:
                # Return last known good value with degraded quality
                return await self.get_cached_reading(sensor_id, quality_score=0.5)
```

### 2. Bulkhead Pattern - Resource Isolation

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

class BulkheadResourceManager:
    """Isolate different types of operations to prevent resource starvation"""
    
    def __init__(self):
        # Separate thread pools for different operation types
        self.sensor_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sensor")
        self.actuator_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="actuator")
        self.analytics_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="analytics")
        
        # Connection pools
        self.sensor_db_pool = create_pool(max_connections=10)
        self.analytics_db_pool = create_pool(max_connections=20)
        
        # Rate limiters
        self.sensor_rate_limiter = RateLimiter(requests_per_second=100)
        self.actuator_rate_limiter = RateLimiter(requests_per_second=10)  # Safety limit
    
    async def execute_sensor_operation(self, operation: Callable) -> Any:
        """Execute sensor operation in isolated resource pool"""
        async with self.sensor_rate_limiter:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.sensor_pool, operation)
    
    async def execute_actuator_operation(self, operation: Callable) -> Any:
        """Execute actuator operation with safety rate limiting"""
        async with self.actuator_rate_limiter:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.actuator_pool, operation)
```

### 3. Timeout & Retry Patterns

```python
import asyncio
import random
from functools import wraps

class RetryConfig:
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

def resilient_operation(timeout: float = 10.0, retry_config: RetryConfig = None):
    """Decorator for resilient operations with timeout and retry"""
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    # Apply timeout to operation
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                except asyncio.TimeoutError as e:
                    last_exception = e
                    logger.warning(f"Operation {func.__name__} timed out (attempt {attempt + 1})")
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Operation {func.__name__} failed (attempt {attempt + 1}): {e}")
                
                if attempt < retry_config.max_attempts - 1:
                    # Calculate exponential backoff with jitter
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                    await asyncio.sleep(delay + jitter)
            
            raise last_exception
        return wrapper
    return decorator

# Usage
class ResilientHardwareService:
    @resilient_operation(timeout=5.0, retry_config=RetryConfig(max_attempts=3))
    async def read_moisture_sensor(self, sensor_id: SensorID) -> SensorReading:
        return await self.hardware.read_sensor(sensor_id)
    
    @resilient_operation(timeout=30.0, retry_config=RetryConfig(max_attempts=2, base_delay=2.0))
    async def water_plant(self, plant_id: PlantID, amount: WaterAmount) -> bool:
        return await self.hardware.activate_pump(plant_id, amount)
```

## Advanced Microservices Patterns

### 1. API Gateway Pattern

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time

class APIGateway:
    def __init__(self):
        self.app = FastAPI(title="PlantOS API Gateway", version="1.0.0")
        self.setup_middleware()
        self.setup_routes()
        self.rate_limiter = RateLimiter()
        self.auth_service = AuthenticationService()
    
    def setup_middleware(self):
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Rate limiting middleware
        self.app.add_middleware(RateLimitingMiddleware, rate_limiter=self.rate_limiter)
        
        # Authentication middleware
        self.app.add_middleware(AuthenticationMiddleware, auth_service=self.auth_service)
        
        # Request/Response logging
        self.app.add_middleware(LoggingMiddleware)
    
    def setup_routes(self):
        # Route to Plant Care Service
        @self.app.api_route("/api/v1/plants/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def proxy_plant_service(request: Request, path: str):
            return await self.proxy_request(request, "plant-care-service", f"/plants/{path}")
        
        # Route to Hardware Service
        @self.app.api_route("/api/v1/hardware/{path:path}", methods=["GET", "POST"])
        async def proxy_hardware_service(request: Request, path: str):
            return await self.proxy_request(request, "hardware-service", f"/hardware/{path}")
        
        # Route to Analytics Service
        @self.app.api_route("/api/v1/analytics/{path:path}", methods=["GET"])
        async def proxy_analytics_service(request: Request, path: str):
            return await self.proxy_request(request, "analytics-service", f"/analytics/{path}")

class RateLimitingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        
        if not await self.rate_limiter.allow_request(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        response = await call_next(request)
        return response
```

### 2. Saga Pattern for Distributed Transactions

```python
from enum import Enum
from typing import Dict, List, Callable
import asyncio

class SagaStep:
    def __init__(self, action: Callable, compensation: Callable, description: str):
        self.action = action
        self.compensation = compensation
        self.description = description

class SagaStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"

class Saga:
    def __init__(self, saga_id: str, steps: List[SagaStep]):
        self.saga_id = saga_id
        self.steps = steps
        self.completed_steps: List[int] = []
        self.status = SagaStatus.PENDING
        self.failure_reason: Optional[str] = None
    
    async def execute(self) -> bool:
        """Execute saga steps in order"""
        try:
            for i, step in enumerate(self.steps):
                logger.info(f"Executing saga step {i+1}/{len(self.steps)}: {step.description}")
                await step.action()
                self.completed_steps.append(i)
            
            self.status = SagaStatus.COMPLETED
            return True
            
        except Exception as e:
            self.status = SagaStatus.FAILED
            self.failure_reason = str(e)
            logger.error(f"Saga {self.saga_id} failed at step {len(self.completed_steps)+1}: {e}")
            
            # Execute compensation actions in reverse order
            await self.compensate()
            return False
    
    async def compensate(self):
        """Execute compensation actions for completed steps"""
        self.status = SagaStatus.COMPENSATING
        
        for step_index in reversed(self.completed_steps):
            try:
                step = self.steps[step_index]
                logger.info(f"Compensating step {step_index+1}: {step.description}")
                await step.compensation()
            except Exception as e:
                logger.error(f"Compensation failed for step {step_index+1}: {e}")
        
        self.status = SagaStatus.COMPENSATED

# Example: Complex plant care workflow
class PlantCareWorkflowSaga:
    def __init__(self, plant_id: PlantID, care_plan: CarePlan):
        self.plant_id = plant_id
        self.care_plan = care_plan
    
    async def create_comprehensive_care_saga(self) -> Saga:
        steps = [
            SagaStep(
                action=lambda: self.reserve_resources(),
                compensation=lambda: self.release_resources(),
                description="Reserve hardware resources"
            ),
            SagaStep(
                action=lambda: self.read_all_sensors(),
                compensation=lambda: self.clear_sensor_cache(),
                description="Read current sensor values"
            ),
            SagaStep(
                action=lambda: self.analyze_plant_needs(),
                compensation=lambda: self.clear_analysis_cache(),
                description="Analyze plant care needs"
            ),
            SagaStep(
                action=lambda: self.execute_watering(),
                compensation=lambda: self.log_watering_reversal(),
                description="Execute watering if needed"
            ),
            SagaStep(
                action=lambda: self.update_care_history(),
                compensation=lambda: self.remove_care_record(),
                description="Update plant care history"
            ),
            SagaStep(
                action=lambda: self.send_notifications(),
                compensation=lambda: self.send_failure_notification(),
                description="Send completion notifications"
            )
        ]
        
        return Saga(f"care-workflow-{self.plant_id}-{uuid4()}", steps)
```

## Security-First Principles

### 1. Zero Trust Architecture

```python
class ZeroTrustSecurityManager:
    """Implement zero trust - verify every request"""
    
    def __init__(self, auth_service: AuthService, audit_service: AuditService):
        self.auth_service = auth_service
        self.audit_service = audit_service
        self.threat_detector = ThreatDetectionService()
    
    async def verify_request(self, request: Request, required_permissions: List[str]) -> bool:
        # 1. Authenticate the request
        token = extract_auth_token(request)
        if not token:
            await self.audit_service.log_security_event("UNAUTHENTICATED_REQUEST", request)
            return False
        
        user_identity = await self.auth_service.validate_token(token)
        if not user_identity:
            await self.audit_service.log_security_event("INVALID_TOKEN", request)
            return False
        
        # 2. Authorize the specific action
        if not await self.auth_service.has_permissions(user_identity, required_permissions):
            await self.audit_service.log_security_event("INSUFFICIENT_PERMISSIONS", request)
            return False
        
        # 3. Check for suspicious patterns
        if await self.threat_detector.is_suspicious(request, user_identity):
            await self.audit_service.log_security_event("SUSPICIOUS_ACTIVITY", request)
            return False
        
        # 4. Validate request integrity
        if not await self.validate_request_integrity(request):
            await self.audit_service.log_security_event("REQUEST_INTEGRITY_VIOLATION", request)
            return False
        
        # Log successful authorization
        await self.audit_service.log_access_granted(user_identity, request)
        return True

class SecureHardwareProxy:
    """Security layer for hardware operations"""
    
    def __init__(self, hardware_service: HardwareService, security_manager: ZeroTrustSecurityManager):
        self.hardware_service = hardware_service
        self.security_manager = security_manager
        self.safety_limits = SafetyLimits()
    
    async def water_plant(self, request: Request, plant_id: PlantID, amount: WaterAmount) -> bool:
        # Verify permissions
        if not await self.security_manager.verify_request(request, ["hardware.water"]):
            raise SecurityError("Unauthorized hardware access")
        
        # Apply safety limits
        if amount > self.safety_limits.max_water_amount:
            raise SafetyError(f"Water amount {amount} exceeds safety limit")
        
        # Check rate limiting
        if not await self.check_watering_rate_limit(plant_id):
            raise SafetyError("Watering rate limit exceeded")
        
        return await self.hardware_service.water_plant(plant_id, amount)
```

### 2. Defense in Depth

```python
class LayeredSecurityValidator:
    """Multiple security validation layers"""
    
    def __init__(self):
        self.validators = [
            InputSanitizationValidator(),
            SQLInjectionValidator(),
            XSSValidator(),
            BusinessLogicValidator(),
            RateLimitValidator(),
            ThreatSignatureValidator()
        ]
    
    async def validate_request(self, request: Request, context: Dict[str, Any]) -> ValidationResult:
        results = []
        
        for validator in self.validators:
            try:
                result = await validator.validate(request, context)
                results.append(result)
                
                if result.severity == SecuritySeverity.CRITICAL:
                    # Immediate rejection for critical threats
                    return ValidationResult(
                        valid=False,
                        severity=SecuritySeverity.CRITICAL,
                        reason=f"Critical security violation: {result.reason}"
                    )
                    
            except Exception as e:
                logger.error(f"Security validator {validator.__class__.__name__} failed: {e}")
                # Fail secure - reject request if validator fails
                return ValidationResult(
                    valid=False,
                    severity=SecuritySeverity.HIGH,
                    reason="Security validation system failure"
                )
        
        # Aggregate results
        highest_severity = max(results, key=lambda r: r.severity.value).severity
        
        return ValidationResult(
            valid=highest_severity <= SecuritySeverity.MEDIUM,
            severity=highest_severity,
            reason="Aggregated security validation"
        )
```

## Performance Engineering

### 1. Performance Budgets & Monitoring

```python
class PerformanceBudget:
    """Define and enforce performance budgets"""
    
    def __init__(self):
        self.budgets = {
            "api_response_time_p95": 100,  # milliseconds
            "sensor_read_latency_p99": 10,  # milliseconds
            "database_query_time_p95": 10,  # milliseconds
            "memory_usage_max": 512,  # MB
            "cpu_usage_avg": 70,  # percentage
        }
        self.alerts = PerformanceAlertManager()
    
    async def check_budget_compliance(self, metrics: Dict[str, float]) -> PerformanceBudgetResult:
        violations = []
        
        for metric_name, budget_value in self.budgets.items():
            actual_value = metrics.get(metric_name)
            if actual_value and actual_value > budget_value:
                violation = PerformanceViolation(
                    metric=metric_name,
                    budget=budget_value,
                    actual=actual_value,
                    severity=self._calculate_severity(budget_value, actual_value)
                )
                violations.append(violation)
                
                # Send alert for significant violations
                if violation.severity >= ViolationSeverity.HIGH:
                    await self.alerts.send_performance_alert(violation)
        
        return PerformanceBudgetResult(
            compliant=len(violations) == 0,
            violations=violations,
            overall_score=self._calculate_performance_score(violations)
        )

class AsyncConnectionPool:
    """High-performance async connection pooling"""
    
    def __init__(self, max_connections: int = 20, min_connections: int = 5):
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self.active_connections = 0
        self.total_created = 0
        self.pool_stats = PoolStatistics()
    
    async def get_connection(self, timeout: float = 5.0) -> Connection:
        start_time = time.time()
        
        try:
            # Try to get existing connection
            connection = await asyncio.wait_for(self.pool.get(), timeout=1.0)
            self.pool_stats.record_connection_acquired(time.time() - start_time)
            return connection
            
        except asyncio.TimeoutError:
            # Create new connection if under limit
            if self.active_connections < self.max_connections:
                connection = await self._create_connection()
                self.active_connections += 1
                self.total_created += 1
                self.pool_stats.record_connection_created(time.time() - start_time)
                return connection
            else:
                # Wait for available connection
                connection = await asyncio.wait_for(self.pool.get(), timeout=timeout - 1.0)
                self.pool_stats.record_connection_acquired(time.time() - start_time)
                return connection
    
    async def return_connection(self, connection: Connection):
        if connection.is_healthy():
            await self.pool.put(connection)
        else:
            # Replace unhealthy connection
            self.active_connections -= 1
            new_connection = await self._create_connection()
            await self.pool.put(new_connection)
```

## Polyglot Architecture Patterns

### 1. Anti-Corruption Layer - Language Boundaries
```python
# Python service calling Go service
class GoStreamingServiceAdapter:
    """Anti-corruption layer for Go streaming service"""
    
    async def start_sensor_stream(self, plant_id: PlantID) -> bool:
        # Convert Python types to gRPC types
        request = streaming_pb2.StartStreamRequest(
            plant_id=str(plant_id),
            sensors=[sensor.to_proto() for sensor in self.sensors]
        )
        
        try:
            response = await self.grpc_client.StartStream(request)
            return response.success
        except grpc.RpcError as e:
            # Convert gRPC errors to Python domain errors
            raise StreamingServiceError(f"Failed to start stream: {e.details()}")
```

### 2. Shared Kernel - Common Types Across Languages
```
# Protocol Buffers for cross-language communication
syntax = "proto3";

message SensorReading {
  string sensor_id = 1;
  string plant_id = 2;
  string sensor_type = 3;
  double value = 4;
  string unit = 5;
  double quality_score = 6;
  int64 timestamp = 7;
}

message PlantStatus {
  string plant_id = 1;
  double health_score = 2;
  double moisture_level = 3;
  bool needs_water = 4;
  int64 last_watered = 5;
}
```

### 3. Event Sourcing - Cross-Service Communication
```python
@dataclass
class PlantEvent:
    event_id: UUID
    plant_id: PlantID
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    
class PlantEventStore:
    async def append_event(self, event: PlantEvent) -> None:
        # Store in both database and Kafka
        await self.db_repository.save_event(event)
        await self.kafka_producer.send('plant-events', event.to_dict())
    
    async def get_plant_history(self, plant_id: PlantID) -> List[PlantEvent]:
        return await self.db_repository.get_events_for_plant(plant_id)
```

## Testing Patterns

### 1. Test Doubles Hierarchy
```python
# Fake - Working implementation for testing
class FakeHardwareManager(HardwareManager):
    def __init__(self):
        super().__init__({})
        self.sensors_data: Dict[SensorID, Decimal] = {}
        self.pump_operations: List[Tuple[PlantID, WaterAmount]] = []
    
    async def read_sensor(self, sensor_id: SensorID) -> SensorReading:
        value = self.sensors_data.get(sensor_id, Decimal('50.0'))
        return SensorReading(sensor_id, value, datetime.utcnow(), 1.0)

# Mock - Behavior verification
class MockHardwareManager:
    def __init__(self):
        self.read_sensor = AsyncMock()
        self.water_plant = AsyncMock()
    
# Stub - Predetermined responses
class StubWeatherService:
    def __init__(self, weather_data: WeatherData):
        self.weather_data = weather_data
    
    async def get_forecast(self) -> WeatherData:
        return self.weather_data
```

### 2. Object Mother Pattern - Test Data Creation
```python
class PlantMother:
    @staticmethod
    def healthy_spider_plant() -> Plant:
        return Plant(
            id=PlantID(UUID('12345678-1234-5678-9012-123456789012')),
            name="Test Spider Plant",
            species_id=SpeciesID(UUID('87654321-4321-8765-2109-876543210987')),
            optimal_moisture_min=Decimal('30.0'),
            optimal_moisture_max=Decimal('60.0'),
            health_score=Decimal('0.95'),
            status="active",
            created_at=datetime.utcnow()
        )
    
    @staticmethod
    def thirsty_plant() -> Plant:
        plant = PlantMother.healthy_spider_plant()
        plant.health_score = Decimal('0.3')
        plant.last_watered_at = datetime.utcnow() - timedelta(days=7)
        return plant

class SensorReadingMother:
    @staticmethod
    def critical_moisture_reading(plant_id: PlantID) -> SensorReading:
        return SensorReading(
            sensor_id=SensorID(UUID('11111111-1111-1111-1111-111111111111')),
            plant_id=plant_id,
            sensor_type=SensorType.MOISTURE,
            value=Decimal('15.0'),  # Critical level
            unit="percentage",
            quality_score=1.0,
            timestamp=datetime.utcnow()
        )
```

## Configuration Management Excellence

### 1. Environment Parity & Configuration as Code

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
from pathlib import Path

@dataclass
class DatabaseConfig:
    url: str
    pool_size: int = 10
    pool_timeout: int = 30
    echo_queries: bool = False
    
    @classmethod
    def from_env(cls, env_prefix: str = "DB_") -> 'DatabaseConfig':
        return cls(
            url=os.getenv(f"{env_prefix}URL", "postgresql://localhost/plantos"),
            pool_size=int(os.getenv(f"{env_prefix}POOL_SIZE", "10")),
            pool_timeout=int(os.getenv(f"{env_prefix}POOL_TIMEOUT", "30")),
            echo_queries=os.getenv(f"{env_prefix}ECHO_QUERIES", "false").lower() == "true"
        )

@dataclass
class HardwareConfig:
    simulation_mode: bool = True
    gpio_pins: Dict[str, int] = field(default_factory=dict)
    i2c_addresses: Dict[str, int] = field(default_factory=dict)
    safety_timeouts: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> 'HardwareConfig':
        return cls(
            simulation_mode=os.getenv("HARDWARE_SIMULATION", "true").lower() == "true",
            gpio_pins=cls._parse_pin_config(os.getenv("GPIO_PINS", "")),
            i2c_addresses=cls._parse_i2c_config(os.getenv("I2C_ADDRESSES", "")),
            safety_timeouts=cls._parse_timeout_config(os.getenv("SAFETY_TIMEOUTS", ""))
        )

class ConfigurationManager:
    """Centralized configuration management with validation"""
    
    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.config = self._load_configuration()
        self._validate_configuration()
    
    def _load_configuration(self) -> PlantOSConfig:
        # Load base configuration
        config_path = Path(f"config/{self.environment}.yaml")
        base_config = self._load_yaml_config(config_path)
        
        # Override with environment variables
        env_overrides = self._extract_env_overrides()
        
        # Merge configurations
        merged_config = self._merge_configs(base_config, env_overrides)
        
        return PlantOSConfig.from_dict(merged_config)
    
    def _validate_configuration(self):
        """Validate configuration integrity and security"""
        validators = [
            SecurityConfigValidator(),
            PerformanceConfigValidator(),
            HardwareConfigValidator(),
            NetworkConfigValidator()
        ]
        
        for validator in validators:
            validator.validate(self.config)

class FeatureFlags:
    """Feature toggle management for safe deployments"""
    
    def __init__(self, config_source: str = "database"):
        self.flags: Dict[str, FeatureFlag] = {}
        self.config_source = config_source
        self.refresh_interval = 60  # seconds
        self._start_refresh_task()
    
    async def is_enabled(self, flag_name: str, context: Dict[str, Any] = None) -> bool:
        flag = self.flags.get(flag_name)
        if not flag:
            return False
        
        # Check global toggle
        if not flag.enabled:
            return False
        
        # Check user/group targeting
        if context and flag.targeting_rules:
            return await self._evaluate_targeting_rules(flag.targeting_rules, context)
        
        # Check percentage rollout
        if flag.percentage < 100:
            user_hash = self._hash_user_id(context.get("user_id", "anonymous"))
            return (user_hash % 100) < flag.percentage
        
        return True
    
    async def get_flag_value(self, flag_name: str, default_value: Any = None, context: Dict[str, Any] = None) -> Any:
        if await self.is_enabled(flag_name, context):
            flag = self.flags.get(flag_name)
            return flag.value if flag else default_value
        return default_value

# Usage in PlantOS
class PlantCareService:
    def __init__(self, feature_flags: FeatureFlags):
        self.feature_flags = feature_flags
    
    async def calculate_watering_schedule(self, plant_id: PlantID) -> WateringSchedule:
        # Check if ML-based scheduling is enabled
        if await self.feature_flags.is_enabled("ml_watering_scheduler", {"plant_id": str(plant_id)}):
            return await self.ml_scheduler.calculate_schedule(plant_id)
        else:
            return await self.basic_scheduler.calculate_schedule(plant_id)
```

## Recommended Implementation Order

1. **Phase 1-2**: Implement core patterns (Strategy, Observer, Factory)
2. **Phase 3-4**: Add hardware patterns (Adapter, Template Method)
3. **Phase 5-6**: Implement data patterns (Repository, Event Sourcing)
4. **Phase 7-8**: Add processing patterns (Chain of Responsibility, Command)
5. **Phase 9-10**: Implement integration patterns (Anti-Corruption Layer)
6. **Phase 11-12**: Add advanced patterns (Builder, complex workflows)
7. **Phase 13-14**: Implement resilience patterns (Circuit Breaker, Bulkhead)
8. **Phase 15-16**: Add security patterns (Zero Trust, Defense in Depth)

These comprehensive patterns make PlantOS enterprise-grade with maximum resilience, security, and operational excellence while maintaining the flexibility needed for an IoT system operating across different environments.