# PlantOS - Architecture Patterns & Best Practices

This document outlines the architectural patterns and principles from The Pragmatic Programmer and classic design patterns that PlantOS implements for optimal reusability, maintainability, and testability.

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

## Design Patterns Implementation

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

## Recommended Implementation Order

1. **Phase 1-2**: Implement core patterns (Strategy, Observer, Factory)
2. **Phase 3-4**: Add hardware patterns (Adapter, Template Method)
3. **Phase 5-6**: Implement data patterns (Repository, Event Sourcing)
4. **Phase 7-8**: Add processing patterns (Chain of Responsibility, Command)
5. **Phase 9-10**: Implement integration patterns (Anti-Corruption Layer)
6. **Phase 11-12**: Add advanced patterns (Builder, complex workflows)

These patterns will make PlantOS more maintainable, testable, and adaptable to changing requirements while following proven software engineering principles.