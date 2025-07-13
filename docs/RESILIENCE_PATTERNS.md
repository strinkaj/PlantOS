# PlantOS - Resilience & Advanced Engineering Patterns

This document extends the architectural patterns to include advanced software engineering principles for maximum resilience, reusability, and operational excellence.

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

These advanced patterns provide PlantOS with enterprise-grade resilience, security, and operational capabilities while maintaining the flexibility needed for an IoT system operating across different environments.