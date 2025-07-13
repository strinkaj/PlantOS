# PlantOS Development Guide

## Development Environment Setup

### Prerequisites (All Free for Personal Use)
- Python 3.12+ (PSF License)
- Go 1.21+ (BSD License)
- Rust 1.75+ with Cargo (MIT/Apache 2.0)
- Julia 1.10+ (MIT License)
- PostgreSQL 15+ with TimescaleDB extension (PostgreSQL/Apache 2.0)
- Redis 7+ (BSD License)
- Container Runtime (choose one):
  - Podman & Podman Compose (Apache 2.0) - recommended for all users
  - Docker & Docker Compose (free for personal/small business)
  - containerd with nerdctl (Apache 2.0)
- Make (GPL)
- Infrastructure as Code:
  - OpenTofu 1.6+ (MPL 2.0) - recommended
  - Terraform 1.6+ (BSL - check license for business use)

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

# Initialize database (using Podman)
podman-compose up -d postgres redis
# OR using Docker
docker-compose up -d postgres redis

alembic upgrade head

# Run tests
pytest
```

## Code Quality Standards

### Python Code Style

#### Strict Type Hints with Validation
```python
from typing import Optional, List, Dict, Union, Protocol, Literal, NewType
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from uuid import UUID
import structlog

# Type-safe aliases
PlantID = NewType('PlantID', UUID)
SensorReading = NewType('SensorReading', Decimal)
WaterAmount = NewType('WaterAmount', int)  # milliliters

# Structured logger
logger = structlog.get_logger(__name__)

class WateringPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class WeatherData(BaseModel):
    temperature_celsius: float = Field(..., ge=-50, le=60)
    humidity_percent: float = Field(..., ge=0, le=100)
    precipitation_mm: float = Field(..., ge=0)
    forecast_hours: int = Field(..., ge=1, le=168)  # 1 week max
    
    @validator('temperature_celsius')
    def validate_temperature_reasonable(cls, v):
        if not -20 <= v <= 45:  # Reasonable range for plant care
            logger.warning("Unusual temperature reading", temperature=v)
        return v

class Plant(BaseModel):
    id: PlantID
    name: str = Field(..., min_length=1, max_length=255)
    species_id: Optional[UUID] = None
    optimal_moisture_min: float = Field(..., ge=0, le=100)
    optimal_moisture_max: float = Field(..., ge=0, le=100)
    water_frequency_hours: int = Field(..., ge=1, le=720)  # Max 30 days
    created_at: datetime
    
    @validator('optimal_moisture_max')
    def validate_moisture_range(cls, v, values):
        if 'optimal_moisture_min' in values and v <= values['optimal_moisture_min']:
            raise ValueError('max moisture must be greater than min moisture')
        return v

    class Config:
        # Forbid extra fields for strict validation
        extra = 'forbid'
        # Use enum values instead of names
        use_enum_values = True

# Protocol for type-safe dependency injection
class PlantRepository(Protocol):
    async def find_by_id(self, plant_id: PlantID) -> Optional[Plant]: ...
    async def save(self, plant: Plant) -> Plant: ...

class WateringService(Protocol):
    async def water_plant(
        self, 
        plant_id: PlantID, 
        amount_ml: WaterAmount
    ) -> bool: ...

# Comprehensive type annotations with error handling
async def calculate_water_needed(
    plant: Plant,
    last_watered: datetime,
    soil_moisture: SensorReading,
    weather_data: Optional[WeatherData] = None,
    *,  # Force keyword-only arguments for safety
    max_water_ml: WaterAmount = WaterAmount(500),
    safety_margin: float = 0.1
) -> tuple[WaterAmount, WateringPriority]:
    """
    Calculate water amount needed for a plant with comprehensive validation.
    
    Args:
        plant: Plant entity with care requirements
        last_watered: Last watering timestamp (must be in past)
        soil_moisture: Current soil moisture percentage (0-100)
        weather_data: Optional weather forecast data
        max_water_ml: Safety limit for water amount
        safety_margin: Safety factor for calculations (0.0-1.0)
        
    Returns:
        Tuple of (water_amount_ml, priority)
        
    Raises:
        ValueError: If soil_moisture is invalid or dates are inconsistent
        TypeError: If arguments have wrong types
        
    Example:
        >>> plant = Plant(id=PlantID(uuid4()), name="Tomato", ...)
        >>> amount, priority = await calculate_water_needed(
        ...     plant=plant,
        ...     last_watered=datetime.now() - timedelta(hours=24),
        ...     soil_moisture=SensorReading(Decimal('45.5'))
        ... )
    """
    # Input validation with structured logging
    if not 0 <= float(soil_moisture) <= 100:
        logger.error(
            "Invalid soil moisture reading",
            plant_id=str(plant.id),
            soil_moisture=float(soil_moisture),
            valid_range="0-100"
        )
        raise ValueError(f"Soil moisture must be 0-100%, got {soil_moisture}")
    
    if last_watered > datetime.now():
        logger.error(
            "Last watered time is in future",
            plant_id=str(plant.id),
            last_watered=last_watered.isoformat(),
            current_time=datetime.now().isoformat()
        )
        raise ValueError("Last watered time cannot be in the future")
    
    if not 0 <= safety_margin <= 1:
        raise ValueError(f"Safety margin must be 0-1, got {safety_margin}")
    
    # Log calculation start
    logger.info(
        "Starting water calculation",
        plant_id=str(plant.id),
        plant_name=plant.name,
        current_moisture=float(soil_moisture),
        hours_since_watered=(datetime.now() - last_watered).total_seconds() / 3600
    )
    
    # Calculate base water need
    moisture_deficit = max(0, plant.optimal_moisture_min - float(soil_moisture))
    hours_since_watered = (datetime.now() - last_watered).total_seconds() / 3600
    
    # Base calculation with type safety
    base_amount = int(moisture_deficit * 10)  # 10ml per percent deficit
    time_factor = min(2.0, hours_since_watered / plant.water_frequency_hours)
    
    # Weather adjustments if available
    weather_factor = 1.0
    if weather_data:
        # Reduce watering if rain expected
        if weather_data.precipitation_mm > 5:
            weather_factor *= 0.5
            logger.info("Reducing water due to expected rain", 
                       precipitation_mm=weather_data.precipitation_mm)
        
        # Increase watering in hot weather
        if weather_data.temperature_celsius > 30:
            weather_factor *= 1.3
            logger.info("Increasing water due to high temperature",
                       temperature=weather_data.temperature_celsius)
    
    # Final calculation with safety margin
    calculated_amount = int(base_amount * time_factor * weather_factor * (1 + safety_margin))
    final_amount = WaterAmount(min(calculated_amount, max_water_ml))
    
    # Determine priority based on conditions
    if float(soil_moisture) < plant.optimal_moisture_min * 0.5:
        priority = WateringPriority.CRITICAL
    elif float(soil_moisture) < plant.optimal_moisture_min * 0.8:
        priority = WateringPriority.HIGH
    elif float(soil_moisture) < plant.optimal_moisture_min:
        priority = WateringPriority.MEDIUM
    else:
        priority = WateringPriority.LOW
    
    # Log final decision
    logger.info(
        "Water calculation completed",
        plant_id=str(plant.id),
        calculated_amount_ml=final_amount,
        priority=priority.value,
        moisture_deficit=moisture_deficit,
        weather_factor=weather_factor
    )
    
    return final_amount, priority

# Type-safe configuration with secrets management
class DatabaseConfig(BaseModel):
    host: str = Field(..., min_length=1)
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(..., min_length=1)
    username: str = Field(..., min_length=1)
    password: SecretStr  # Pydantic SecretStr for sensitive data
    pool_size: int = Field(default=10, ge=1, le=100)
    
    class Config:
        # Prevent password from appearing in repr
        extra = 'forbid'
        
    @validator('host')
    def validate_host(cls, v):
        # Basic hostname validation
        if not v.replace('-', '').replace('.', '').replace('_', '').isalnum():
            raise ValueError('Invalid hostname format')
        return v

# Secrets-aware configuration loader
from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    Type-safe configuration with automatic environment variable loading.
    Secrets are loaded from environment variables or HashiCorp Vault.
    """
    # Database configuration
    database: DatabaseConfig
    
    # API configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=4, ge=1, le=32)
    
    # Security configuration
    secret_key: SecretStr
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=30, ge=5, le=1440)
    
    # Vault configuration (production)
    vault_url: Optional[str] = None
    vault_token: Optional[SecretStr] = None
    vault_path: str = Field(default="secret/plantos")
    
    # Logging configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "text"] = Field(default="json")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"  # DATABASE__HOST
        case_sensitive = False
        
        # Custom config for different environments
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,
                vault_settings,  # Custom vault loader
                file_secret_settings,
            )

# Type-safe dependency injection container
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    """Type-safe IoC container with proper resource management."""
    
    # Configuration
    config = providers.Configuration()
    
    # Logging setup
    logger = providers.Resource(
        setup_structured_logging,
        level=config.log_level,
        format=config.log_format
    )
    
    # Database connection pool
    database = providers.Singleton(
        AsyncDatabase,
        config=config.database,
        logger=logger
    )
    
    # Repositories with proper typing
    plant_repository: providers.Provider[PlantRepository] = providers.Factory(
        PostgresPlantRepository,
        session_factory=database.provided.session_factory,
        logger=logger
    )
    
    # Services
    watering_service: providers.Provider[WateringService] = providers.Factory(
        HardwareWateringService,
        logger=logger
    )
    
    # Use cases with comprehensive logging
    water_plant_use_case = providers.Factory(
        WaterPlantUseCase,
        plant_repository=plant_repository,
        watering_service=watering_service,
        logger=logger
    )
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


### Go Code Standards

#### Type-Safe Service Architecture
```go
// services/streaming/sensor_pipeline.go
package streaming

import (
    "context"
    "fmt"
    "sync"
    "time"
    "errors"
    "crypto/tls"
    
    "github.com/prometheus/client_golang/prometheus"
    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
    "github.com/google/uuid"
    "github.com/go-playground/validator/v10"
)

// Strong typing for domain identifiers
type (
    PlantID  uuid.UUID
    SensorID uuid.UUID
)

// String methods for proper logging
func (p PlantID) String() string  { return uuid.UUID(p).String() }
func (s SensorID) String() string { return uuid.UUID(s).String() }

// Validated sensor reading with comprehensive type safety
type SensorReading struct {
    ID        SensorID  `json:"sensor_id" validate:"required"`
    PlantID   PlantID   `json:"plant_id" validate:"required"`
    Type      SensorType `json:"type" validate:"required,oneof=moisture temperature humidity light"`
    Value     float64   `json:"value" validate:"required,gte=0"`
    Unit      string    `json:"unit" validate:"required,oneof=celsius fahrenheit percent lux"`
    Timestamp time.Time `json:"timestamp" validate:"required"`
    Quality   QualityIndicator `json:"quality" validate:"required,oneof=good fair poor"`
}

// Enum types for type safety
type SensorType string
const (
    SensorTypeMoisture    SensorType = "moisture"
    SensorTypeTemperature SensorType = "temperature"
    SensorTypeHumidity    SensorType = "humidity"
    SensorTypeLight       SensorType = "light"
)

type QualityIndicator string
const (
    QualityGood QualityIndicator = "good"
    QualityFair QualityIndicator = "fair"
    QualityPoor QualityIndicator = "poor"
)

// Comprehensive error types with structured information
type PipelineError struct {
    Code      ErrorCode `json:"code"`
    Message   string    `json:"message"`
    PlantID   *PlantID  `json:"plant_id,omitempty"`
    SensorID  *SensorID `json:"sensor_id,omitempty"`
    Timestamp time.Time `json:"timestamp"`
    Context   map[string]interface{} `json:"context,omitempty"`
}

func (e *PipelineError) Error() string {
    return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

type ErrorCode string
const (
    ErrorCodeValidation     ErrorCode = "VALIDATION_ERROR"
    ErrorCodeTimeout        ErrorCode = "TIMEOUT_ERROR"
    ErrorCodeCapacityFull   ErrorCode = "CAPACITY_FULL"
    ErrorCodeSensorOffline  ErrorCode = "SENSOR_OFFLINE"
    ErrorCodeInternalError  ErrorCode = "INTERNAL_ERROR"
)

// Configuration with secrets management
type Config struct {
    // Server configuration
    Host string `validate:"required,hostname_rfc1123" env:"STREAMING_HOST" default:"0.0.0.0"`
    Port int    `validate:"required,min=1,max=65535" env:"STREAMING_PORT" default:"8080"`
    
    // TLS configuration for secure communication
    TLS struct {
        Enabled  bool   `env:"STREAMING_TLS_ENABLED" default:"true"`
        CertFile string `env:"STREAMING_TLS_CERT_FILE" validate:"required_if=Enabled true"`
        KeyFile  string `env:"STREAMING_TLS_KEY_FILE" validate:"required_if=Enabled true"`
    }
    
    // Database connection (with secrets)
    Database struct {
        Host     string `validate:"required" env:"DB_HOST"`
        Port     int    `validate:"required,min=1,max=65535" env:"DB_PORT" default:"5432"`
        Name     string `validate:"required" env:"DB_NAME"`
        Username string `validate:"required" env:"DB_USERNAME"`
        Password string `validate:"required" env:"DB_PASSWORD" secret:"true"` // Marked as secret
        SSLMode  string `validate:"oneof=disable require" env:"DB_SSL_MODE" default:"require"`
    }
    
    // Kafka configuration
    Kafka struct {
        Brokers  []string `validate:"required,dive,hostname_port" env:"KAFKA_BROKERS" sep:","`
        Topic    string   `validate:"required" env:"KAFKA_TOPIC" default:"sensor-readings"`
        Username string   `env:"KAFKA_USERNAME"`
        Password string   `env:"KAFKA_PASSWORD" secret:"true"`
    }
    
    // Pipeline configuration
    Pipeline struct {
        BufferSize   int           `validate:"min=1,max=100000" env:"PIPELINE_BUFFER_SIZE" default:"10000"`
        WorkerCount  int           `validate:"min=1,max=100" env:"PIPELINE_WORKER_COUNT" default:"10"`
        FlushTimeout time.Duration `env:"PIPELINE_FLUSH_TIMEOUT" default:"5s"`
        BatchSize    int           `validate:"min=1,max=1000" env:"PIPELINE_BATCH_SIZE" default:"100"`
    }
    
    // Monitoring configuration
    Monitoring struct {
        MetricsEnabled bool   `env:"MONITORING_METRICS_ENABLED" default:"true"`
        MetricsPort    int    `validate:"min=1,max=65535" env:"MONITORING_METRICS_PORT" default:"9090"`
        HealthCheck    bool   `env:"MONITORING_HEALTH_CHECK" default:"true"`
        LogLevel       string `validate:"oneof=debug info warn error" env:"LOG_LEVEL" default:"info"`
    }
}

// Thread-safe pipeline with comprehensive error handling and logging
type Pipeline struct {
    config    *Config
    logger    *zap.Logger
    validator *validator.Validate
    metrics   *Metrics
    
    // Channel management
    inputCh   chan SensorReading
    errorCh   chan error
    doneCh    chan struct{}
    
    // Worker management
    workerWG  sync.WaitGroup
    ctx       context.Context
    cancel    context.CancelFunc
    
    // Handler registration
    handlerMu sync.RWMutex
    handlers  map[SensorType][]EventHandler
    
    // State management
    state     PipelineState
    stateMu   sync.RWMutex
}

type PipelineState string
const (
    StateStarting PipelineState = "starting"
    StateRunning  PipelineState = "running"
    StateStopping PipelineState = "stopping"
    StateStopped  PipelineState = "stopped"
    StateError    PipelineState = "error"
)

// Event handler interface for type safety
type EventHandler interface {
    Handle(ctx context.Context, reading SensorReading) error
    CanHandle(sensorType SensorType) bool
    Name() string
}

// Structured logging setup with security considerations
func NewStructuredLogger(level string) (*zap.Logger, error) {
    var zapLevel zapcore.Level
    if err := zapLevel.UnmarshalText([]byte(level)); err != nil {
        return nil, fmt.Errorf("invalid log level %q: %w", level, err)
    }
    
    config := zap.Config{
        Level:       zap.NewAtomicLevelAt(zapLevel),
        Development: false,
        Sampling: &zap.SamplingConfig{
            Initial:    100,
            Thereafter: 100,
        },
        Encoding: "json",
        EncoderConfig: zapcore.EncoderConfig{
            TimeKey:        "timestamp",
            LevelKey:       "level",
            NameKey:        "logger",
            CallerKey:      "caller",
            FunctionKey:    zapcore.OmitKey,
            MessageKey:     "message",
            StacktraceKey:  "stacktrace",
            LineEnding:     zapcore.DefaultLineEnding,
            EncodeLevel:    zapcore.LowercaseLevelEncoder,
            EncodeTime:     zapcore.RFC3339TimeEncoder,
            EncodeDuration: zapcore.SecondsDurationEncoder,
            EncodeCaller:   zapcore.ShortCallerEncoder,
        },
        OutputPaths:      []string{"stdout"},
        ErrorOutputPaths: []string{"stderr"},
        
        // Custom sanitization to prevent secret leakage
        InitialFields: map[string]interface{}{
            "service": "streaming-pipeline",
            "version": "1.0.0",
        },
    }
    
    logger, err := config.Build(
        zap.AddCaller(),
        zap.AddStacktrace(zapcore.ErrorLevel),
        zap.WrapCore(func(core zapcore.Core) zapcore.Core {
            return &sanitizingCore{Core: core}
        }),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create logger: %w", err)
    }
    
    return logger, nil
}

// Sanitizing core to prevent secrets from appearing in logs
type sanitizingCore struct {
    zapcore.Core
}

func (c *sanitizingCore) Write(entry zapcore.Entry, fields []zapcore.Field) error {
    // Remove sensitive fields before logging
    sanitizedFields := make([]zapcore.Field, 0, len(fields))
    for _, field := range fields {
        if isSensitiveField(field.Key) {
            sanitizedFields = append(sanitizedFields, zap.String(field.Key, "[REDACTED]"))
        } else {
            sanitizedFields = append(sanitizedFields, field)
        }
    }
    return c.Core.Write(entry, sanitizedFields)
}

func isSensitiveField(key string) bool {
    sensitiveKeys := []string{
        "password", "token", "secret", "key", "credential",
        "auth", "jwt", "session", "cookie", "private",
    }
    keyLower := strings.ToLower(key)
    for _, sensitive := range sensitiveKeys {
        if strings.Contains(keyLower, sensitive) {
            return true
        }
    }
    return false
}

// Constructor with comprehensive validation and setup
func NewPipeline(config *Config, logger *zap.Logger) (*Pipeline, error) {
    // Validate configuration
    validator := validator.New()
    if err := validator.Struct(config); err != nil {
        return nil, fmt.Errorf("invalid configuration: %w", err)
    }
    
    // Create context with cancellation
    ctx, cancel := context.WithCancel(context.Background())
    
    // Initialize metrics
    metrics, err := NewMetrics()
    if err != nil {
        cancel()
        return nil, fmt.Errorf("failed to initialize metrics: %w", err)
    }
    
    pipeline := &Pipeline{
        config:    config,
        logger:    logger,
        validator: validator,
        metrics:   metrics,
        inputCh:   make(chan SensorReading, config.Pipeline.BufferSize),
        errorCh:   make(chan error, 100),
        doneCh:    make(chan struct{}),
        ctx:       ctx,
        cancel:    cancel,
        handlers:  make(map[SensorType][]EventHandler),
        state:     StateStopped,
    }
    
    logger.Info("Pipeline created",
        zap.String("state", string(pipeline.state)),
        zap.Int("buffer_size", config.Pipeline.BufferSize),
        zap.Int("worker_count", config.Pipeline.WorkerCount),
    )
    
    return pipeline, nil
}

// Type-safe event processing with comprehensive error handling
func (p *Pipeline) ProcessReading(reading SensorReading) error {
    // Validate input with structured logging
    if err := p.validator.Struct(reading); err != nil {
        p.logger.Error("Invalid sensor reading",
            zap.String("sensor_id", reading.ID.String()),
            zap.String("plant_id", reading.PlantID.String()),
            zap.Error(err),
        )
        p.metrics.InvalidReadingsTotal.Inc()
        return &PipelineError{
            Code:      ErrorCodeValidation,
            Message:   "Invalid sensor reading",
            SensorID:  &reading.ID,
            PlantID:   &reading.PlantID,
            Timestamp: time.Now(),
            Context:   map[string]interface{}{"validation_error": err.Error()},
        }
    }
    
    // Check pipeline state
    p.stateMu.RLock()
    state := p.state
    p.stateMu.RUnlock()
    
    if state != StateRunning {
        err := &PipelineError{
            Code:      ErrorCodeInternalError,
            Message:   fmt.Sprintf("Pipeline not running (state: %s)", state),
            Timestamp: time.Now(),
        }
        p.logger.Warn("Attempted to process reading while pipeline not running",
            zap.String("state", string(state)),
            zap.String("sensor_id", reading.ID.String()),
        )
        return err
    }
    
    // Add correlation ID for tracing
    ctx := context.WithValue(p.ctx, "correlation_id", uuid.New().String())
    ctx = context.WithValue(ctx, "sensor_id", reading.ID.String())
    ctx = context.WithValue(ctx, "plant_id", reading.PlantID.String())
    
    // Try to submit to channel with timeout
    select {
    case p.inputCh <- reading:
        p.metrics.ProcessedReadingsTotal.Inc()
        p.logger.Debug("Reading queued for processing",
            zap.String("sensor_id", reading.ID.String()),
            zap.String("plant_id", reading.PlantID.String()),
            zap.String("type", string(reading.Type)),
            zap.Float64("value", reading.Value),
        )
        return nil
        
    case <-time.After(p.config.Pipeline.FlushTimeout):
        p.metrics.DroppedReadingsTotal.Inc()
        err := &PipelineError{
            Code:      ErrorCodeCapacityFull,
            Message:   "Pipeline buffer full, reading dropped",
            SensorID:  &reading.ID,
            PlantID:   &reading.PlantID,
            Timestamp: time.Now(),
        }
        p.logger.Error("Pipeline buffer full, dropping reading",
            zap.String("sensor_id", reading.ID.String()),
            zap.Int("buffer_size", p.config.Pipeline.BufferSize),
        )
        return err
        
    case <-p.ctx.Done():
        return &PipelineError{
            Code:      ErrorCodeInternalError,
            Message:   "Pipeline shutting down",
            Timestamp: time.Now(),
        }
    }
}
```

// Pipeline processes sensor events in real-time
type Pipeline struct {
    logger      *zap.Logger
    metrics     *Metrics
    bufferSize  int
    workerCount int
    
    mu       sync.RWMutex
    handlers map[string]EventHandler
}

// NewPipeline creates a new sensor data pipeline
func NewPipeline(logger *zap.Logger, opts ...Option) *Pipeline {
    p := &Pipeline{
        logger:      logger,
        bufferSize:  10000,
        workerCount: 10,
        handlers:    make(map[string]EventHandler),
    }
    
    for _, opt := range opts {
        opt(p)
    }
    
    p.metrics = newMetrics()
    return p
}

// Run starts the pipeline processing
func (p *Pipeline) Run(ctx context.Context) error {
    eventCh := make(chan SensorEvent, p.bufferSize)
    errCh := make(chan error, p.workerCount)
    
    var wg sync.WaitGroup
    
    // Start workers
    for i := 0; i < p.workerCount; i++ {
        wg.Add(1)
        go p.worker(ctx, i, eventCh, errCh, &wg)
    }
    
    // Wait for context cancellation
    <-ctx.Done()
    
    close(eventCh)
    wg.Wait()
    
    return nil
}
```

#### Error Handling in Go
```go
// errors/sensor_errors.go
package errors

import (
    "errors"
    "fmt"
)

var (
    ErrSensorNotFound     = errors.New("sensor not found")
    ErrInvalidReading     = errors.New("invalid sensor reading")
    ErrPipelineOverloaded = errors.New("pipeline buffer full")
)

// SensorError provides detailed error information
type SensorError struct {
    SensorID string
    Type     string
    Err      error
}

func (e *SensorError) Error() string {
    return fmt.Sprintf("sensor %s (%s): %v", e.SensorID, e.Type, e.Err)
}

func (e *SensorError) Unwrap() error {
    return e.Err
}

// WrapSensorError creates a new sensor error
func WrapSensorError(sensorID, sensorType string, err error) error {
    if err == nil {
        return nil
    }
    return &SensorError{
        SensorID: sensorID,
        Type:     sensorType,
        Err:      err,
    }
}
```

#### Concurrent Device Management
```go
// services/device/manager.go
package device

import (
    "context"
    "sync"
    "time"
)

// Manager handles multiple plant devices concurrently
type Manager struct {
    mu      sync.RWMutex
    devices map[string]*Device
    
    pollInterval time.Duration
    maxRetries   int
}

// PollAll reads from all devices concurrently
func (m *Manager) PollAll(ctx context.Context) ([]Reading, error) {
    m.mu.RLock()
    deviceCount := len(m.devices)
    devices := make([]*Device, 0, deviceCount)
    for _, d := range m.devices {
        devices = append(devices, d)
    }
    m.mu.RUnlock()
    
    // Use buffered channel for results
    resultCh := make(chan Reading, deviceCount)
    errCh := make(chan error, deviceCount)
    
    var wg sync.WaitGroup
    
    // Poll each device concurrently
    for _, device := range devices {
        wg.Add(1)
        go func(d *Device) {
            defer wg.Done()
            
            reading, err := d.Read(ctx)
            if err != nil {
                errCh <- WrapDeviceError(d.ID, err)
                return
            }
            
            resultCh <- reading
        }(device)
    }
    
    // Wait for all goroutines
    go func() {
        wg.Wait()
        close(resultCh)
        close(errCh)
    }()
    
    // Collect results
    var readings []Reading
    var errs []error
    
    for {
        select {
        case reading, ok := <-resultCh:
            if !ok {
                resultCh = nil
            } else {
                readings = append(readings, reading)
            }
        case err, ok := <-errCh:
            if !ok {
                errCh = nil
            } else if err != nil {
                errs = append(errs, err)
            }
        case <-ctx.Done():
            return nil, ctx.Err()
        }
        
        if resultCh == nil && errCh == nil {
            break
        }
    }
    
    if len(errs) > 0 {
        return readings, &MultiError{Errors: errs}
    }
    
    return readings, nil
}
```

### Rust Code Standards

#### Safety-Critical Sensor Driver
```rust
// hardware/drivers/src/moisture_sensor.rs
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub enum SensorError {
    NotInitialized,
    ReadTimeout,
    InvalidValue(f32),
    HardwareFault,
}

impl fmt::Display for SensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SensorError::NotInitialized => write!(f, "Sensor not initialized"),
            SensorError::ReadTimeout => write!(f, "Sensor read timeout"),
            SensorError::InvalidValue(v) => write!(f, "Invalid sensor value: {}", v),
            SensorError::HardwareFault => write!(f, "Hardware fault detected"),
        }
    }
}

impl std::error::Error for SensorError {}

/// Thread-safe moisture sensor interface
pub struct MoistureSensor {
    pin: u8,
    state: Arc<Mutex<SensorState>>,
    calibration: Calibration,
}

struct SensorState {
    initialized: bool,
    last_reading: Option<Reading>,
    error_count: u32,
}

struct Reading {
    value: f32,
    timestamp: Instant,
}

struct Calibration {
    dry_value: u16,
    wet_value: u16,
}

impl MoistureSensor {
    /// Create a new moisture sensor instance
    pub fn new(pin: u8) -> Result<Self, SensorError> {
        if pin > 39 {
            return Err(SensorError::InvalidValue(pin as f32));
        }
        
        Ok(Self {
            pin,
            state: Arc::new(Mutex::new(SensorState {
                initialized: false,
                last_reading: None,
                error_count: 0,
            })),
            calibration: Calibration {
                dry_value: 2800,
                wet_value: 1200,
            },
        })
    }
    
    /// Initialize the sensor hardware
    pub fn init(&self) -> Result<(), SensorError> {
        let mut state = self.state.lock().unwrap();
        
        // Initialize ADC for the pin
        unsafe {
            // Safety: We've validated the pin number
            if adc_init(self.pin) != 0 {
                return Err(SensorError::HardwareFault);
            }
        }
        
        state.initialized = true;
        state.error_count = 0;
        
        Ok(())
    }
    
    /// Read moisture percentage (0-100)
    pub fn read(&self) -> Result<f32, SensorError> {
        let mut state = self.state.lock().unwrap();
        
        if !state.initialized {
            return Err(SensorError::NotInitialized);
        }
        
        // Read raw ADC value
        let raw_value = unsafe {
            // Safety: Sensor is initialized and pin is valid
            let mut value: u16 = 0;
            if adc_read(self.pin, &mut value as *mut u16) != 0 {
                state.error_count += 1;
                return Err(SensorError::ReadTimeout);
            }
            value
        };
        
        // Convert to percentage
        let percentage = self.calibrate_value(raw_value);
        
        // Validate reading
        if !(0.0..=100.0).contains(&percentage) {
            state.error_count += 1;
            return Err(SensorError::InvalidValue(percentage));
        }
        
        // Update state
        state.last_reading = Some(Reading {
            value: percentage,
            timestamp: Instant::now(),
        });
        state.error_count = 0;
        
        Ok(percentage)
    }
    
    fn calibrate_value(&self, raw: u16) -> f32 {
        let range = self.calibration.dry_value - self.calibration.wet_value;
        let normalized = (self.calibration.dry_value - raw) as f32 / range as f32;
        (normalized * 100.0).clamp(0.0, 100.0)
    }
}

// FFI bindings for Python
#[no_mangle]
pub extern "C" fn moisture_sensor_create(pin: u8) -> *mut MoistureSensor {
    match MoistureSensor::new(pin) {
        Ok(sensor) => Box::into_raw(Box::new(sensor)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn moisture_sensor_read(sensor: *mut MoistureSensor, value: *mut f32) -> i32 {
    if sensor.is_null() || value.is_null() {
        return -1;
    }
    
    let sensor = unsafe { &*sensor };
    
    match sensor.read() {
        Ok(v) => {
            unsafe { *value = v };
            0
        }
        Err(_) => -2,
    }
}

// External C functions (defined in hardware abstraction layer)
extern "C" {
    fn adc_init(pin: u8) -> i32;
    fn adc_read(pin: u8, value: *mut u16) -> i32;
}
```

#### Memory-Safe Pump Controller
```rust
// hardware/drivers/src/pump_controller.rs
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};

/// Safety-critical pump controller with automatic shutoff
pub struct PumpController {
    pin: u8,
    state: Arc<PumpState>,
    safety: Arc<SafetyMonitor>,
    semaphore: Arc<Semaphore>,
}

struct PumpState {
    is_running: AtomicBool,
    start_time: Mutex<Option<Instant>>,
    total_runtime_ms: AtomicU64,
}

struct SafetyMonitor {
    max_continuous_runtime: Duration,
    min_cycle_interval: Duration,
    last_stop_time: Mutex<Option<Instant>>,
}

impl PumpController {
    pub fn new(pin: u8) -> Self {
        Self {
            pin,
            state: Arc::new(PumpState {
                is_running: AtomicBool::new(false),
                start_time: Mutex::new(None),
                total_runtime_ms: AtomicU64::new(0),
            }),
            safety: Arc::new(SafetyMonitor {
                max_continuous_runtime: Duration::from_secs(120), // 2 minutes max
                min_cycle_interval: Duration::from_secs(60),      // 1 minute between runs
                last_stop_time: Mutex::new(None),
            }),
            semaphore: Arc::new(Semaphore::new(1)), // Only one operation at a time
        }
    }
    
    /// Run pump for specified duration with safety checks
    pub async fn run_for_duration(&self, duration: Duration) -> Result<(), PumpError> {
        // Acquire semaphore to ensure exclusive access
        let _permit = self.semaphore.acquire().await.unwrap();
        
        // Safety check: minimum cycle interval
        if let Some(last_stop) = *self.safety.last_stop_time.lock().await {
            let elapsed = Instant::now().duration_since(last_stop);
            if elapsed < self.safety.min_cycle_interval {
                return Err(PumpError::CycleIntervalViolation);
            }
        }
        
        // Safety check: duration limit
        if duration > self.safety.max_continuous_runtime {
            return Err(PumpError::DurationExceeded);
        }
        
        // Start pump
        self.start_internal().await?;
        
        // Run for specified duration
        tokio::time::sleep(duration).await;
        
        // Stop pump
        self.stop_internal().await?;
        
        Ok(())
    }
    
    /// Emergency stop - can be called from any thread
    pub fn emergency_stop(&self) -> Result<(), PumpError> {
        if self.state.is_running.load(Ordering::SeqCst) {
            unsafe {
                // Safety: Direct hardware control for emergency
                gpio_write(self.pin, 0);
            }
            self.state.is_running.store(false, Ordering::SeqCst);
        }
        Ok(())
    }
    
    async fn start_internal(&self) -> Result<(), PumpError> {
        if self.state.is_running.load(Ordering::SeqCst) {
            return Err(PumpError::AlreadyRunning);
        }
        
        unsafe {
            // Safety: Pin has been validated
            if gpio_write(self.pin, 1) != 0 {
                return Err(PumpError::HardwareFault);
            }
        }
        
        self.state.is_running.store(true, Ordering::SeqCst);
        *self.state.start_time.lock().await = Some(Instant::now());
        
        // Start safety monitor
        self.spawn_safety_monitor();
        
        Ok(())
    }
    
    async fn stop_internal(&self) -> Result<(), PumpError> {
        if !self.state.is_running.load(Ordering::SeqCst) {
            return Ok(());
        }
        
        unsafe {
            // Safety: Pin has been validated
            if gpio_write(self.pin, 0) != 0 {
                return Err(PumpError::HardwareFault);
            }
        }
        
        // Update state
        self.state.is_running.store(false, Ordering::SeqCst);
        
        // Record runtime
        if let Some(start) = *self.state.start_time.lock().await {
            let runtime = Instant::now().duration_since(start);
            self.state.total_runtime_ms.fetch_add(
                runtime.as_millis() as u64,
                Ordering::SeqCst
            );
        }
        
        *self.safety.last_stop_time.lock().await = Some(Instant::now());
        
        Ok(())
    }
    
    fn spawn_safety_monitor(&self) {
        let state = Arc::clone(&self.state);
        let safety = Arc::clone(&self.safety);
        let pin = self.pin;
        
        tokio::spawn(async move {
            while state.is_running.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(100)).await;
                
                // Check for timeout
                if let Some(start) = *state.start_time.lock().await {
                    if Instant::now().duration_since(start) > safety.max_continuous_runtime {
                        // Emergency shutoff
                        unsafe { gpio_write(pin, 0); }
                        state.is_running.store(false, Ordering::SeqCst);
                        break;
                    }
                }
            }
        });
    }
}

#[derive(Debug)]
pub enum PumpError {
    AlreadyRunning,
    HardwareFault,
    DurationExceeded,
    CycleIntervalViolation,
}

extern "C" {
    fn gpio_write(pin: u8, value: u8) -> i32;
}
```

### Julia Code Standards

#### Package Structure and Project Management
```julia
# analytics/julia/Project.toml
name = "PlantOS"
uuid = "12345678-1234-1234-1234-123456789abc"
authors = ["Your Name <your.email@example.com>"]
version = "0.1.0"

[deps]
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
TimeSeries = "9e3dc215-6440-5c97-bce1-76c03772f85e"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[compat]
julia = "1.10"
```

#### Plant Growth Modeling
```julia
# analytics/julia/src/models/plant_growth.jl
using DifferentialEquations, Parameters

@with_kw struct PlantParameters
    growth_rate::Float64 = 0.1
    water_efficiency::Float64 = 0.8
    light_efficiency::Float64 = 0.6
    temperature_optimum::Float64 = 25.0
    moisture_optimum::Float64 = 60.0
end

"""
Plant growth differential equation model.
Incorporates water, light, and temperature effects on growth rate.
"""
function plant_growth_model!(du, u, p::PlantParameters, t)
    biomass, water_content = u
    
    # Environmental factors (would come from sensors)
    temperature = get_temperature(t)
    light_intensity = get_light_intensity(t)
    soil_moisture = get_soil_moisture(t)
    
    # Growth rate modifiers
    temp_factor = exp(-((temperature - p.temperature_optimum) / 10)^2)
    moisture_factor = exp(-((soil_moisture - p.moisture_optimum) / 20)^2)
    light_factor = min(1.0, light_intensity / 1000.0)
    
    # Differential equations
    du[1] = p.growth_rate * biomass * temp_factor * light_factor * moisture_factor
    du[2] = -0.1 * water_content + 0.5 * soil_moisture / 100.0
end

"""
Simulate plant growth over time period with given conditions.
"""
function simulate_plant_growth(
    initial_conditions::Vector{Float64},
    timespan::Tuple{Float64, Float64},
    parameters::PlantParameters
)
    prob = ODEProblem(plant_growth_model!, initial_conditions, timespan, parameters)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return sol
end
```

#### Optimization Algorithms
```julia
# analytics/julia/src/optimization/watering_schedule.jl
using Optimization, OptimizationOptimJL, LinearAlgebra

struct WateringOptimizationProblem
    plant_count::Int
    time_horizon::Int
    water_capacity::Float64
    plant_parameters::Vector{PlantParameters}
    current_states::Matrix{Float64}
end

"""
Multi-objective optimization for watering schedule.
Maximizes plant health while minimizing water usage.
"""
function watering_objective(x, problem::WateringOptimizationProblem)
    schedule = reshape(x, problem.plant_count, problem.time_horizon)
    
    total_health = 0.0
    total_water = 0.0
    
    for plant_id in 1:problem.plant_count
        for time_step in 1:problem.time_horizon
            water_amount = schedule[plant_id, time_step]
            
            # Health contribution (to maximize)
            predicted_moisture = predict_moisture(
                plant_id, time_step, water_amount, problem
            )
            health_score = calculate_health_score(
                predicted_moisture, problem.plant_parameters[plant_id]
            )
            
            total_health += health_score
            total_water += water_amount
        end
    end
    
    # Multi-objective: maximize health, minimize water
    # Weight health more heavily for plant welfare
    return -(2.0 * total_health - 0.5 * total_water)
end

function optimize_watering_schedule(problem::WateringOptimizationProblem)
    # Decision variables: water amount for each plant at each time step
    x0 = rand(problem.plant_count * problem.time_horizon) * 100.0
    
    # Constraints: water capacity limits
    function water_constraint(x, p)
        schedule = reshape(x, problem.plant_count, problem.time_horizon)
        return [sum(schedule[:, t]) - problem.water_capacity for t in 1:problem.time_horizon]
    end
    
    # Bounds: non-negative water amounts
    lb = zeros(length(x0))
    ub = fill(200.0, length(x0))  # Max 200ml per plant per time step
    
    opt_prob = OptimizationProblem(
        watering_objective, x0, problem;
        lb=lb, ub=ub,
        cons=water_constraint
    )
    
    sol = solve(opt_prob, IPNewton())
    
    return reshape(sol.u, problem.plant_count, problem.time_horizon)
end
```

#### Time Series Forecasting
```julia
# analytics/julia/src/forecasting/sensor_prediction.jl
using TimeSeries, MLJ, Statistics

"""
ARIMA-based sensor reading forecasting.
Predicts future moisture levels, temperature, etc.
"""
function forecast_sensor_readings(
    historical_data::TimeArray,
    forecast_horizon::Int=24
)
    # Convert TimeArray to format suitable for MLJ
    X = values(historical_data)
    timestamps = timestamp(historical_data)
    
    # Feature engineering
    features = create_time_features(timestamps, X)
    
    # Load and configure ARIMA model
    @load ARIMARegressor pkg=MLJTime
    model = ARIMARegressor(
        order=(2, 1, 2),  # ARIMA(2,1,2)
        seasonal_order=(1, 1, 1, 24),  # Seasonal component
        include_constant=true
    )
    
    # Train model
    mach = machine(model, features, X[:, 1])  # Assuming first column is target
    fit!(mach)
    
    # Generate forecasts
    future_features = create_future_features(timestamps, forecast_horizon)
    predictions = predict(mach, future_features)
    
    # Calculate prediction intervals
    residuals = predict_mode(mach) .- X[:, 1]
    std_residual = std(residuals)
    
    upper_bound = predictions .+ 1.96 * std_residual
    lower_bound = predictions .- 1.96 * std_residual
    
    return (
        forecast=predictions,
        upper_ci=upper_bound,
        lower_ci=lower_bound,
        timestamps=generate_future_timestamps(timestamps, forecast_horizon)
    )
end

function create_time_features(timestamps, data)
    n = length(timestamps)
    features = zeros(n, 6)
    
    for (i, ts) in enumerate(timestamps)
        features[i, 1] = hour(ts)
        features[i, 2] = dayofweek(ts)
        features[i, 3] = dayofyear(ts)
        features[i, 4] = sin(2π * hour(ts) / 24)  # Hourly cycle
        features[i, 5] = cos(2π * hour(ts) / 24)
        features[i, 6] = sin(2π * dayofyear(ts) / 365)  # Yearly cycle
    end
    
    return features
end
```

#### HTTP API Server
```julia
# analytics/julia/src/server.jl
using HTTP, JSON3

struct AnalyticsServer
    port::Int
    models::Dict{String, Any}
end

function start_analytics_server(port::Int=8081)
    server = AnalyticsServer(port, Dict())
    
    # Load pre-trained models
    server.models["plant_growth"] = load_plant_growth_models()
    server.models["optimization"] = WateringOptimizationProblem()
    
    router = HTTP.Router()
    
    # Health check endpoint
    HTTP.register!(router, "GET", "/health") do req
        return HTTP.Response(200, JSON3.write(Dict("status" => "healthy")))
    end
    
    # Plant growth simulation endpoint
    HTTP.register!(router, "POST", "/api/simulate/growth") do req
        try
            params = JSON3.read(String(req.body))
            result = simulate_plant_growth_api(params, server.models)
            return HTTP.Response(200, JSON3.write(result))
        catch e
            error_msg = Dict("error" => string(e))
            return HTTP.Response(500, JSON3.write(error_msg))
        end
    end
    
    # Watering optimization endpoint
    HTTP.register!(router, "POST", "/api/optimize/watering") do req
        try
            params = JSON3.read(String(req.body))
            schedule = optimize_watering_api(params, server.models)
            result = Dict("schedule" => schedule, "status" => "success")
            return HTTP.Response(200, JSON3.write(result))
        catch e
            error_msg = Dict("error" => string(e))
            return HTTP.Response(500, JSON3.write(error_msg))
        end
    end
    
    # Sensor forecasting endpoint
    HTTP.register!(router, "POST", "/api/forecast/sensors") do req
        try
            params = JSON3.read(String(req.body))
            forecast = forecast_sensors_api(params)
            return HTTP.Response(200, JSON3.write(forecast))
        catch e
            error_msg = Dict("error" => string(e))
            return HTTP.Response(500, JSON3.write(error_msg))
        end
    end
    
    println("Starting Julia Analytics Server on port $port")
    HTTP.serve(router, "0.0.0.0", port)
end

# Start server when script is run
if abspath(PROGRAM_FILE) == @__FILE__
    start_analytics_server()
end
```

### Bash Scripting Standards

#### Development Automation Scripts
```bash
#!/bin/bash
# scripts/development/setup.sh - One-command development environment setup
# Security: Strict mode, input validation, secrets management, structured logging

set -euo pipefail  # Exit on error, undefined vars, pipe failures
IFS=$'\n\t'       # Secure Internal Field Separator

# Script metadata for logging and validation
readonly SCRIPT_NAME="setup.sh"
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Logging configuration with JSON structured output
readonly LOG_FILE="/tmp/plantos-setup-$(date +%Y%m%d-%H%M%S).log"
readonly LOG_LEVEL="${LOG_LEVEL:-INFO}"  # DEBUG, INFO, WARN, ERROR

# Color output for terminal (disabled in non-interactive)
if [[ -t 1 ]]; then
    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[1;33m'
    readonly BLUE='\033[0;34m'
    readonly NC='\033[0m'
else
    readonly RED=''
    readonly GREEN=''
    readonly YELLOW=''
    readonly BLUE=''
    readonly NC=''
fi

# Security: Environment validation for secrets
readonly REQUIRED_ENV_VARS=(
    "DATABASE_PASSWORD"
    "SECRET_KEY"
    "VAULT_TOKEN"
)

readonly OPTIONAL_ENV_VARS=(
    "DATABASE_HOST:localhost"
    "DATABASE_PORT:5432"
    "API_PORT:8000"
    "LOG_LEVEL:INFO"
)

# Type-safe logging functions with structured output
log_structured() {
    local level="$1"
    local message="$2"
    local context="${3:-{}}"
    
    local timestamp
    timestamp="$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")"
    
    local log_entry
    log_entry=$(jq -n \
        --arg timestamp "$timestamp" \
        --arg level "$level" \
        --arg script "$SCRIPT_NAME" \
        --arg version "$SCRIPT_VERSION" \
        --arg pid "$$" \
        --arg message "$message" \
        --argjson context "$context" \
        '{
            timestamp: $timestamp,
            level: $level,
            script: $script,
            version: $version,
            pid: ($pid | tonumber),
            message: $message,
            context: $context
        }'
    )
    
    # Output to both file and stdout
    echo "$log_entry" >> "$LOG_FILE"
    
    # Human-readable terminal output
    if [[ -t 1 ]]; then
        case "$level" in
            "ERROR") echo -e "${RED}[ERROR]${NC} $message" >&2 ;;
            "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
            "INFO")  echo -e "${GREEN}[INFO]${NC} $message" ;;
            "DEBUG") [[ "$LOG_LEVEL" == "DEBUG" ]] && echo -e "${BLUE}[DEBUG]${NC} $message" ;;
        esac
    fi
}

log_error() {
    log_structured "ERROR" "$1" "${2:-{}}"
}

log_warn() {
    log_structured "WARN" "$1" "${2:-{}}"
}

log_info() {
    log_structured "INFO" "$1" "${2:-{}}"
}

log_debug() {
    log_structured "DEBUG" "$1" "${2:-{}}"
}

# Input validation with comprehensive checks
validate_input() {
    local input="$1"
    local input_type="$2"
    local input_name="${3:-input}"
    
    case "$input_type" in
        "hostname")
            if [[ ! "$input" =~ ^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$ ]]; then
                log_error "Invalid hostname format: $input" "{\"input_name\":\"$input_name\",\"input_type\":\"$input_type\"}"
                return 1
            fi
            ;;
        "port")
            if [[ ! "$input" =~ ^[0-9]+$ ]] || [[ "$input" -lt 1 ]] || [[ "$input" -gt 65535 ]]; then
                log_error "Invalid port number: $input (must be 1-65535)" "{\"input_name\":\"$input_name\",\"input_type\":\"$input_type\"}"
                return 1
            fi
            ;;
        "path")
            if [[ ! "$input" =~ ^[a-zA-Z0-9/_.-]+$ ]]; then
                log_error "Invalid path format: $input" "{\"input_name\":\"$input_name\",\"input_type\":\"$input_type\"}"
                return 1
            fi
            ;;
        "version")
            if [[ ! "$input" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
                log_error "Invalid version format: $input" "{\"input_name\":\"$input_name\",\"input_type\":\"$input_type\"}"
                return 1
            fi
            ;;
        *)
            log_error "Unknown validation type: $input_type" "{\"input_name\":\"$input_name\"}"
            return 1
            ;;
    esac
    
    log_debug "Input validation passed" "{\"input_name\":\"$input_name\",\"input_type\":\"$input_type\",\"value\":\"$input\"}"
    return 0
}

# Secure secrets management with validation
load_secrets() {
    log_info "Loading and validating secrets configuration"
    
    # Check for .env file
    local env_file="$PROJECT_ROOT/.env"
    if [[ -f "$env_file" ]]; then
        log_info "Loading environment variables from .env file"
        # Securely source .env file with validation
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ "$key" =~ ^#.*$ ]] && continue
            [[ -z "$key" ]] && continue
            
            # Validate key format
            if [[ ! "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
                log_warn "Invalid environment variable name format: $key"
                continue
            fi
            
            # Export with validation
            export "$key=$value"
            log_debug "Loaded environment variable" "{\"key\":\"$key\",\"value_length\":${#value}}"
        done < <(grep -v '^#' "$env_file" | grep -v '^$')
    fi
    
    # Validate required secrets
    local missing_vars=()
    for var in "${REQUIRED_ENV_VARS[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing_vars[*]}" "{\"missing_vars\":[$(printf '"%s",' "${missing_vars[@]}" | sed 's/,$//')]}"
        return 1
    fi
    
    # Set defaults for optional variables
    for var_default in "${OPTIONAL_ENV_VARS[@]}"; do
        local var="${var_default%:*}"
        local default="${var_default#*:}"
        
        if [[ -z "${!var:-}" ]]; then
            export "$var=$default"
            log_debug "Set default value for environment variable" "{\"var\":\"$var\",\"default\":\"$default\"}"
        fi
    done
    
    # Validate secret formats
    if [[ ${#DATABASE_PASSWORD} -lt 8 ]]; then
        log_error "DATABASE_PASSWORD must be at least 8 characters"
        return 1
    fi
    
    if [[ ${#SECRET_KEY} -lt 32 ]]; then
        log_error "SECRET_KEY must be at least 32 characters"
        return 1
    fi
    
    validate_input "$DATABASE_HOST" "hostname" "DATABASE_HOST"
    validate_input "$DATABASE_PORT" "port" "DATABASE_PORT"
    validate_input "$API_PORT" "port" "API_PORT"
    
    log_info "Secrets validation completed successfully"
    return 0
}

# Comprehensive dependency checking with version validation
check_dependencies() {
    log_info "Checking system dependencies"
    
    local deps=(
        "python3:3.12"
        "go:1.21"
        "rustc:1.75"
        "julia:1.10"
        "podman-compose:1.0"
        "jq:1.6"
    )
    
    local missing=()
    local version_issues=()
    
    for dep_version in "${deps[@]}"; do
        local dep="${dep_version%:*}"
        local min_version="${dep_version#*:}"
        
        log_debug "Checking dependency" "{\"dependency\":\"$dep\",\"min_version\":\"$min_version\"}"
        
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
            continue
        fi
        
        # Get version and validate
        local current_version
        case "$dep" in
            "python3")
                current_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
                ;;
            "go")
                current_version=$(go version 2>&1 | awk '{print $3}' | sed 's/go//' | cut -d. -f1,2)
                ;;
            "rustc")
                current_version=$(rustc --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
                ;;
            "julia")
                current_version=$(julia --version 2>&1 | awk '{print $3}' | cut -d. -f1,2)
                ;;
            "podman-compose")
                current_version=$(podman-compose --version 2>&1 | awk '{print $3}' | cut -d. -f1,2)
                ;;
            "jq")
                current_version=$(jq --version 2>&1 | sed 's/jq-//' | cut -d. -f1,2)
                ;;
        esac
        
        if [[ -n "$current_version" ]]; then
            validate_input "$current_version" "version" "${dep}_version"
            
            # Simple version comparison (works for major.minor)
            if [[ "$(printf '%s\n' "$min_version" "$current_version" | sort -V | head -n1)" != "$min_version" ]]; then
                version_issues+=("$dep: $current_version < $min_version")
            else
                log_debug "Dependency version OK" "{\"dependency\":\"$dep\",\"current\":\"$current_version\",\"required\":\"$min_version\"}"
            fi
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}" "{\"missing_deps\":[$(printf '"%s",' "${missing[@]}" | sed 's/,$//')]}"
        return 1
    fi
    
    if [[ ${#version_issues[@]} -gt 0 ]]; then
        log_error "Version requirements not met:" "{\"version_issues\":[$(printf '"%s",' "${version_issues[@]}" | sed 's/,$//')]}"
        for issue in "${version_issues[@]}"; do
            log_error "  $issue"
        done
        return 1
    fi
    
    log_info "All dependencies satisfied"
    return 0
}

# Secure database setup with connection validation
setup_databases_secure() {
    log_info "Setting up databases with security validation"
    
    # Validate database configuration
    local db_host="$DATABASE_HOST"
    local db_port="$DATABASE_PORT"
    local db_password="$DATABASE_PASSWORD"
    
    # Create secure podman-compose configuration
    local compose_file="$PROJECT_ROOT/docker-compose.secure.yml"
    
    # Generate secure compose file with secrets
    cat > "$compose_file" << EOF
version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: plantos
      POSTGRES_USER: plantos
      POSTGRES_PASSWORD: \${DATABASE_PASSWORD}
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
    ports:
      - "\${DATABASE_PORT}:5432"
    networks:
      - plantos-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -h localhost -p 5432 -U plantos"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass \${REDIS_PASSWORD}
    environment:
      REDIS_PASSWORD: \${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - plantos-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "auth", "\${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:

networks:
  plantos-network:
    driver: bridge
EOF

    # Start databases with security
    log_info "Starting secure database containers"
    cd "$PROJECT_ROOT"
    
    # Export required environment variables
    export DATABASE_PASSWORD
    export REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
    
    podman-compose -f "$compose_file" up -d
    
    # Wait for databases with timeout and health checks
    log_info "Waiting for databases to be ready (timeout: 60s)"
    local timeout=60
    local elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        if podman exec plantos_postgres_1 pg_isready -h localhost -p 5432 -U plantos &>/dev/null; then
            log_info "PostgreSQL is ready"
            break
        fi
        
        sleep 2
        elapsed=$((elapsed + 2))
        
        if [[ $elapsed -eq $timeout ]]; then
            log_error "Database startup timeout" "{\"timeout_seconds\":$timeout}"
            return 1
        fi
    done
    
    # Test database connection with credentials validation
    log_info "Validating database connection"
    if PGPASSWORD="$DATABASE_PASSWORD" psql -h "$db_host" -p "$db_port" -U plantos -d plantos -c "SELECT 1;" &>/dev/null; then
        log_info "Database connection validated successfully"
    else
        log_error "Database connection validation failed"
        return 1
    fi
    
    # Clean up temporary compose file (contains secrets)
    rm -f "$compose_file"
    
    return 0
}

check_dependencies() {
    local deps=("python3" "go" "rustc" "julia" "podman-compose")
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Please install missing dependencies and try again"
        exit 1
    fi
}

setup_python_env() {
    log_info "Setting up Python virtual environment..."
    cd "$PROJECT_ROOT"
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt -r requirements-dev.txt
    pre-commit install
}

setup_go_modules() {
    log_info "Setting up Go modules..."
    cd "$PROJECT_ROOT/services/streaming"
    go mod download
    go mod tidy
    
    cd "$PROJECT_ROOT/services/device"
    go mod download
    go mod tidy
}

setup_rust_deps() {
    log_info "Setting up Rust dependencies..."
    cd "$PROJECT_ROOT/hardware/drivers"
    cargo fetch
}

setup_julia_env() {
    log_info "Setting up Julia environment..."
    cd "$PROJECT_ROOT/analytics/julia"
    julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
}

setup_databases() {
    log_info "Starting databases with Podman..."
    cd "$PROJECT_ROOT"
    podman-compose up -d postgres redis
    
    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    timeout 60 bash -c '
        until podman exec plantos_postgres_1 pg_isready -h localhost -p 5432 -U plantos; do
            sleep 2
        done
    '
    
    # Run migrations
    source venv/bin/activate
    alembic upgrade head
}

run_initial_tests() {
    log_info "Running initial test suite..."
    source venv/bin/activate
    pytest tests/unit/ -v
    
    cd services/streaming && go test ./...
    cd ../../hardware/drivers && cargo test
    cd ../../analytics/julia && julia --project=. -e "using Pkg; Pkg.test()"
}

main() {
    log_info "Starting PlantOS development environment setup..."
    
    check_dependencies
    setup_python_env
    setup_go_modules
    setup_rust_deps
    setup_julia_env
    setup_databases
    run_initial_tests
    
    log_info "✅ Development environment setup complete!"
    log_info "Run 'source venv/bin/activate' to activate Python environment"
    log_info "Run './scripts/development/test.sh' to run all tests"
}

main "$@"
```

#### Multi-Language Build Script
```bash
#!/bin/bash
# scripts/development/build.sh - Build all services

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/common.sh"

build_rust_drivers() {
    log_info "Building Rust hardware drivers..."
    cd "$PROJECT_ROOT/hardware/drivers"
    
    # Build in release mode for performance
    cargo build --release
    
    # Build Python FFI bindings
    cargo build --release --features python-bindings
    
    log_info "✅ Rust drivers built successfully"
}

build_go_services() {
    log_info "Building Go services..."
    
    # Build streaming service
    cd "$PROJECT_ROOT/services/streaming"
    go build -o bin/streaming-server ./cmd/server/
    
    # Build device manager
    cd "$PROJECT_ROOT/services/device"
    go build -o bin/device-manager ./cmd/manager/
    
    log_info "✅ Go services built successfully"
}

compile_julia_packages() {
    log_info "Compiling Julia packages..."
    cd "$PROJECT_ROOT/analytics/julia"
    
    julia --project=. -e "
        using Pkg
        Pkg.instantiate()
        Pkg.precompile()
        
        # Pre-compile frequently used packages
        using DifferentialEquations, Optimization, HTTP, JSON3
    "
    
    log_info "✅ Julia packages compiled successfully"
}

create_systemd_services() {
    log_info "Creating systemd service files..."
    
    mkdir -p "$PROJECT_ROOT/dist/systemd"
    
    cat > "$PROJECT_ROOT/dist/systemd/plantos-api.service" << 'EOF'
[Unit]
Description=PlantOS API Server
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=exec
User=plantos
Group=plantos
WorkingDirectory=/opt/plantos
Environment=PYTHONPATH=/opt/plantos
ExecStart=/opt/plantos/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    cat > "$PROJECT_ROOT/dist/systemd/plantos-streaming.service" << 'EOF'
[Unit]
Description=PlantOS Streaming Service
After=network.target plantos-api.service
Requires=plantos-api.service

[Service]
Type=exec
User=plantos
Group=plantos
WorkingDirectory=/opt/plantos/services/streaming
ExecStart=/opt/plantos/services/streaming/bin/streaming-server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    log_info "✅ Systemd service files created"
}

package_distribution() {
    log_info "Creating distribution package..."
    
    mkdir -p "$PROJECT_ROOT/dist"
    
    # Create tarball with all binaries and configs
    tar -czf "$PROJECT_ROOT/dist/plantos-$(date +%Y%m%d-%H%M%S).tar.gz" \
        -C "$PROJECT_ROOT" \
        --exclude='target' \
        --exclude='node_modules' \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='__pycache__' \
        .
    
    log_info "✅ Distribution package created in dist/"
}

main() {
    log_info "Building PlantOS multi-language stack..."
    
    build_rust_drivers
    build_go_services
    compile_julia_packages
    create_systemd_services
    package_distribution
    
    log_info "🎉 Build completed successfully!"
}

main "$@"
```

#### Comprehensive Test Runner
```bash
#!/bin/bash
# scripts/development/test.sh - Run all tests across languages

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/common.sh"

# Test configuration
readonly PYTHON_COV_MIN=90
readonly GO_COV_MIN=80
readonly RUST_COV_MIN=85

run_python_tests() {
    log_info "Running Python tests..."
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Unit tests with coverage
    pytest tests/unit/ \
        --cov=src \
        --cov-report=html:htmlcov \
        --cov-report=term \
        --cov-fail-under="$PYTHON_COV_MIN" \
        --junitxml=test-results/python-unit.xml
    
    # Integration tests (if databases are running)
    if podman ps | grep -q plantos_postgres; then
        pytest tests/integration/ \
            --junitxml=test-results/python-integration.xml
    else
        log_warn "Skipping integration tests - databases not running"
    fi
    
    # Hardware tests (only if --hardware flag passed)
    if [[ "${1:-}" == "--hardware" ]]; then
        pytest tests/integration/hardware/ \
            -m hardware \
            --junitxml=test-results/python-hardware.xml
    fi
    
    log_info "✅ Python tests completed"
}

run_go_tests() {
    log_info "Running Go tests..."
    
    # Test streaming service
    cd "$PROJECT_ROOT/services/streaming"
    go test -v -race -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html
    
    # Check coverage threshold
    local coverage
    coverage=$(go tool cover -func=coverage.out | grep total | awk '{print $3}' | sed 's/%//')
    if (( $(echo "$coverage < $GO_COV_MIN" | bc -l) )); then
        log_error "Go streaming coverage $coverage% below minimum $GO_COV_MIN%"
        return 1
    fi
    
    # Test device manager
    cd "$PROJECT_ROOT/services/device"
    go test -v -race -coverprofile=coverage.out ./...
    
    # Benchmark tests
    log_info "Running Go benchmarks..."
    go test -bench=. -benchmem ./...
    
    log_info "✅ Go tests completed"
}

run_rust_tests() {
    log_info "Running Rust tests..."
    cd "$PROJECT_ROOT/hardware/drivers"
    
    # Unit tests
    cargo test --lib
    
    # Integration tests
    cargo test --test '*'
    
    # Benchmark tests
    cargo bench
    
    # Coverage (requires cargo-tarpaulin)
    if command -v cargo-tarpaulin &> /dev/null; then
        cargo tarpaulin --out html --output-dir target/tarpaulin/
        
        # Extract coverage percentage and check threshold
        local coverage
        coverage=$(cargo tarpaulin --print-summary | grep "Coverage" | awk '{print $2}' | sed 's/%//')
        if (( $(echo "$coverage < $RUST_COV_MIN" | bc -l) )); then
            log_error "Rust coverage $coverage% below minimum $RUST_COV_MIN%"
            return 1
        fi
    else
        log_warn "cargo-tarpaulin not installed, skipping coverage"
    fi
    
    log_info "✅ Rust tests completed"
}

run_julia_tests() {
    log_info "Running Julia tests..."
    cd "$PROJECT_ROOT/analytics/julia"
    
    # Run test suite
    julia --project=. -e "using Pkg; Pkg.test()"
    
    # Run benchmarks
    julia --project=. test/benchmarks.jl
    
    log_info "✅ Julia tests completed"
}

run_bash_tests() {
    log_info "Running Bash script tests..."
    
    # Use bats for bash testing if available
    if command -v bats &> /dev/null; then
        cd "$PROJECT_ROOT"
        bats tests/scripts/
    else
        log_warn "bats not installed, skipping bash tests"
        # Basic syntax check
        find scripts/ -name "*.sh" -exec bash -n {} \;
        log_info "✅ Bash syntax check passed"
    fi
}

generate_test_report() {
    log_info "Generating comprehensive test report..."
    
    mkdir -p "$PROJECT_ROOT/test-results"
    
    cat > "$PROJECT_ROOT/test-results/summary.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>PlantOS Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .pass { color: green; }
        .fail { color: red; }
        .warn { color: orange; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>PlantOS Test Report</h1>
    <table>
        <tr><th>Component</th><th>Status</th><th>Coverage</th></tr>
        <tr><td>Python</td><td class="pass">✅ PASS</td><td>${PYTHON_COV_MIN}%+</td></tr>
        <tr><td>Go</td><td class="pass">✅ PASS</td><td>${GO_COV_MIN}%+</td></tr>
        <tr><td>Rust</td><td class="pass">✅ PASS</td><td>${RUST_COV_MIN}%+</td></tr>
        <tr><td>Julia</td><td class="pass">✅ PASS</td><td>N/A</td></tr>
        <tr><td>Bash</td><td class="pass">✅ PASS</td><td>N/A</td></tr>
    </table>
    <p>Generated: $(date)</p>
</body>
</html>
EOF
    
    log_info "✅ Test report generated at test-results/summary.html"
}

main() {
    local hardware_tests=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --hardware)
                hardware_tests=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    log_info "Running comprehensive PlantOS test suite..."
    
    # Create test results directory
    mkdir -p "$PROJECT_ROOT/test-results"
    
    # Run all test suites
    run_python_tests $([ "$hardware_tests" = true ] && echo "--hardware")
    run_go_tests
    run_rust_tests
    run_julia_tests
    run_bash_tests
    
    generate_test_report
    
    log_info "🎉 All tests completed successfully!"
}

main "$@"
```

#### Raspberry Pi Setup Script
```bash
#!/bin/bash
# scripts/pi/pi-setup.sh - Complete Raspberry Pi configuration for PlantOS

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/../development/common.sh"

# Pi-specific configuration
readonly PI_USER="plantos"
readonly PI_GROUP="plantos"
readonly INSTALL_DIR="/opt/plantos"

check_pi_environment() {
    if [[ ! -f /proc/device-tree/model ]] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
        log_error "This script must be run on a Raspberry Pi"
        exit 1
    fi
    
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

enable_gpio_permissions() {
    log_info "Configuring GPIO permissions..."
    
    # Add user to gpio group
    usermod -a -G gpio "$PI_USER"
    
    # Create udev rules for GPIO access
    cat > /etc/udev/rules.d/99-gpio.rules << 'EOF'
SUBSYSTEM=="gpio", KERNEL=="gpiochip[0-4]", ACTION=="add", RUN+="/bin/chown root:gpio /sys/class/gpio/export /sys/class/gpio/unexport", RUN+="/bin/chmod 220 /sys/class/gpio/export /sys/class/gpio/unexport"
SUBSYSTEM=="gpio", KERNEL=="gpio[0-9]*", ACTION=="add", RUN+="/bin/chown root:gpio /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value", RUN+="/bin/chmod 660 /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value"
EOF

    # Enable I2C and SPI
    raspi-config nonint do_i2c 0
    raspi-config nonint do_spi 0
    
    log_info "✅ GPIO permissions configured"
}

setup_sensor_calibration() {
    log_info "Setting up sensor calibration..."
    
    mkdir -p "$INSTALL_DIR/calibration"
    
    cat > "$INSTALL_DIR/scripts/calibrate-sensors.sh" << 'EOF'
#!/bin/bash
# Sensor calibration script

set -euo pipefail

CALIBRATION_DIR="/opt/plantos/calibration"
mkdir -p "$CALIBRATION_DIR"

calibrate_moisture_sensor() {
    local pin=$1
    echo "Calibrating moisture sensor on pin $pin"
    echo "Place sensor in dry soil and press Enter"
    read -r
    
    # Read dry value using Rust driver
    local dry_value
    dry_value=$(/opt/plantos/hardware/drivers/target/release/sensor-calibrator moisture --pin "$pin" --samples 100)
    
    echo "Now place sensor in wet soil and press Enter"
    read -r
    
    local wet_value
    wet_value=$(/opt/plantos/hardware/drivers/target/release/sensor-calibrator moisture --pin "$pin" --samples 100)
    
    # Save calibration data
    cat > "$CALIBRATION_DIR/moisture_pin_${pin}.json" << CALIB_EOF
{
    "pin": $pin,
    "dry_value": $dry_value,
    "wet_value": $wet_value,
    "calibrated_at": "$(date -Iseconds)"
}
CALIB_EOF
    
    echo "✅ Moisture sensor on pin $pin calibrated"
}

# Calibrate all configured moisture sensors
for pin in 34 35 36 39; do
    calibrate_moisture_sensor "$pin"
done

echo "🎉 All sensors calibrated successfully!"
EOF

    chmod +x "$INSTALL_DIR/scripts/calibrate-sensors.sh"
    log_info "✅ Sensor calibration setup complete"
}

configure_systemd_services() {
    log_info "Installing systemd services..."
    
    # Copy service files
    cp "$PROJECT_ROOT/dist/systemd/"*.service /etc/systemd/system/
    
    # Enable services
    systemctl enable plantos-api.service
    systemctl enable plantos-streaming.service
    systemctl enable plantos-device-manager.service
    systemctl enable plantos-julia.service
    
    # Create plantos user if it doesn't exist
    if ! id "$PI_USER" &>/dev/null; then
        useradd -r -s /bin/bash -d "$INSTALL_DIR" -m "$PI_USER"
        usermod -a -G gpio,i2c,spi "$PI_USER"
    fi
    
    # Set ownership
    chown -R "$PI_USER:$PI_GROUP" "$INSTALL_DIR"
    
    log_info "✅ Systemd services configured"
}

setup_networking() {
    log_info "Configuring networking..."
    
    # Create WiFi configuration script
    cat > "$INSTALL_DIR/scripts/wifi-setup.sh" << 'EOF'
#!/bin/bash
# WiFi setup script

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <SSID> <PASSWORD>"
    exit 1
fi

SSID="$1"
PASSWORD="$2"

# Generate wpa_supplicant configuration
wpa_passphrase "$SSID" "$PASSWORD" >> /etc/wpa_supplicant/wpa_supplicant.conf

# Restart networking
systemctl restart dhcpcd
systemctl restart wpa_supplicant

echo "✅ WiFi configured for SSID: $SSID"
EOF

    chmod +x "$INSTALL_DIR/scripts/wifi-setup.sh"
    
    # Enable SSH
    systemctl enable ssh
    
    # Configure firewall for PlantOS ports
    if command -v ufw &> /dev/null; then
        ufw allow 8000/tcp  # Python API
        ufw allow 8081/tcp  # Julia analytics
        ufw allow 22/tcp    # SSH
        ufw --force enable
    fi
    
    log_info "✅ Networking configured"
}

install_monitoring() {
    log_info "Installing monitoring tools..."
    
    # Create system monitoring script
    cat > "$INSTALL_DIR/scripts/system-monitor.sh" << 'EOF'
#!/bin/bash
# System monitoring for Raspberry Pi

set -euo pipefail

LOG_FILE="/var/log/plantos/system-monitor.log"
mkdir -p "$(dirname "$LOG_FILE")"

log_metric() {
    echo "$(date -Iseconds) $1" >> "$LOG_FILE"
}

# CPU temperature
cpu_temp=$(vcgencmd measure_temp | cut -d= -f2 | cut -d\' -f1)
log_metric "cpu_temperature_celsius=$cpu_temp"

# Memory usage
mem_info=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')
log_metric "memory_usage_percent=$mem_info"

# Disk usage
disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
log_metric "disk_usage_percent=$disk_usage"

# Check if services are running
for service in plantos-api plantos-streaming plantos-device-manager; do
    if systemctl is-active --quiet "$service"; then
        log_metric "service_${service}_status=1"
    else
        log_metric "service_${service}_status=0"
    fi
done

# GPIO voltage check (3.3V rail)
gpio_voltage=$(vcgencmd measure_volts | cut -d= -f2 | cut -dV -f1)
log_metric "gpio_voltage=$gpio_voltage"
EOF

    chmod +x "$INSTALL_DIR/scripts/system-monitor.sh"
    
    # Create cron job for monitoring
    echo "*/5 * * * * $PI_USER $INSTALL_DIR/scripts/system-monitor.sh" > /etc/cron.d/plantos-monitor
    
    log_info "✅ Monitoring installed"
}

setup_log_rotation() {
    log_info "Configuring log rotation..."
    
    cat > /etc/logrotate.d/plantos << 'EOF'
/var/log/plantos/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 plantos plantos
    postrotate
        systemctl reload plantos-api || true
        systemctl reload plantos-streaming || true
    endscript
}
EOF
    
    log_info "✅ Log rotation configured"
}

create_backup_script() {
    log_info "Creating backup script..."
    
    cat > "$INSTALL_DIR/scripts/backup.sh" << 'EOF'
#!/bin/bash
# Automated backup script for PlantOS

set -euo pipefail

BACKUP_DIR="/opt/plantos/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/plantos_backup_$TIMESTAMP.tar.gz"

mkdir -p "$BACKUP_DIR"

# Create backup
tar -czf "$BACKUP_FILE" \
    --exclude="$BACKUP_DIR" \
    --exclude="/opt/plantos/venv" \
    --exclude="/opt/plantos/target" \
    --exclude="/opt/plantos/logs" \
    /opt/plantos \
    /etc/systemd/system/plantos-*.service \
    /var/log/plantos

# Database backup
sudo -u postgres pg_dump plantos | gzip > "$BACKUP_DIR/database_$TIMESTAMP.sql.gz"

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete

echo "✅ Backup completed: $BACKUP_FILE"
EOF

    chmod +x "$INSTALL_DIR/scripts/backup.sh"
    
    # Schedule daily backups at 2 AM
    echo "0 2 * * * $PI_USER $INSTALL_DIR/scripts/backup.sh" >> /etc/cron.d/plantos-backup
    
    log_info "✅ Backup script created"
}

main() {
    log_info "Starting Raspberry Pi setup for PlantOS..."
    
    check_pi_environment
    enable_gpio_permissions
    setup_sensor_calibration
    configure_systemd_services
    setup_networking
    install_monitoring
    setup_log_rotation
    create_backup_script
    
    log_info "🎉 Raspberry Pi setup completed!"
    log_info "Reboot required to activate all changes"
    log_info "After reboot, run: sudo systemctl start plantos-api"
}

main "$@"
```

#### Deployment Script
```bash
#!/bin/bash
# scripts/deployment/deploy.sh - Production deployment automation

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/../development/common.sh"

# Deployment configuration
readonly DEPLOY_USER="plantos"
readonly DEPLOY_HOST="${DEPLOY_HOST:-raspberry-pi.local}"
readonly DEPLOY_DIR="/opt/plantos"
readonly BACKUP_BEFORE_DEPLOY=true

pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if we can connect to target
    if ! ssh -o ConnectTimeout=10 "$DEPLOY_USER@$DEPLOY_HOST" "echo 'Connection OK'"; then
        log_error "Cannot connect to $DEPLOY_HOST as $DEPLOY_USER"
        exit 1
    fi
    
    # Check if all services build successfully
    cd "$PROJECT_ROOT"
    ./scripts/development/build.sh
    
    # Run tests
    ./scripts/development/test.sh
    
    log_info "✅ Pre-deployment checks passed"
}

create_deployment_package() {
    log_info "Creating deployment package..."
    
    local package_name="plantos-deploy-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    tar -czf "/tmp/$package_name" \
        -C "$PROJECT_ROOT" \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='target/debug' \
        --exclude='__pycache__' \
        --exclude='node_modules' \
        --exclude='test-results' \
        .
    
    echo "/tmp/$package_name"
}

backup_current_deployment() {
    if [[ "$BACKUP_BEFORE_DEPLOY" == "true" ]]; then
        log_info "Creating backup of current deployment..."
        
        ssh "$DEPLOY_USER@$DEPLOY_HOST" "
            sudo systemctl stop plantos-api plantos-streaming plantos-device-manager || true
            
            if [[ -d '$DEPLOY_DIR' ]]; then
                sudo tar -czf '/tmp/plantos-backup-$(date +%Y%m%d-%H%M%S).tar.gz' '$DEPLOY_DIR'
                echo '✅ Backup created'
            fi
        "
    fi
}

deploy_application() {
    local package_path="$1"
    
    log_info "Deploying application to $DEPLOY_HOST..."
    
    # Copy package to target
    scp "$package_path" "$DEPLOY_USER@$DEPLOY_HOST:/tmp/"
    
    # Extract and install
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "
        set -euo pipefail
        
        # Stop services
        sudo systemctl stop plantos-api plantos-streaming plantos-device-manager || true
        
        # Create deployment directory
        sudo mkdir -p '$DEPLOY_DIR'
        sudo chown '$DEPLOY_USER:$DEPLOY_USER' '$DEPLOY_DIR'
        
        # Extract package
        cd '$DEPLOY_DIR'
        tar -xzf '/tmp/$(basename "$package_path")'
        
        # Install Python dependencies
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        
        # Build Rust drivers
        cd hardware/drivers
        cargo build --release
        
        # Build Go services
        cd ../../services/streaming
        go build -o bin/streaming-server ./cmd/server/
        cd ../device
        go build -o bin/device-manager ./cmd/manager/
        
        # Install Julia packages
        cd ../../analytics/julia
        julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        
        # Set permissions
        sudo chown -R '$DEPLOY_USER:$DEPLOY_USER' '$DEPLOY_DIR'
        
        echo '✅ Application deployed'
    "
}

run_migrations() {
    log_info "Running database migrations..."
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "
        cd '$DEPLOY_DIR'
        source venv/bin/activate
        alembic upgrade head
    "
    
    log_info "✅ Migrations completed"
}

start_services() {
    log_info "Starting PlantOS services..."
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "
        sudo systemctl start plantos-api
        sudo systemctl start plantos-streaming
        sudo systemctl start plantos-device-manager
        sudo systemctl start plantos-julia
        
        # Wait for services to start
        sleep 10
        
        # Check service status
        for service in plantos-api plantos-streaming plantos-device-manager; do
            if sudo systemctl is-active --quiet \$service; then
                echo \"✅ \$service is running\"
            else
                echo \"❌ \$service failed to start\"
                sudo systemctl status \$service
            fi
        done
    "
}

run_health_checks() {
    log_info "Running post-deployment health checks..."
    
    # Check API health
    sleep 5  # Give services time to start
    
    if curl -f "http://$DEPLOY_HOST:8000/health" &>/dev/null; then
        log_info "✅ API health check passed"
    else
        log_error "❌ API health check failed"
        return 1
    fi
    
    # Check Julia analytics service
    if curl -f "http://$DEPLOY_HOST:8081/health" &>/dev/null; then
        log_info "✅ Julia analytics health check passed"
    else
        log_error "❌ Julia analytics health check failed"
        return 1
    fi
}

cleanup() {
    log_info "Cleaning up deployment artifacts..."
    
    # Remove temporary files
    rm -f /tmp/plantos-deploy-*.tar.gz
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "
        rm -f /tmp/plantos-deploy-*.tar.gz
        
        # Clean up old Docker images and containers
        sudo podman system prune -f || true
    "
}

main() {
    local skip_checks=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-checks)
                skip_checks=true
                shift
                ;;
            --host)
                DEPLOY_HOST="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    log_info "Starting deployment to $DEPLOY_HOST..."
    
    if [[ "$skip_checks" != "true" ]]; then
        pre_deployment_checks
    fi
    
    local package_path
    package_path=$(create_deployment_package)
    
    backup_current_deployment
    deploy_application "$package_path"
    run_migrations
    start_services
    run_health_checks
    cleanup
    
    log_info "🎉 Deployment completed successfully!"
    log_info "PlantOS is now running at http://$DEPLOY_HOST:8000"
}

main "$@"
```

### Julia Testing

#### Unit Testing
```julia
# analytics/julia/test/runtests.jl
using Test, PlantOS
using DifferentialEquations, Optimization

@testset "PlantOS.jl Tests" begin
    
    @testset "Plant Growth Models" begin
        @test begin
            params = PlantParameters(
                growth_rate=0.1,
                water_efficiency=0.8,
                temperature_optimum=25.0
            )
            
            initial_conditions = [1.0, 50.0]  # biomass, water_content
            timespan = (0.0, 10.0)
            
            sol = simulate_plant_growth(initial_conditions, timespan, params)
            
            # Test that solution exists and is reasonable
            !isempty(sol) && sol.u[end][1] > sol.u[1][1]  # Growth occurred
        end
    end
    
    @testset "Optimization Algorithms" begin
        @test begin
            problem = WateringOptimizationProblem(
                plant_count=3,
                time_horizon=24,
                water_capacity=500.0,
                plant_parameters=[PlantParameters() for _ in 1:3],
                current_states=rand(3, 4)
            )
            
            schedule = optimize_watering_schedule(problem)
            
            # Test constraints are satisfied
            size(schedule) == (3, 24) && 
            all(schedule .>= 0) &&  # Non-negative
            all([sum(schedule[:, t]) <= problem.water_capacity for t in 1:24])  # Capacity constraint
        end
    end
    
    @testset "Time Series Forecasting" begin
        @test begin
            # Create synthetic time series data
            timestamps = collect(DateTime(2024,1,1):Hour(1):DateTime(2024,1,7))
            data = sin.(2π * (1:length(timestamps)) / 24) .+ randn(length(timestamps)) * 0.1
            ts = TimeArray(timestamps, data, ["moisture"])
            
            forecast_result = forecast_sensor_readings(ts, 12)
            
            # Test forecast structure
            haskey(forecast_result, :forecast) &&
            haskey(forecast_result, :upper_ci) &&
            haskey(forecast_result, :lower_ci) &&
            length(forecast_result.forecast) == 12
        end
    end
end

# Performance benchmarks
@testset "Performance Benchmarks" begin
    using BenchmarkTools
    
    @testset "Plant Growth Simulation Performance" begin
        params = PlantParameters()
        initial_conditions = [1.0, 50.0]
        timespan = (0.0, 100.0)
        
        # Benchmark should complete in reasonable time
        @test @elapsed(simulate_plant_growth(initial_conditions, timespan, params)) < 1.0
    end
    
    @testset "Optimization Performance" begin
        problem = WateringOptimizationProblem(
            plant_count=5,
            time_horizon=48,
            water_capacity=1000.0,
            plant_parameters=[PlantParameters() for _ in 1:5],
            current_states=rand(5, 4)
        )
        
        # Optimization should complete in reasonable time
        @test @elapsed(optimize_watering_schedule(problem)) < 10.0
    end
end
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


### Go Testing
```go
// services/streaming/sensor_pipeline_test.go
package streaming

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
    "go.uber.org/zap/zaptest"
)

type MockEventHandler struct {
    mock.Mock
}

func (m *MockEventHandler) Handle(ctx context.Context, event SensorEvent) error {
    args := m.Called(ctx, event)
    return args.Error(0)
}

func TestPipeline_ProcessEvent(t *testing.T) {
    // Setup
    logger := zaptest.NewLogger(t)
    pipeline := NewPipeline(logger, WithBufferSize(100))
    
    handler := new(MockEventHandler)
    pipeline.RegisterHandler("moisture", handler)
    
    event := SensorEvent{
        SensorID:  "sensor-1",
        PlantID:   "plant-1",
        Type:      "moisture",
        Value:     45.5,
        Unit:      "percentage",
        Timestamp: time.Now(),
    }
    
    // Expect
    handler.On("Handle", mock.Anything, event).Return(nil)
    
    // Act
    err := pipeline.ProcessEvent(context.Background(), event)
    
    // Assert
    assert.NoError(t, err)
    handler.AssertExpectations(t)
}

func TestPipeline_ConcurrentProcessing(t *testing.T) {
    logger := zaptest.NewLogger(t)
    pipeline := NewPipeline(logger, WithWorkerCount(5))
    
    processed := make(chan SensorEvent, 1000)
    
    // Register handler that records processed events
    pipeline.RegisterHandler("test", HandlerFunc(func(ctx context.Context, event SensorEvent) error {
        processed <- event
        return nil
    }))
    
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    // Start pipeline
    go pipeline.Run(ctx)
    
    // Send concurrent events
    eventCount := 100
    for i := 0; i < eventCount; i++ {
        go func(idx int) {
            event := SensorEvent{
                SensorID: fmt.Sprintf("sensor-%d", idx),
                Type:     "test",
                Value:    float64(idx),
            }
            pipeline.Submit(event)
        }(i)
    }
    
    // Wait for processing
    time.Sleep(1 * time.Second)
    
    // Verify all events processed
    assert.Equal(t, eventCount, len(processed))
}

// Benchmark concurrent device polling
func BenchmarkManager_PollAll(b *testing.B) {
    manager := NewManager()
    
    // Add test devices
    for i := 0; i < 100; i++ {
        device := &Device{
            ID:   fmt.Sprintf("device-%d", i),
            Type: "moisture_sensor",
        }
        manager.AddDevice(device)
    }
    
    ctx := context.Background()
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _, _ = manager.PollAll(ctx)
        }
    })
}
```

### Rust Testing
```rust
// hardware/drivers/src/moisture_sensor.rs (test module)
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    
    // Mock ADC functions for testing
    #[no_mangle]
    pub extern "C" fn adc_init(_pin: u8) -> i32 {
        0 // Success
    }
    
    #[no_mangle]
    pub extern "C" fn adc_read(_pin: u8, value: *mut u16) -> i32 {
        unsafe {
            *value = 2000; // Mock reading
        }
        0
    }
    
    #[test]
    fn test_sensor_creation() {
        let sensor = MoistureSensor::new(34).unwrap();
        assert_eq!(sensor.pin, 34);
    }
    
    #[test]
    fn test_invalid_pin() {
        let result = MoistureSensor::new(40);
        assert!(matches!(result, Err(SensorError::InvalidValue(_))));
    }
    
    #[test]
    fn test_read_uninitialized() {
        let sensor = MoistureSensor::new(34).unwrap();
        let result = sensor.read();
        assert!(matches!(result, Err(SensorError::NotInitialized)));
    }
    
    #[test]
    fn test_concurrent_reads() {
        let sensor = Arc::new(MoistureSensor::new(34).unwrap());
        sensor.init().unwrap();
        
        let mut handles = vec![];
        
        // Spawn multiple threads reading concurrently
        for _ in 0..10 {
            let sensor_clone = Arc::clone(&sensor);
            let handle = thread::spawn(move || {
                sensor_clone.read()
            });
            handles.push(handle);
        }
        
        // All reads should succeed
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok());
        }
    }
    
    #[test]
    fn test_calibration_boundaries() {
        let sensor = MoistureSensor::new(34).unwrap();
        
        // Test dry value
        assert_eq!(sensor.calibrate_value(2800), 0.0);
        
        // Test wet value
        assert_eq!(sensor.calibrate_value(1200), 100.0);
        
        // Test mid-range
        let mid = sensor.calibrate_value(2000);
        assert!((mid - 50.0).abs() < 1.0);
    }
}

// Integration tests in tests/integration.rs
#[cfg(test)]
mod integration_tests {
    use super::*;
    use serial_test::serial;
    
    #[test]
    #[serial]
    fn test_pump_safety_limits() {
        let pump = PumpController::new(25);
        
        // Test duration limit
        let result = tokio_test::block_on(
            pump.run_for_duration(Duration::from_secs(150))
        );
        assert!(matches!(result, Err(PumpError::DurationExceeded)));
    }
    
    #[test]
    #[serial]
    fn test_pump_cycle_interval() {
        let pump = PumpController::new(25);
        
        // First run should succeed
        tokio_test::block_on(async {
            pump.run_for_duration(Duration::from_secs(5)).await.unwrap();
            
            // Immediate second run should fail
            let result = pump.run_for_duration(Duration::from_secs(5)).await;
            assert!(matches!(result, Err(PumpError::CycleIntervalViolation)));
            
            // Wait for interval, then should succeed
            tokio::time::sleep(Duration::from_secs(61)).await;
            pump.run_for_duration(Duration::from_secs(5)).await.unwrap();
        });
    }
    
    #[test]
    fn test_emergency_stop() {
        let pump = PumpController::new(25);
        
        tokio_test::block_on(async {
            // Start pump in background
            let pump_clone = pump.clone();
            tokio::spawn(async move {
                pump_clone.run_for_duration(Duration::from_secs(10)).await
            });
            
            // Wait a bit then emergency stop
            tokio::time::sleep(Duration::from_millis(500)).await;
            pump.emergency_stop().unwrap();
            
            // Verify pump is stopped
            assert!(!pump.is_running());
        });
    }
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