# PlantOS Development Guide

## Development Environment Setup

### Prerequisites (All Free for Personal Use)
- Python 3.12+ (PSF License)
- Go 1.21+ (BSD License)
- Rust 1.75+ with Cargo (MIT/Apache 2.0)
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


### Go Code Standards

#### Service Structure
```go
// services/streaming/sensor_pipeline.go
package streaming

import (
    "context"
    "fmt"
    "sync"
    "time"
    
    "github.com/prometheus/client_golang/prometheus"
    "go.uber.org/zap"
)

// SensorEvent represents a single sensor reading
type SensorEvent struct {
    SensorID  string    `json:"sensor_id"`
    PlantID   string    `json:"plant_id"`
    Type      string    `json:"type"`
    Value     float64   `json:"value"`
    Unit      string    `json:"unit"`
    Timestamp time.Time `json:"timestamp"`
}

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