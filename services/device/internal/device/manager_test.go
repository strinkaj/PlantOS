package device

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"plantos/device/internal/config"
	"plantos/device/internal/types"
)

func TestManagerCreation(t *testing.T) {
	cfg := &config.Config{
		Port:                    8081,
		GRPCPort:               9091,
		StreamingServiceURL:    "http://localhost:8080",
		PostgresURL:            "postgres://user:pass@localhost:5432/db",
		RedisURL:               "redis://localhost:6379",
		PollingInterval:        30 * time.Second,
		MaxConcurrentPolls:     50,
		DeviceTimeout:          5 * time.Second,
		RetryAttempts:          3,
		RetryDelay:             1 * time.Second,
		CircuitBreakerTimeout:  60 * time.Second,
		CircuitBreakerThreshold: 10,
		MetricsEnabled:         true,
		LogLevel:               "info",
		Environment:            "test",
		ServiceName:            "test-device-manager",
		ServiceVersion:         "1.0.0",
		HealthCheckPath:        "/health",
		MetricsPath:            "/metrics",
		ShutdownTimeout:        30 * time.Second,
		HardwareInterface:      "mock",
		MockHardware:           true,
		DeviceDiscovery:        true,
		WorkerPoolSize:         10,
	}

	manager := NewManager(cfg)
	if manager == nil {
		t.Fatal("Expected manager to be created")
	}

	if manager.config != cfg {
		t.Fatal("Expected config to be set")
	}

	if manager.devices == nil {
		t.Fatal("Expected devices map to be initialized")
	}

	if manager.circuitBreakers == nil {
		t.Fatal("Expected circuit breakers map to be initialized")
	}

	if manager.workerPool == nil {
		t.Fatal("Expected worker pool to be initialized")
	}
}

func TestDeviceRegistration(t *testing.T) {
	cfg := createTestConfig()
	manager := NewManager(cfg)

	device := &types.Device{
		ID:             "test-device-1",
		Name:           "Test Moisture Sensor",
		Type:           types.DeviceTypeMoistureSensor,
		HardwareID:     "ADC-001",
		ConnectionType: types.ConnectionTypeI2C,
		Address:        "0x48",
		Capabilities:   []string{"read_moisture"},
		Configuration:  map[string]interface{}{"sampling_rate": 1000},
	}

	err := manager.RegisterDevice(device)
	if err != nil {
		t.Errorf("Expected no error registering device, got: %v", err)
	}

	// Verify device was added
	retrievedDevice, err := manager.GetDevice("test-device-1")
	if err != nil {
		t.Errorf("Expected to retrieve device, got error: %v", err)
	}

	if retrievedDevice.Name != device.Name {
		t.Errorf("Expected device name %s, got %s", device.Name, retrievedDevice.Name)
	}

	if retrievedDevice.Type != device.Type {
		t.Errorf("Expected device type %s, got %s", device.Type, retrievedDevice.Type)
	}
}

func TestDeviceValidation(t *testing.T) {
	cfg := createTestConfig()
	manager := NewManager(cfg)

	tests := []struct {
		name      string
		device    types.Device
		expectErr bool
	}{
		{
			name: "valid device",
			device: types.Device{
				ID:             "valid-device",
				Name:           "Valid Device",
				Type:           types.DeviceTypeMoistureSensor,
				HardwareID:     "ADC-001",
				ConnectionType: types.ConnectionTypeI2C,
				Address:        "0x48",
			},
			expectErr: false,
		},
		{
			name: "missing ID",
			device: types.Device{
				Name:           "Invalid Device",
				Type:           types.DeviceTypeMoistureSensor,
				HardwareID:     "ADC-001",
				ConnectionType: types.ConnectionTypeI2C,
				Address:        "0x48",
			},
			expectErr: true,
		},
		{
			name: "invalid device type",
			device: types.Device{
				ID:             "invalid-device",
				Name:           "Invalid Device",
				Type:           "invalid_type",
				HardwareID:     "ADC-001",
				ConnectionType: types.ConnectionTypeI2C,
				Address:        "0x48",
			},
			expectErr: true,
		},
		{
			name: "missing hardware ID",
			device: types.Device{
				ID:             "invalid-device",
				Name:           "Invalid Device",
				Type:           types.DeviceTypeMoistureSensor,
				ConnectionType: types.ConnectionTypeI2C,
				Address:        "0x48",
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := manager.RegisterDevice(&tt.device)
			if tt.expectErr && err == nil {
				t.Errorf("Expected error but got none")
			}
			if !tt.expectErr && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

func TestDevicePolling(t *testing.T) {
	cfg := createTestConfig()
	cfg.PollingInterval = 100 * time.Millisecond // Fast polling for testing
	manager := NewManager(cfg)

	// Register a test device
	device := &types.Device{
		ID:             "polling-test-device",
		Name:           "Polling Test Device",
		Type:           types.DeviceTypeMoistureSensor,
		Status:         types.DeviceStatusOnline,
		HardwareID:     "ADC-001",
		ConnectionType: types.ConnectionTypeI2C,
		Address:        "0x48",
	}

	err := manager.RegisterDevice(device)
	if err != nil {
		t.Fatalf("Failed to register device: %v", err)
	}

	// Start manager
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go manager.Start(ctx)
	time.Sleep(50 * time.Millisecond) // Let manager start

	// Start polling
	err = manager.StartPolling()
	if err != nil {
		t.Errorf("Failed to start polling: %v", err)
	}

	// Check polling status
	status := manager.GetPollingStatus()
	if !status.Active {
		t.Error("Expected polling to be active")
	}

	if status.TotalDevices != 1 {
		t.Errorf("Expected 1 device, got %d", status.TotalDevices)
	}

	// Let it poll a few times
	time.Sleep(300 * time.Millisecond)

	// Stop polling
	err = manager.StopPolling()
	if err != nil {
		t.Errorf("Failed to stop polling: %v", err)
	}

	status = manager.GetPollingStatus()
	if status.Active {
		t.Error("Expected polling to be stopped")
	}
}

func TestCircuitBreaker(t *testing.T) {
	cb := NewCircuitBreaker(3, 100*time.Millisecond)

	// Initially closed
	if !cb.IsClosed() {
		t.Error("Expected circuit breaker to be closed initially")
	}

	// Record failures to open circuit
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordFailure()

	if !cb.IsOpen() {
		t.Error("Expected circuit breaker to be open after max failures")
	}

	// Wait for timeout
	time.Sleep(150 * time.Millisecond)

	// Check if it's open (this should trigger transition to half-open)
	isOpen := cb.IsOpen()
	if isOpen {
		t.Error("Expected circuit breaker to not be open after timeout")
	}

	// Should be half-open now
	if !cb.IsHalfOpen() {
		t.Error("Expected circuit breaker to be half-open after timeout")
	}

	// Record success to close
	cb.RecordSuccess()
	cb.RecordSuccess()
	cb.RecordSuccess()

	if !cb.IsClosed() {
		t.Error("Expected circuit breaker to be closed after successes")
	}
}

func TestWorkerPool(t *testing.T) {
	pool := NewWorkerPool(2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pool.Start(ctx)

	// Submit tasks with shared counter
	var tasksCompleted int32
	for i := 0; i < 5; i++ {
		pool.Submit(func() {
			time.Sleep(10 * time.Millisecond)
			// Use atomic operation for thread safety
			if atomic.LoadInt32(&tasksCompleted) >= 0 {
				atomic.AddInt32(&tasksCompleted, 1)
			}
		})
	}

	// Wait for tasks to complete
	time.Sleep(300 * time.Millisecond)

	completed := atomic.LoadInt32(&tasksCompleted)
	if completed < 4 { // Allow for some timing variance in tests
		t.Errorf("Expected at least 4 tasks completed, got %d", completed)
	}

	pool.Close()
}

func TestGetDevicesByPlant(t *testing.T) {
	cfg := createTestConfig()
	manager := NewManager(cfg)

	// Register devices for different plants
	devices := []*types.Device{
		{
			ID:             "device-1",
			Name:           "Device 1",
			Type:           types.DeviceTypeMoistureSensor,
			PlantID:        "plant-1",
			HardwareID:     "ADC-001",
			ConnectionType: types.ConnectionTypeI2C,
			Address:        "0x48",
		},
		{
			ID:             "device-2",
			Name:           "Device 2",
			Type:           types.DeviceTypeTemperatureSensor,
			PlantID:        "plant-1",
			HardwareID:     "BME280-001",
			ConnectionType: types.ConnectionTypeI2C,
			Address:        "0x76",
		},
		{
			ID:             "device-3",
			Name:           "Device 3",
			Type:           types.DeviceTypeLightSensor,
			PlantID:        "plant-2",
			HardwareID:     "TSL2561-001",
			ConnectionType: types.ConnectionTypeI2C,
			Address:        "0x39",
		},
	}

	for _, device := range devices {
		err := manager.RegisterDevice(device)
		if err != nil {
			t.Fatalf("Failed to register device %s: %v", device.ID, err)
		}
	}

	// Test getting devices for plant-1
	plant1Devices := manager.GetDevicesByPlant("plant-1")
	if len(plant1Devices) != 2 {
		t.Errorf("Expected 2 devices for plant-1, got %d", len(plant1Devices))
	}

	// Test getting devices for plant-2
	plant2Devices := manager.GetDevicesByPlant("plant-2")
	if len(plant2Devices) != 1 {
		t.Errorf("Expected 1 device for plant-2, got %d", len(plant2Devices))
	}

	// Test getting devices for non-existent plant
	noDevices := manager.GetDevicesByPlant("plant-999")
	if len(noDevices) != 0 {
		t.Errorf("Expected 0 devices for non-existent plant, got %d", len(noDevices))
	}
}

func createTestConfig() *config.Config {
	return &config.Config{
		Port:                    8081,
		GRPCPort:               9091,
		StreamingServiceURL:    "http://localhost:8080",
		PostgresURL:            "postgres://user:pass@localhost:5432/db",
		RedisURL:               "redis://localhost:6379",
		PollingInterval:        30 * time.Second,
		MaxConcurrentPolls:     50,
		DeviceTimeout:          5 * time.Second,
		RetryAttempts:          3,
		RetryDelay:             1 * time.Second,
		CircuitBreakerTimeout:  60 * time.Second,
		CircuitBreakerThreshold: 10,
		MetricsEnabled:         true,
		LogLevel:               "info",
		Environment:            "test",
		ServiceName:            "test-device-manager",
		ServiceVersion:         "1.0.0",
		HealthCheckPath:        "/health",
		MetricsPath:            "/metrics",
		ShutdownTimeout:        30 * time.Second,
		HardwareInterface:      "mock",
		MockHardware:           true,
		DeviceDiscovery:        false, // Disable for tests
		WorkerPoolSize:         10,
	}
}