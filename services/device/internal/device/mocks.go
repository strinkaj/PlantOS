package device

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"plantos/device/internal/types"
)

// Mock implementations for development and testing

// mockDeviceClient is a mock implementation of DeviceClient
type mockDeviceClient struct{}

func (m *mockDeviceClient) ReadSensor(deviceID, sensorType string, timeout time.Duration) (float64, error) {
	log.Printf("MockDevice: Reading sensor %s on device %s", sensorType, deviceID)
	
	// Simulate occasional failures
	if rand.Float32() < 0.05 { // 5% failure rate
		return 0, fmt.Errorf("sensor read timeout")
	}
	
	// Generate realistic sensor values
	switch sensorType {
	case string(types.DeviceTypeMoistureSensor):
		return 30 + rand.Float64()*40, nil // 30-70%
	case string(types.DeviceTypeTemperatureSensor):
		return 18 + rand.Float64()*12, nil // 18-30°C
	case string(types.DeviceTypeHumiditySensor):
		return 40 + rand.Float64()*30, nil // 40-70%
	case string(types.DeviceTypeLightSensor):
		return rand.Float64() * 10000, nil // 0-10000 lux
	case string(types.DeviceTypePHSensor):
		return 6.0 + rand.Float64()*2.0, nil // 6.0-8.0 pH
	default:
		return rand.Float64() * 100, nil
	}
}

func (m *mockDeviceClient) ExecuteCommand(deviceID, command string, params map[string]interface{}, timeout time.Duration) error {
	log.Printf("MockDevice: Executing command %s on device %s with params %v", command, deviceID, params)
	
	// Simulate command execution delay
	time.Sleep(time.Millisecond * 100)
	
	// Simulate occasional failures
	if rand.Float32() < 0.03 { // 3% failure rate
		return fmt.Errorf("command execution failed")
	}
	
	return nil
}

func (m *mockDeviceClient) DiscoverDevices() ([]types.Device, error) {
	log.Println("MockDevice: Discovering devices")
	
	// Return some mock devices
	devices := []types.Device{
		{
			ID:             "moisture-001",
			Name:           "Soil Moisture Sensor 1",
			Type:           types.DeviceTypeMoistureSensor,
			Status:         types.DeviceStatusOffline,
			HardwareID:     "ADC-001",
			ConnectionType: types.ConnectionTypeI2C,
			Address:        "0x48",
			Capabilities:   []string{"read_moisture", "calibrate"},
			Configuration:  map[string]interface{}{"sampling_rate": 1000, "resolution": 12},
			CreatedAt:      time.Now().UTC(),
			UpdatedAt:      time.Now().UTC(),
		},
		{
			ID:             "temp-001",
			Name:           "Temperature Sensor 1",
			Type:           types.DeviceTypeTemperatureSensor,
			Status:         types.DeviceStatusOffline,
			HardwareID:     "BME280-001",
			ConnectionType: types.ConnectionTypeI2C,
			Address:        "0x76",
			Capabilities:   []string{"read_temperature", "read_humidity", "read_pressure"},
			Configuration:  map[string]interface{}{"precision": "high", "filter": "enabled"},
			CreatedAt:      time.Now().UTC(),
			UpdatedAt:      time.Now().UTC(),
		},
		{
			ID:             "light-001",
			Name:           "Light Sensor 1",
			Type:           types.DeviceTypeLightSensor,
			Status:         types.DeviceStatusOffline,
			HardwareID:     "TSL2561-001",
			ConnectionType: types.ConnectionTypeI2C,
			Address:        "0x39",
			Capabilities:   []string{"read_lux", "read_infrared", "read_visible"},
			Configuration:  map[string]interface{}{"gain": "auto", "integration_time": "402ms"},
			CreatedAt:      time.Now().UTC(),
			UpdatedAt:      time.Now().UTC(),
		},
	}
	
	return devices, nil
}

func (m *mockDeviceClient) ValidateDevice(device *types.Device) error {
	log.Printf("MockDevice: Validating device %s", device.ID)
	
	// Basic validation - check if hardware ID is accessible
	if device.HardwareID == "" {
		return fmt.Errorf("hardware ID is required")
	}
	
	// Simulate hardware validation
	time.Sleep(time.Millisecond * 50)
	
	return nil
}

func (m *mockDeviceClient) Close() error {
	log.Println("MockDevice: Closing device client")
	return nil
}

// mockStreamingClient is a mock implementation of StreamingClient
type mockStreamingClient struct{}

func (m *mockStreamingClient) SendReading(reading types.DeviceReading) error {
	log.Printf("MockStreaming: Sending reading - Device: %s, Type: %s, Value: %f", 
		reading.DeviceID, reading.SensorType, reading.Value)
	
	// Simulate network call
	time.Sleep(time.Millisecond * 10)
	
	// Simulate occasional failures
	if rand.Float32() < 0.02 { // 2% failure rate
		return fmt.Errorf("streaming service unavailable")
	}
	
	return nil
}

func (m *mockStreamingClient) SendBatchReadings(readings []types.DeviceReading) error {
	log.Printf("MockStreaming: Sending batch of %d readings", len(readings))
	
	// Simulate network call
	time.Sleep(time.Millisecond * 50)
	
	// Simulate occasional failures
	if rand.Float32() < 0.01 { // 1% failure rate
		return fmt.Errorf("batch send failed")
	}
	
	return nil
}

func (m *mockStreamingClient) Close() error {
	log.Println("MockStreaming: Closing streaming client")
	return nil
}

// mockDatabaseClient is a mock implementation of DatabaseClient
type mockDatabaseClient struct {
	devices map[string]*types.Device
}

func (m *mockDatabaseClient) SaveDevice(device *types.Device) error {
	if m.devices == nil {
		m.devices = make(map[string]*types.Device)
	}
	
	deviceCopy := *device
	m.devices[device.ID] = &deviceCopy
	
	log.Printf("MockDB: Saved device %s", device.ID)
	return nil
}

func (m *mockDatabaseClient) LoadDevices() ([]types.Device, error) {
	log.Println("MockDB: Loading devices")
	
	if m.devices == nil {
		return []types.Device{}, nil
	}
	
	devices := make([]types.Device, 0, len(m.devices))
	for _, device := range m.devices {
		devices = append(devices, *device)
	}
	
	log.Printf("MockDB: Loaded %d devices", len(devices))
	return devices, nil
}

func (m *mockDatabaseClient) UpdateDeviceStatus(deviceID string, status types.DeviceStatus) error {
	if m.devices == nil {
		return fmt.Errorf("device not found: %s", deviceID)
	}
	
	device, exists := m.devices[deviceID]
	if !exists {
		return fmt.Errorf("device not found: %s", deviceID)
	}
	
	device.Status = status
	device.UpdatedAt = time.Now().UTC()
	
	log.Printf("MockDB: Updated device %s status to %s", deviceID, status)
	return nil
}

func (m *mockDatabaseClient) DeleteDevice(deviceID string) error {
	if m.devices == nil {
		return fmt.Errorf("device not found: %s", deviceID)
	}
	
	delete(m.devices, deviceID)
	log.Printf("MockDB: Deleted device %s", deviceID)
	return nil
}

func (m *mockDatabaseClient) Close() error {
	log.Println("MockDB: Closing database client")
	return nil
}

// mockMetrics is a mock implementation of Metrics
type mockMetrics struct{}

func (m *mockMetrics) IncrementCounter(name string, labels map[string]string) {
	log.Printf("MockMetrics: Incrementing counter %s with labels %v", name, labels)
}

func (m *mockMetrics) RecordHistogram(name string, value float64, labels map[string]string) {
	log.Printf("MockMetrics: Recording histogram %s = %f with labels %v", name, value, labels)
}

func (m *mockMetrics) RecordGauge(name string, value float64, labels map[string]string) {
	log.Printf("MockMetrics: Recording gauge %s = %f with labels %v", name, value, labels)
}

// mockLogger is a mock implementation of Logger
type mockLogger struct{}

func (m *mockLogger) Info(msg string, fields ...interface{}) {
	log.Printf("INFO: %s %v", msg, fields)
}

func (m *mockLogger) Error(msg string, fields ...interface{}) {
	log.Printf("ERROR: %s %v", msg, fields)
}

func (m *mockLogger) Debug(msg string, fields ...interface{}) {
	log.Printf("DEBUG: %s %v", msg, fields)
}

func (m *mockLogger) Warn(msg string, fields ...interface{}) {
	log.Printf("WARN: %s %v", msg, fields)
}