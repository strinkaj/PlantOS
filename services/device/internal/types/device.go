package types

import (
	"fmt"
	"time"
)

// DeviceType represents the type of device
type DeviceType string

const (
	DeviceTypeMoistureSensor    DeviceType = "moisture_sensor"
	DeviceTypeTemperatureSensor DeviceType = "temperature_sensor"
	DeviceTypeHumiditySensor    DeviceType = "humidity_sensor"
	DeviceTypeLightSensor       DeviceType = "light_sensor"
	DeviceTypePHSensor          DeviceType = "ph_sensor"
	DeviceTypeCamera            DeviceType = "camera"
	DeviceTypeWaterPump         DeviceType = "water_pump"
	DeviceTypeFan               DeviceType = "fan"
	DeviceTypeLED               DeviceType = "led"
	DeviceTypeValve             DeviceType = "valve"
)

// DeviceStatus represents the current status of a device
type DeviceStatus string

const (
	DeviceStatusOnline    DeviceStatus = "online"
	DeviceStatusOffline   DeviceStatus = "offline"
	DeviceStatusError     DeviceStatus = "error"
	DeviceStatusMaintenance DeviceStatus = "maintenance"
	DeviceStatusCalibrating DeviceStatus = "calibrating"
)

// ConnectionType represents how the device is connected
type ConnectionType string

const (
	ConnectionTypeI2C    ConnectionType = "i2c"
	ConnectionTypeSPI    ConnectionType = "spi"
	ConnectionTypeGPIO   ConnectionType = "gpio"
	ConnectionTypeUSB    ConnectionType = "usb"
	ConnectionTypeSerial ConnectionType = "serial"
	ConnectionTypeNetwork ConnectionType = "network"
)

// Device represents a hardware device in the system
type Device struct {
	ID              string            `json:"id"`
	Name            string            `json:"name"`
	Type            DeviceType        `json:"type"`
	Status          DeviceStatus      `json:"status"`
	PlantID         string            `json:"plant_id,omitempty"`
	HardwareID      string            `json:"hardware_id"`
	ConnectionType  ConnectionType    `json:"connection_type"`
	Address         string            `json:"address"`
	Capabilities    []string          `json:"capabilities"`
	Configuration   map[string]interface{} `json:"configuration"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	LastSeen        time.Time         `json:"last_seen"`
	LastReading     time.Time         `json:"last_reading,omitempty"`
	ErrorCount      int               `json:"error_count"`
	LastError       string            `json:"last_error,omitempty"`
	CreatedAt       time.Time         `json:"created_at"`
	UpdatedAt       time.Time         `json:"updated_at"`
	CalibrationData map[string]float64 `json:"calibration_data,omitempty"`
	Firmware        string            `json:"firmware,omitempty"`
	Location        string            `json:"location,omitempty"`
	Tags            []string          `json:"tags,omitempty"`
}

// Validate validates the device data
func (d *Device) Validate() error {
	if d.ID == "" {
		return fmt.Errorf("device ID is required")
	}
	
	if d.Name == "" {
		return fmt.Errorf("device name is required")
	}
	
	if d.Type == "" {
		return fmt.Errorf("device type is required")
	}
	
	if !isValidDeviceType(d.Type) {
		return fmt.Errorf("invalid device type: %s", d.Type)
	}
	
	if d.HardwareID == "" {
		return fmt.Errorf("hardware ID is required")
	}
	
	if d.ConnectionType == "" {
		return fmt.Errorf("connection type is required")
	}
	
	if !isValidConnectionType(d.ConnectionType) {
		return fmt.Errorf("invalid connection type: %s", d.ConnectionType)
	}
	
	if d.Address == "" {
		return fmt.Errorf("device address is required")
	}
	
	return nil
}

// IsOnline returns true if the device is currently online
func (d *Device) IsOnline() bool {
	return d.Status == DeviceStatusOnline
}

// IsHealthy returns true if the device is healthy (online with no recent errors)
func (d *Device) IsHealthy() bool {
	return d.Status == DeviceStatusOnline && d.ErrorCount < 5
}

// IsSensor returns true if the device is a sensor
func (d *Device) IsSensor() bool {
	sensorTypes := []DeviceType{
		DeviceTypeMoistureSensor,
		DeviceTypeTemperatureSensor,
		DeviceTypeHumiditySensor,
		DeviceTypeLightSensor,
		DeviceTypePHSensor,
		DeviceTypeCamera,
	}
	
	for _, sensorType := range sensorTypes {
		if d.Type == sensorType {
			return true
		}
	}
	
	return false
}

// IsActuator returns true if the device is an actuator
func (d *Device) IsActuator() bool {
	actuatorTypes := []DeviceType{
		DeviceTypeWaterPump,
		DeviceTypeFan,
		DeviceTypeLED,
		DeviceTypeValve,
	}
	
	for _, actuatorType := range actuatorTypes {
		if d.Type == actuatorType {
			return true
		}
	}
	
	return false
}

// UpdateStatus updates the device status and last seen time
func (d *Device) UpdateStatus(status DeviceStatus) {
	d.Status = status
	d.LastSeen = time.Now().UTC()
	d.UpdatedAt = time.Now().UTC()
}

// RecordError records an error for the device
func (d *Device) RecordError(errorMsg string) {
	d.ErrorCount++
	d.LastError = errorMsg
	d.Status = DeviceStatusError
	d.UpdatedAt = time.Now().UTC()
}

// ClearErrors clears the error count and status
func (d *Device) ClearErrors() {
	d.ErrorCount = 0
	d.LastError = ""
	if d.Status == DeviceStatusError {
		d.Status = DeviceStatusOnline
	}
	d.UpdatedAt = time.Now().UTC()
}

// GetExpectedReadingInterval returns the expected interval between readings for this device type
func (d *Device) GetExpectedReadingInterval() time.Duration {
	switch d.Type {
	case DeviceTypeMoistureSensor:
		return 5 * time.Minute
	case DeviceTypeTemperatureSensor, DeviceTypeHumiditySensor:
		return 2 * time.Minute
	case DeviceTypeLightSensor:
		return 1 * time.Minute
	case DeviceTypePHSensor:
		return 10 * time.Minute
	case DeviceTypeCamera:
		return 1 * time.Hour
	default:
		return 5 * time.Minute
	}
}

// isValidDeviceType checks if the device type is valid
func isValidDeviceType(deviceType DeviceType) bool {
	validTypes := []DeviceType{
		DeviceTypeMoistureSensor,
		DeviceTypeTemperatureSensor,
		DeviceTypeHumiditySensor,
		DeviceTypeLightSensor,
		DeviceTypePHSensor,
		DeviceTypeCamera,
		DeviceTypeWaterPump,
		DeviceTypeFan,
		DeviceTypeLED,
		DeviceTypeValve,
	}
	
	for _, validType := range validTypes {
		if deviceType == validType {
			return true
		}
	}
	
	return false
}

// isValidConnectionType checks if the connection type is valid
func isValidConnectionType(connectionType ConnectionType) bool {
	validTypes := []ConnectionType{
		ConnectionTypeI2C,
		ConnectionTypeSPI,
		ConnectionTypeGPIO,
		ConnectionTypeUSB,
		ConnectionTypeSerial,
		ConnectionTypeNetwork,
	}
	
	for _, validType := range validTypes {
		if connectionType == validType {
			return true
		}
	}
	
	return false
}

// DeviceReading represents a reading from a device
type DeviceReading struct {
	DeviceID    string                 `json:"device_id"`
	PlantID     string                 `json:"plant_id,omitempty"`
	SensorType  string                 `json:"sensor_type"`
	Value       float64                `json:"value"`
	Unit        string                 `json:"unit"`
	Timestamp   time.Time              `json:"timestamp"`
	Quality     string                 `json:"quality"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// DeviceCommand represents a command to send to a device
type DeviceCommand struct {
	ID         string                 `json:"id"`
	DeviceID   string                 `json:"device_id"`
	Command    string                 `json:"command"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Timestamp  time.Time              `json:"timestamp"`
	ExpiresAt  time.Time              `json:"expires_at"`
	Status     string                 `json:"status"`
	Result     map[string]interface{} `json:"result,omitempty"`
	Error      string                 `json:"error,omitempty"`
}

// PollingStatus represents the status of device polling
type PollingStatus struct {
	Active           bool              `json:"active"`
	TotalDevices     int               `json:"total_devices"`
	OnlineDevices    int               `json:"online_devices"`
	OfflineDevices   int               `json:"offline_devices"`
	ErrorDevices     int               `json:"error_devices"`
	LastPollTime     time.Time         `json:"last_poll_time"`
	AverageLatency   time.Duration     `json:"average_latency"`
	SuccessRate      float64           `json:"success_rate"`
	WorkerCount      int               `json:"worker_count"`
	QueueSize        int               `json:"queue_size"`
	DeviceStats      map[string]int    `json:"device_stats"`
}

// DeviceManagerStats represents statistics for the device manager
type DeviceManagerStats struct {
	ManagedDevices    int               `json:"managed_devices"`
	ActivePolling     int               `json:"active_polling"`
	TotalReadings     int64             `json:"total_readings"`
	FailedReadings    int64             `json:"failed_readings"`
	LastReading       time.Time         `json:"last_reading"`
	Uptime            time.Duration     `json:"uptime"`
	CircuitBreakers   map[string]string `json:"circuit_breakers"`
	WorkerPoolStatus  WorkerPoolStatus  `json:"worker_pool_status"`
}

// WorkerPoolStatus represents the status of the worker pool
type WorkerPoolStatus struct {
	Size        int `json:"size"`
	Active      int `json:"active"`
	Idle        int `json:"idle"`
	QueueLength int `json:"queue_length"`
}