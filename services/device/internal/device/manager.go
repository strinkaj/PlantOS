package device

import (
	"context"
	"fmt"
	"sync"
	"time"

	"plantos/device/internal/config"
	"plantos/device/internal/types"
)

// Manager manages device operations and polling
type Manager struct {
	config          *config.Config
	devices         map[string]*types.Device
	devicesMu       sync.RWMutex
	pollingActive   bool
	pollingMu       sync.RWMutex
	workerPool      *WorkerPool
	circuitBreakers map[string]*CircuitBreaker
	cbMu            sync.RWMutex
	deviceClient    DeviceClient
	streamingClient StreamingClient
	dbClient        DatabaseClient
	metrics         Metrics
	logger          Logger
	startTime       time.Time
	stats           types.DeviceManagerStats
	statsMu         sync.RWMutex
}

// DeviceClient interface for hardware device operations
type DeviceClient interface {
	ReadSensor(deviceID, sensorType string, timeout time.Duration) (float64, error)
	ExecuteCommand(deviceID, command string, params map[string]interface{}, timeout time.Duration) error
	DiscoverDevices() ([]types.Device, error)
	ValidateDevice(device *types.Device) error
	Close() error
}

// StreamingClient interface for sending data to streaming service
type StreamingClient interface {
	SendReading(reading types.DeviceReading) error
	SendBatchReadings(readings []types.DeviceReading) error
	Close() error
}

// DatabaseClient interface for device persistence
type DatabaseClient interface {
	SaveDevice(device *types.Device) error
	LoadDevices() ([]types.Device, error)
	UpdateDeviceStatus(deviceID string, status types.DeviceStatus) error
	DeleteDevice(deviceID string) error
	Close() error
}

// Metrics interface for monitoring
type Metrics interface {
	IncrementCounter(name string, labels map[string]string)
	RecordHistogram(name string, value float64, labels map[string]string)
	RecordGauge(name string, value float64, labels map[string]string)
}

// Logger interface for structured logging
type Logger interface {
	Info(msg string, fields ...interface{})
	Error(msg string, fields ...interface{})
	Debug(msg string, fields ...interface{})
	Warn(msg string, fields ...interface{})
}

// NewManager creates a new device manager
func NewManager(cfg *config.Config) *Manager {
	return &Manager{
		config:          cfg,
		devices:         make(map[string]*types.Device),
		circuitBreakers: make(map[string]*CircuitBreaker),
		workerPool:      NewWorkerPool(cfg.WorkerPoolSize),
		deviceClient:    &mockDeviceClient{},
		streamingClient: &mockStreamingClient{},
		dbClient:        &mockDatabaseClient{},
		metrics:         &mockMetrics{},
		logger:          &mockLogger{},
		startTime:       time.Now().UTC(),
	}
}

// Start starts the device manager
func (m *Manager) Start(ctx context.Context) error {
	m.logger.Info("Starting device manager",
		"service", m.config.ServiceName,
		"version", m.config.ServiceVersion,
		"port", m.config.Port)

	// Load existing devices from database
	if err := m.loadDevices(); err != nil {
		m.logger.Error("Failed to load devices from database", "error", err)
		return fmt.Errorf("failed to load devices: %w", err)
	}

	// Start worker pool
	m.workerPool.Start(ctx)

	// Discover devices if enabled
	if m.config.DeviceDiscovery {
		go m.deviceDiscoveryLoop(ctx)
	}

	// Start metrics collection
	go m.metricsLoop(ctx)

	<-ctx.Done()
	m.logger.Info("Device manager stopped")
	return nil
}

// RegisterDevice registers a new device
func (m *Manager) RegisterDevice(device *types.Device) error {
	if err := device.Validate(); err != nil {
		m.logger.Error("Invalid device registration", "error", err, "device_id", device.ID)
		return fmt.Errorf("device validation failed: %w", err)
	}

	// Validate with hardware interface
	if err := m.deviceClient.ValidateDevice(device); err != nil {
		m.logger.Error("Device hardware validation failed", "error", err, "device_id", device.ID)
		return fmt.Errorf("hardware validation failed: %w", err)
	}

	// Set timestamps
	now := time.Now().UTC()
	device.CreatedAt = now
	device.UpdatedAt = now
	device.Status = types.DeviceStatusOffline

	// Save to database
	if err := m.dbClient.SaveDevice(device); err != nil {
		m.logger.Error("Failed to save device to database", "error", err, "device_id", device.ID)
		return fmt.Errorf("failed to save device: %w", err)
	}

	// Add to in-memory store
	m.devicesMu.Lock()
	m.devices[device.ID] = device
	m.devicesMu.Unlock()

	// Create circuit breaker for device
	m.cbMu.Lock()
	m.circuitBreakers[device.ID] = NewCircuitBreaker(
		m.config.CircuitBreakerThreshold,
		m.config.CircuitBreakerTimeout,
	)
	m.cbMu.Unlock()

	m.logger.Info("Device registered successfully",
		"device_id", device.ID,
		"device_type", device.Type,
		"plant_id", device.PlantID)

	m.metrics.IncrementCounter("devices_registered", map[string]string{
		"device_type": string(device.Type),
		"plant_id":    device.PlantID,
	})

	return nil
}

// GetDevice retrieves a device by ID
func (m *Manager) GetDevice(deviceID string) (*types.Device, error) {
	m.devicesMu.RLock()
	device, exists := m.devices[deviceID]
	m.devicesMu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("device not found: %s", deviceID)
	}

	return device, nil
}

// ListDevices returns all registered devices
func (m *Manager) ListDevices() []*types.Device {
	m.devicesMu.RLock()
	defer m.devicesMu.RUnlock()

	devices := make([]*types.Device, 0, len(m.devices))
	for _, device := range m.devices {
		devices = append(devices, device)
	}

	return devices
}

// GetDevicesByPlant returns devices for a specific plant
func (m *Manager) GetDevicesByPlant(plantID string) []*types.Device {
	m.devicesMu.RLock()
	defer m.devicesMu.RUnlock()

	var devices []*types.Device
	for _, device := range m.devices {
		if device.PlantID == plantID {
			devices = append(devices, device)
		}
	}

	return devices
}

// StartPolling starts device polling
func (m *Manager) StartPolling() error {
	m.pollingMu.Lock()
	defer m.pollingMu.Unlock()

	if m.pollingActive {
		return fmt.Errorf("polling is already active")
	}

	m.pollingActive = true
	m.logger.Info("Device polling started")

	// Start polling goroutine
	go m.pollingLoop()

	return nil
}

// StopPolling stops device polling
func (m *Manager) StopPolling() error {
	m.pollingMu.Lock()
	defer m.pollingMu.Unlock()

	if !m.pollingActive {
		return fmt.Errorf("polling is not active")
	}

	m.pollingActive = false
	m.logger.Info("Device polling stopped")

	return nil
}

// GetPollingStatus returns the current polling status
func (m *Manager) GetPollingStatus() types.PollingStatus {
	m.pollingMu.RLock()
	active := m.pollingActive
	m.pollingMu.RUnlock()

	m.devicesMu.RLock()
	defer m.devicesMu.RUnlock()

	status := types.PollingStatus{
		Active:         active,
		TotalDevices:   len(m.devices),
		DeviceStats:    make(map[string]int),
		WorkerCount:    m.workerPool.Size(),
		QueueSize:      m.workerPool.QueueSize(),
	}

	// Count devices by status
	for _, device := range m.devices {
		switch device.Status {
		case types.DeviceStatusOnline:
			status.OnlineDevices++
		case types.DeviceStatusOffline:
			status.OfflineDevices++
		case types.DeviceStatusError:
			status.ErrorDevices++
		}

		// Count by type
		typeKey := string(device.Type)
		status.DeviceStats[typeKey]++
	}

	return status
}

// pollingLoop runs the main polling loop
func (m *Manager) pollingLoop() {
	ticker := time.NewTicker(m.config.PollingInterval)
	defer ticker.Stop()

	for {
		m.pollingMu.RLock()
		active := m.pollingActive
		m.pollingMu.RUnlock()

		if !active {
			break
		}

		select {
		case <-ticker.C:
			m.pollAllDevices()
		}
	}
}

// pollAllDevices polls all registered sensor devices
func (m *Manager) pollAllDevices() {
	m.devicesMu.RLock()
	devices := make([]*types.Device, 0, len(m.devices))
	for _, device := range m.devices {
		if device.IsSensor() && device.IsOnline() {
			devices = append(devices, device)
		}
	}
	m.devicesMu.RUnlock()

	m.logger.Debug("Polling devices", "count", len(devices))

	// Submit polling tasks to worker pool
	for _, device := range devices {
		device := device // Capture for closure
		task := func() {
			m.pollDevice(device)
		}
		m.workerPool.Submit(task)
	}
}

// pollDevice polls a single device
func (m *Manager) pollDevice(device *types.Device) {
	start := time.Now()

	// Check circuit breaker
	m.cbMu.RLock()
	cb, exists := m.circuitBreakers[device.ID]
	m.cbMu.RUnlock()

	if exists && cb.IsOpen() {
		m.logger.Debug("Circuit breaker open, skipping device poll", "device_id", device.ID)
		return
	}

	// Read from device
	sensorType := string(device.Type)
	value, err := m.deviceClient.ReadSensor(device.ID, sensorType, m.config.DeviceTimeout)

	if err != nil {
		m.handleDeviceError(device, err)
		if exists {
			cb.RecordFailure()
		}
		return
	}

	// Success - record and send data
	if exists {
		cb.RecordSuccess()
	}

	reading := types.DeviceReading{
		DeviceID:   device.ID,
		PlantID:    device.PlantID,
		SensorType: sensorType,
		Value:      value,
		Unit:       m.getUnitForSensorType(sensorType),
		Timestamp:  time.Now().UTC(),
		Quality:    "good",
	}

	// Send to streaming service
	if err := m.streamingClient.SendReading(reading); err != nil {
		m.logger.Error("Failed to send reading to streaming service",
			"error", err,
			"device_id", device.ID)
	}

	// Update device status
	device.UpdateStatus(types.DeviceStatusOnline)
	device.LastReading = reading.Timestamp
	device.ClearErrors()

	duration := time.Since(start)
	m.metrics.RecordHistogram("device_poll_duration", duration.Seconds(), map[string]string{
		"device_id":   device.ID,
		"device_type": string(device.Type),
		"plant_id":    device.PlantID,
	})

	m.logger.Debug("Device polled successfully",
		"device_id", device.ID,
		"value", value,
		"duration", duration)
}

// handleDeviceError handles errors from device polling
func (m *Manager) handleDeviceError(device *types.Device, err error) {
	device.RecordError(err.Error())

	m.logger.Error("Device polling failed",
		"device_id", device.ID,
		"error", err,
		"error_count", device.ErrorCount)

	m.metrics.IncrementCounter("device_poll_errors", map[string]string{
		"device_id":   device.ID,
		"device_type": string(device.Type),
		"plant_id":    device.PlantID,
	})

	// Update status in database
	if err := m.dbClient.UpdateDeviceStatus(device.ID, device.Status); err != nil {
		m.logger.Error("Failed to update device status in database",
			"error", err,
			"device_id", device.ID)
	}
}

// getUnitForSensorType returns the appropriate unit for a sensor type
func (m *Manager) getUnitForSensorType(sensorType string) string {
	switch sensorType {
	case string(types.DeviceTypeMoistureSensor):
		return "%"
	case string(types.DeviceTypeTemperatureSensor):
		return "°C"
	case string(types.DeviceTypeHumiditySensor):
		return "%"
	case string(types.DeviceTypeLightSensor):
		return "lux"
	case string(types.DeviceTypePHSensor):
		return "pH"
	default:
		return ""
	}
}

// loadDevices loads devices from the database
func (m *Manager) loadDevices() error {
	devices, err := m.dbClient.LoadDevices()
	if err != nil {
		return err
	}

	m.devicesMu.Lock()
	defer m.devicesMu.Unlock()

	for _, device := range devices {
		device := device // Copy for pointer
		m.devices[device.ID] = &device

		// Create circuit breaker
		m.cbMu.Lock()
		m.circuitBreakers[device.ID] = NewCircuitBreaker(
			m.config.CircuitBreakerThreshold,
			m.config.CircuitBreakerTimeout,
		)
		m.cbMu.Unlock()
	}

	m.logger.Info("Loaded devices from database", "count", len(devices))
	return nil
}

// deviceDiscoveryLoop runs device discovery periodically
func (m *Manager) deviceDiscoveryLoop(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.discoverDevices()
		case <-ctx.Done():
			return
		}
	}
}

// discoverDevices discovers new devices
func (m *Manager) discoverDevices() {
	m.logger.Debug("Discovering devices")

	discoveredDevices, err := m.deviceClient.DiscoverDevices()
	if err != nil {
		m.logger.Error("Device discovery failed", "error", err)
		return
	}

	newDeviceCount := 0
	for _, device := range discoveredDevices {
		m.devicesMu.RLock()
		_, exists := m.devices[device.ID]
		m.devicesMu.RUnlock()

		if !exists {
			if err := m.RegisterDevice(&device); err != nil {
				m.logger.Error("Failed to register discovered device",
					"error", err,
					"device_id", device.ID)
			} else {
				newDeviceCount++
			}
		}
	}

	if newDeviceCount > 0 {
		m.logger.Info("Discovered new devices", "count", newDeviceCount)
	}
}

// metricsLoop collects and reports metrics
func (m *Manager) metricsLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.collectMetrics()
		case <-ctx.Done():
			return
		}
	}
}

// collectMetrics collects current metrics
func (m *Manager) collectMetrics() {
	m.devicesMu.RLock()
	totalDevices := len(m.devices)
	onlineDevices := 0
	errorDevices := 0

	for _, device := range m.devices {
		if device.IsOnline() {
			onlineDevices++
		}
		if device.Status == types.DeviceStatusError {
			errorDevices++
		}
	}
	m.devicesMu.RUnlock()

	m.pollingMu.RLock()
	pollingActive := m.pollingActive
	m.pollingMu.RUnlock()

	m.metrics.RecordGauge("total_devices", float64(totalDevices), nil)
	m.metrics.RecordGauge("online_devices", float64(onlineDevices), nil)
	m.metrics.RecordGauge("error_devices", float64(errorDevices), nil)
	m.metrics.RecordGauge("polling_active", boolToFloat64(pollingActive), nil)
	m.metrics.RecordGauge("worker_pool_size", float64(m.workerPool.Size()), nil)
	m.metrics.RecordGauge("worker_pool_queue_size", float64(m.workerPool.QueueSize()), nil)
}

// GetStats returns current manager statistics
func (m *Manager) GetStats() types.DeviceManagerStats {
	m.statsMu.RLock()
	defer m.statsMu.RUnlock()

	m.devicesMu.RLock()
	deviceCount := len(m.devices)
	m.devicesMu.RUnlock()

	m.pollingMu.RLock()
	pollingActive := m.pollingActive
	m.pollingMu.RUnlock()

	stats := types.DeviceManagerStats{
		ManagedDevices:   deviceCount,
		ActivePolling:    boolToInt(pollingActive),
		Uptime:          time.Since(m.startTime),
		CircuitBreakers: make(map[string]string),
		WorkerPoolStatus: types.WorkerPoolStatus{
			Size:        m.workerPool.Size(),
			QueueLength: m.workerPool.QueueSize(),
		},
	}

	// Get circuit breaker statuses
	m.cbMu.RLock()
	for deviceID, cb := range m.circuitBreakers {
		if cb.IsOpen() {
			stats.CircuitBreakers[deviceID] = "open"
		} else {
			stats.CircuitBreakers[deviceID] = "closed"
		}
	}
	m.cbMu.RUnlock()

	return stats
}

// Close closes the device manager
func (m *Manager) Close() error {
	m.logger.Info("Closing device manager")

	// Stop polling
	m.StopPolling()

	// Close worker pool
	m.workerPool.Close()

	// Close clients
	if err := m.deviceClient.Close(); err != nil {
		m.logger.Error("Failed to close device client", "error", err)
	}

	if err := m.streamingClient.Close(); err != nil {
		m.logger.Error("Failed to close streaming client", "error", err)
	}

	if err := m.dbClient.Close(); err != nil {
		m.logger.Error("Failed to close database client", "error", err)
	}

	m.logger.Info("Device manager closed")
	return nil
}

// Helper functions
func boolToFloat64(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}