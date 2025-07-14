package streaming

import (
	"context"
	"testing"
	"time"

	"plantos/streaming/internal/config"
	"plantos/streaming/internal/types"
)

func TestServiceCreation(t *testing.T) {
	cfg := &config.Config{
		Port:            8080,
		KafkaURL:        "localhost:9092",
		RedisURL:        "redis://localhost:6379",
		PostgresURL:     "postgres://user:pass@localhost:5432/db",
		BufferSize:      1000,
		FlushInterval:   5 * time.Second,
		MaxRetries:      3,
		MetricsEnabled:  true,
		LogLevel:        "info",
		Environment:     "test",
		ServiceName:     "test-streaming",
		ServiceVersion:  "1.0.0",
		GRPCPort:        9090,
		HealthCheckPath: "/health",
		MetricsPath:     "/metrics",
		ShutdownTimeout: 30 * time.Second,
	}

	service := NewService(cfg)
	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.config != cfg {
		t.Fatal("Expected config to be set")
	}

	if service.buffer == nil {
		t.Fatal("Expected buffer to be initialized")
	}

	if service.subscribers == nil {
		t.Fatal("Expected subscribers map to be initialized")
	}
}

func TestSensorDataValidation(t *testing.T) {
	tests := []struct {
		name      string
		data      types.SensorData
		expectErr bool
	}{
		{
			name: "valid sensor data",
			data: types.SensorData{
				ID:         "test-1",
				PlantID:    "plant-1",
				SensorID:   "sensor-1",
				SensorType: types.SensorTypeMoisture,
				Value:      50.0,
				Unit:       "%",
				Timestamp:  time.Now(),
				Source:     "test",
			},
			expectErr: false,
		},
		{
			name: "missing plant ID",
			data: types.SensorData{
				ID:         "test-1",
				SensorID:   "sensor-1",
				SensorType: types.SensorTypeMoisture,
				Value:      50.0,
				Unit:       "%",
				Timestamp:  time.Now(),
				Source:     "test",
			},
			expectErr: true,
		},
		{
			name: "invalid sensor type",
			data: types.SensorData{
				ID:         "test-1",
				PlantID:    "plant-1",
				SensorID:   "sensor-1",
				SensorType: "invalid",
				Value:      50.0,
				Unit:       "%",
				Timestamp:  time.Now(),
				Source:     "test",
			},
			expectErr: true,
		},
		{
			name: "moisture value out of range",
			data: types.SensorData{
				ID:         "test-1",
				PlantID:    "plant-1",
				SensorID:   "sensor-1",
				SensorType: types.SensorTypeMoisture,
				Value:      150.0, // Invalid: > 100
				Unit:       "%",
				Timestamp:  time.Now(),
				Source:     "test",
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.data.Validate()
			if tt.expectErr && err == nil {
				t.Errorf("Expected error but got none")
			}
			if !tt.expectErr && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

func TestServiceIngestion(t *testing.T) {
	cfg := &config.Config{
		Port:            8080,
		KafkaURL:        "localhost:9092",
		RedisURL:        "redis://localhost:6379",
		PostgresURL:     "postgres://user:pass@localhost:5432/db",
		BufferSize:      10, // Small buffer for testing
		FlushInterval:   5 * time.Second,
		MaxRetries:      3,
		MetricsEnabled:  true,
		LogLevel:        "info",
		Environment:     "test",
		ServiceName:     "test-streaming",
		ServiceVersion:  "1.0.0",
		GRPCPort:        9090,
		HealthCheckPath: "/health",
		MetricsPath:     "/metrics",
		ShutdownTimeout: 30 * time.Second,
	}

	service := NewService(cfg)

	// Test valid data ingestion
	validData := types.SensorData{
		ID:         "test-1",
		PlantID:    "plant-1",
		SensorID:   "sensor-1",
		SensorType: types.SensorTypeMoisture,
		Value:      50.0,
		Unit:       "%",
		Timestamp:  time.Now(),
		Source:     "test",
	}

	err := service.IngestSensorData(validData)
	if err != nil {
		t.Errorf("Expected no error for valid data, got: %v", err)
	}

	// Test buffer overflow
	for i := 0; i < cfg.BufferSize+1; i++ {
		data := types.SensorData{
			ID:         "test-" + string(rune(i)),
			PlantID:    "plant-1",
			SensorID:   "sensor-1",
			SensorType: types.SensorTypeMoisture,
			Value:      50.0,
			Unit:       "%",
			Timestamp:  time.Now(),
			Source:     "test",
		}
		service.IngestSensorData(data)
	}

	// The buffer should be full now
	err = service.IngestSensorData(validData)
	if err != ErrBufferFull {
		t.Errorf("Expected ErrBufferFull, got: %v", err)
	}
}

func TestSubscription(t *testing.T) {
	cfg := &config.Config{
		Port:            8080,
		KafkaURL:        "localhost:9092",
		RedisURL:        "redis://localhost:6379",
		PostgresURL:     "postgres://user:pass@localhost:5432/db",
		BufferSize:      1000,
		FlushInterval:   5 * time.Second,
		MaxRetries:      3,
		MetricsEnabled:  true,
		LogLevel:        "info",
		Environment:     "test",
		ServiceName:     "test-streaming",
		ServiceVersion:  "1.0.0",
		GRPCPort:        9090,
		HealthCheckPath: "/health",
		MetricsPath:     "/metrics",
		ShutdownTimeout: 30 * time.Second,
	}

	service := NewService(cfg)

	// Test subscription
	subscriberID := "test-subscriber"
	ch := service.Subscribe(subscriberID)
	if ch == nil {
		t.Fatal("Expected channel to be returned")
	}

	// Test data delivery to subscriber
	testData := types.SensorData{
		ID:         "test-1",
		PlantID:    "plant-1",
		SensorID:   "sensor-1",
		SensorType: types.SensorTypeMoisture,
		Value:      50.0,
		Unit:       "%",
		Timestamp:  time.Now(),
		Source:     "test",
	}

	// Start a goroutine to consume from the channel
	received := make(chan types.SensorData, 1)
	go func() {
		data := <-ch
		received <- data
	}()

	// Ingest data
	err := service.IngestSensorData(testData)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	// Check if data was received
	select {
	case data := <-received:
		if data.ID != testData.ID {
			t.Errorf("Expected data ID %s, got %s", testData.ID, data.ID)
		}
	case <-time.After(1 * time.Second):
		t.Error("Expected to receive data within 1 second")
	}

	// Test unsubscription
	service.Unsubscribe(subscriberID)

	// Channel should be closed after unsubscription
	select {
	case _, ok := <-ch:
		if ok {
			t.Error("Expected channel to be closed")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Expected channel to be closed immediately")
	}
}

func TestServiceStartStop(t *testing.T) {
	cfg := &config.Config{
		Port:            8080,
		KafkaURL:        "localhost:9092",
		RedisURL:        "redis://localhost:6379",
		PostgresURL:     "postgres://user:pass@localhost:5432/db",
		BufferSize:      1000,
		FlushInterval:   5 * time.Second,
		MaxRetries:      3,
		MetricsEnabled:  true,
		LogLevel:        "info",
		Environment:     "test",
		ServiceName:     "test-streaming",
		ServiceVersion:  "1.0.0",
		GRPCPort:        9090,
		HealthCheckPath: "/health",
		MetricsPath:     "/metrics",
		ShutdownTimeout: 30 * time.Second,
	}

	service := NewService(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	
	// Start service in a goroutine
	done := make(chan error, 1)
	go func() {
		done <- service.Start(ctx)
	}()

	// Give it a moment to start
	time.Sleep(100 * time.Millisecond)

	// Stop the service
	cancel()

	// Wait for service to stop
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Expected no error on stop, got: %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Error("Service did not stop within 1 second")
	}
}