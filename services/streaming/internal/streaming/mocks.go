package streaming

import (
	"log"
	"time"

	"plantos/streaming/internal/types"
)

// Mock implementations for development and testing

// mockKafkaClient is a mock implementation of KafkaClient
type mockKafkaClient struct{}

func (m *mockKafkaClient) Publish(topic string, data []byte) error {
	log.Printf("MockKafka: Publishing to topic %s: %s", topic, string(data))
	return nil
}

func (m *mockKafkaClient) Subscribe(topic string, handler func([]byte)) error {
	log.Printf("MockKafka: Subscribing to topic %s", topic)
	return nil
}

func (m *mockKafkaClient) Close() error {
	log.Println("MockKafka: Closing connection")
	return nil
}

// mockRedisClient is a mock implementation of RedisClient
type mockRedisClient struct {
	data map[string]interface{}
}

func (m *mockRedisClient) Set(key string, value interface{}, expiration time.Duration) error {
	if m.data == nil {
		m.data = make(map[string]interface{})
	}
	m.data[key] = value
	log.Printf("MockRedis: Set key %s with expiration %v", key, expiration)
	return nil
}

func (m *mockRedisClient) Get(key string) (interface{}, error) {
	if m.data == nil {
		return nil, nil
	}
	value, exists := m.data[key]
	log.Printf("MockRedis: Get key %s, exists: %t", key, exists)
	return value, nil
}

func (m *mockRedisClient) Close() error {
	log.Println("MockRedis: Closing connection")
	return nil
}

// mockDatabaseClient is a mock implementation of DatabaseClient
type mockDatabaseClient struct{}

func (m *mockDatabaseClient) BatchInsert(data []types.SensorData) error {
	log.Printf("MockDatabase: Batch inserting %d records", len(data))
	for _, record := range data {
		log.Printf("MockDatabase: Inserting record - PlantID: %s, SensorType: %s, Value: %f", 
			record.PlantID, record.SensorType, record.Value)
	}
	return nil
}

func (m *mockDatabaseClient) Close() error {
	log.Println("MockDatabase: Closing connection")
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