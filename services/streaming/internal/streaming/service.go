package streaming

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"plantos/streaming/internal/config"
	"plantos/streaming/internal/types"
)

// Service represents the streaming service
type Service struct {
	config      *config.Config
	buffer      chan types.SensorData
	subscribers map[string]chan types.SensorData
	mu          sync.RWMutex
	kafkaClient KafkaClient
	redisClient RedisClient
	dbClient    DatabaseClient
	metrics     Metrics
	logger      Logger
}

// KafkaClient interface for Kafka operations
type KafkaClient interface {
	Publish(topic string, data []byte) error
	Subscribe(topic string, handler func([]byte)) error
	Close() error
}

// RedisClient interface for Redis operations
type RedisClient interface {
	Set(key string, value interface{}, expiration time.Duration) error
	Get(key string) (interface{}, error)
	Close() error
}

// DatabaseClient interface for database operations
type DatabaseClient interface {
	BatchInsert(data []types.SensorData) error
	Close() error
}

// Logger interface for structured logging
type Logger interface {
	Info(msg string, fields ...interface{})
	Error(msg string, fields ...interface{})
	Debug(msg string, fields ...interface{})
	Warn(msg string, fields ...interface{})
}

// Metrics interface for monitoring
type Metrics interface {
	IncrementCounter(name string, labels map[string]string)
	RecordHistogram(name string, value float64, labels map[string]string)
	RecordGauge(name string, value float64, labels map[string]string)
}

// NewService creates a new streaming service
func NewService(cfg *config.Config) *Service {
	return &Service{
		config:      cfg,
		buffer:      make(chan types.SensorData, cfg.BufferSize),
		subscribers: make(map[string]chan types.SensorData),
		mu:          sync.RWMutex{},
		// TODO: Initialize actual clients
		kafkaClient: &mockKafkaClient{},
		redisClient: &mockRedisClient{},
		dbClient:    &mockDatabaseClient{},
		metrics:     &mockMetrics{},
		logger:      &mockLogger{},
	}
}

// Start starts the streaming service
func (s *Service) Start(ctx context.Context) error {
	s.logger.Info("Starting streaming service", 
		"service", s.config.ServiceName,
		"version", s.config.ServiceVersion,
		"port", s.config.Port)

	// Start buffer processing goroutine
	go s.processBuffer(ctx)

	// Start periodic flush goroutine
	go s.periodicFlush(ctx)

	// Start metrics collection
	go s.collectMetrics(ctx)

	<-ctx.Done()
	s.logger.Info("Streaming service stopped")
	return nil
}

// IngestSensorData ingests sensor data into the streaming pipeline
func (s *Service) IngestSensorData(data types.SensorData) error {
	// Validate data
	if err := data.Validate(); err != nil {
		s.logger.Error("Invalid sensor data", "error", err)
		s.metrics.IncrementCounter("sensor_data_validation_errors", map[string]string{
			"plant_id": data.PlantID,
			"sensor_type": data.SensorType,
		})
		return err
	}

	// Add to buffer
	select {
	case s.buffer <- data:
		s.metrics.IncrementCounter("sensor_data_ingested", map[string]string{
			"plant_id": data.PlantID,
			"sensor_type": data.SensorType,
		})
		
		// Notify subscribers
		s.notifySubscribers(data)
		
		return nil
	default:
		s.logger.Error("Buffer full, dropping sensor data", 
			"plant_id", data.PlantID,
			"sensor_type", data.SensorType)
		s.metrics.IncrementCounter("sensor_data_dropped", map[string]string{
			"plant_id": data.PlantID,
			"sensor_type": data.SensorType,
		})
		return ErrBufferFull
	}
}

// Subscribe creates a subscription for streaming data
func (s *Service) Subscribe(id string) chan types.SensorData {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	ch := make(chan types.SensorData, 100)
	s.subscribers[id] = ch
	
	s.logger.Info("New subscriber added", "subscriber_id", id)
	s.metrics.IncrementCounter("subscribers_added", map[string]string{
		"subscriber_id": id,
	})
	
	return ch
}

// Unsubscribe removes a subscription
func (s *Service) Unsubscribe(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if ch, exists := s.subscribers[id]; exists {
		close(ch)
		delete(s.subscribers, id)
		
		s.logger.Info("Subscriber removed", "subscriber_id", id)
		s.metrics.IncrementCounter("subscribers_removed", map[string]string{
			"subscriber_id": id,
		})
	}
}

// processBuffer processes buffered sensor data
func (s *Service) processBuffer(ctx context.Context) {
	batch := make([]types.SensorData, 0, s.config.BufferSize/10)
	
	for {
		select {
		case data := <-s.buffer:
			batch = append(batch, data)
			
			// Process batch when it reaches certain size
			if len(batch) >= s.config.BufferSize/10 {
				s.processBatch(batch)
				batch = batch[:0] // Reset slice
			}
			
		case <-ctx.Done():
			// Process remaining data before shutdown
			if len(batch) > 0 {
				s.processBatch(batch)
			}
			return
		}
	}
}

// processBatch processes a batch of sensor data
func (s *Service) processBatch(batch []types.SensorData) {
	start := time.Now()
	
	// Store in database
	if err := s.dbClient.BatchInsert(batch); err != nil {
		s.logger.Error("Failed to insert batch to database", "error", err, "batch_size", len(batch))
		s.metrics.IncrementCounter("database_insert_errors", map[string]string{
			"batch_size": string(rune(len(batch))),
		})
		return
	}
	
	// Publish to Kafka
	for _, data := range batch {
		if jsonData, err := json.Marshal(data); err == nil {
			if err := s.kafkaClient.Publish("sensor-data", jsonData); err != nil {
				s.logger.Error("Failed to publish to Kafka", "error", err)
				s.metrics.IncrementCounter("kafka_publish_errors", map[string]string{
					"plant_id": data.PlantID,
					"sensor_type": data.SensorType,
				})
			}
		}
	}
	
	// Cache recent data in Redis
	s.cacheRecentData(batch)
	
	duration := time.Since(start)
	s.metrics.RecordHistogram("batch_processing_duration", duration.Seconds(), map[string]string{
		"batch_size": string(rune(len(batch))),
	})
	
	s.logger.Info("Processed batch", 
		"batch_size", len(batch),
		"duration", duration)
}

// cacheRecentData caches recent sensor data in Redis
func (s *Service) cacheRecentData(batch []types.SensorData) {
	for _, data := range batch {
		key := "recent:" + data.PlantID + ":" + data.SensorType
		if err := s.redisClient.Set(key, data, 15*time.Minute); err != nil {
			s.logger.Error("Failed to cache recent data", "error", err)
		}
	}
}

// periodicFlush flushes data periodically
func (s *Service) periodicFlush(ctx context.Context) {
	ticker := time.NewTicker(s.config.FlushInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			s.logger.Debug("Periodic flush triggered")
			// Force flush of any remaining buffered data
			// This is handled by the buffer processing goroutine
			
		case <-ctx.Done():
			return
		}
	}
}

// collectMetrics collects and reports metrics
func (s *Service) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			s.mu.RLock()
			subscriberCount := len(s.subscribers)
			s.mu.RUnlock()
			
			bufferSize := len(s.buffer)
			
			s.metrics.RecordGauge("active_subscribers", float64(subscriberCount), nil)
			s.metrics.RecordGauge("buffer_size", float64(bufferSize), nil)
			
		case <-ctx.Done():
			return
		}
	}
}

// notifySubscribers notifies all subscribers of new data
func (s *Service) notifySubscribers(data types.SensorData) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	for id, ch := range s.subscribers {
		select {
		case ch <- data:
		default:
			// Subscriber channel is full, skip
			s.logger.Warn("Subscriber channel full, skipping", "subscriber_id", id)
			s.metrics.IncrementCounter("subscriber_drops", map[string]string{
				"subscriber_id": id,
			})
		}
	}
}

// Close closes the streaming service
func (s *Service) Close() error {
	s.logger.Info("Closing streaming service")
	
	// Close all subscriber channels
	s.mu.Lock()
	for id, ch := range s.subscribers {
		close(ch)
		delete(s.subscribers, id)
	}
	s.mu.Unlock()
	
	// Close clients
	if err := s.kafkaClient.Close(); err != nil {
		s.logger.Error("Failed to close Kafka client", "error", err)
	}
	
	if err := s.redisClient.Close(); err != nil {
		s.logger.Error("Failed to close Redis client", "error", err)
	}
	
	if err := s.dbClient.Close(); err != nil {
		s.logger.Error("Failed to close database client", "error", err)
	}
	
	close(s.buffer)
	
	s.logger.Info("Streaming service closed")
	return nil
}