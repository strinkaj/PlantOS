package streaming

import "errors"

// Common streaming service errors
var (
	ErrBufferFull           = errors.New("buffer is full")
	ErrInvalidData          = errors.New("invalid sensor data")
	ErrSubscriberNotFound   = errors.New("subscriber not found")
	ErrKafkaUnavailable     = errors.New("kafka is unavailable")
	ErrRedisUnavailable     = errors.New("redis is unavailable")
	ErrDatabaseUnavailable  = errors.New("database is unavailable")
	ErrServiceUnavailable   = errors.New("service is unavailable")
	ErrInvalidConfiguration = errors.New("invalid configuration")
	ErrShutdownTimeout      = errors.New("shutdown timeout exceeded")
)