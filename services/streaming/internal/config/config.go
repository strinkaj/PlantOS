package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config holds the configuration for the streaming service
type Config struct {
	Port             int           `json:"port"`
	KafkaURL         string        `json:"kafka_url"`
	RedisURL         string        `json:"redis_url"`
	PostgresURL      string        `json:"postgres_url"`
	BufferSize       int           `json:"buffer_size"`
	FlushInterval    time.Duration `json:"flush_interval"`
	MaxRetries       int           `json:"max_retries"`
	MetricsEnabled   bool          `json:"metrics_enabled"`
	LogLevel         string        `json:"log_level"`
	Environment      string        `json:"environment"`
	ServiceName      string        `json:"service_name"`
	ServiceVersion   string        `json:"service_version"`
	GRPCPort         int           `json:"grpc_port"`
	HealthCheckPath  string        `json:"health_check_path"`
	MetricsPath      string        `json:"metrics_path"`
	ShutdownTimeout  time.Duration `json:"shutdown_timeout"`
}

// Load loads configuration from environment variables with sensible defaults
func Load() (*Config, error) {
	config := &Config{
		Port:             getEnvInt("STREAMING_PORT", 8080),
		KafkaURL:         getEnvString("KAFKA_URL", "localhost:9092"),
		RedisURL:         getEnvString("REDIS_URL", "redis://localhost:6379"),
		PostgresURL:      getEnvString("POSTGRES_URL", "postgres://user:password@localhost:5432/plantos"),
		BufferSize:       getEnvInt("BUFFER_SIZE", 1000),
		FlushInterval:    getEnvDuration("FLUSH_INTERVAL", 5*time.Second),
		MaxRetries:       getEnvInt("MAX_RETRIES", 3),
		MetricsEnabled:   getEnvBool("METRICS_ENABLED", true),
		LogLevel:         getEnvString("LOG_LEVEL", "info"),
		Environment:      getEnvString("ENVIRONMENT", "development"),
		ServiceName:      getEnvString("SERVICE_NAME", "plantos-streaming"),
		ServiceVersion:   getEnvString("SERVICE_VERSION", "1.0.0"),
		GRPCPort:         getEnvInt("GRPC_PORT", 9090),
		HealthCheckPath:  getEnvString("HEALTH_CHECK_PATH", "/health"),
		MetricsPath:      getEnvString("METRICS_PATH", "/metrics"),
		ShutdownTimeout:  getEnvDuration("SHUTDOWN_TIMEOUT", 30*time.Second),
	}

	// Validate configuration
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return config, nil
}

// validate validates the configuration
func (c *Config) validate() error {
	if c.Port <= 0 || c.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Port)
	}

	if c.GRPCPort <= 0 || c.GRPCPort > 65535 {
		return fmt.Errorf("invalid gRPC port: %d", c.GRPCPort)
	}

	if c.BufferSize <= 0 {
		return fmt.Errorf("buffer size must be positive: %d", c.BufferSize)
	}

	if c.FlushInterval <= 0 {
		return fmt.Errorf("flush interval must be positive: %v", c.FlushInterval)
	}

	if c.MaxRetries < 0 {
		return fmt.Errorf("max retries must be non-negative: %d", c.MaxRetries)
	}

	if c.KafkaURL == "" {
		return fmt.Errorf("kafka URL is required")
	}

	if c.RedisURL == "" {
		return fmt.Errorf("redis URL is required")
	}

	if c.PostgresURL == "" {
		return fmt.Errorf("postgres URL is required")
	}

	return nil
}

// Helper functions for environment variable parsing
func getEnvString(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}