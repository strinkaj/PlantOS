package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config holds the configuration for the device manager service
type Config struct {
	Port                int           `json:"port"`
	GRPCPort           int           `json:"grpc_port"`
	StreamingServiceURL string        `json:"streaming_service_url"`
	PostgresURL        string        `json:"postgres_url"`
	RedisURL           string        `json:"redis_url"`
	PollingInterval    time.Duration `json:"polling_interval"`
	MaxConcurrentPolls int           `json:"max_concurrent_polls"`
	DeviceTimeout      time.Duration `json:"device_timeout"`
	RetryAttempts      int           `json:"retry_attempts"`
	RetryDelay         time.Duration `json:"retry_delay"`
	CircuitBreakerTimeout time.Duration `json:"circuit_breaker_timeout"`
	CircuitBreakerThreshold int       `json:"circuit_breaker_threshold"`
	MetricsEnabled     bool          `json:"metrics_enabled"`
	LogLevel           string        `json:"log_level"`
	Environment        string        `json:"environment"`
	ServiceName        string        `json:"service_name"`
	ServiceVersion     string        `json:"service_version"`
	HealthCheckPath    string        `json:"health_check_path"`
	MetricsPath        string        `json:"metrics_path"`
	ShutdownTimeout    time.Duration `json:"shutdown_timeout"`
	HardwareInterface  string        `json:"hardware_interface"`
	MockHardware       bool          `json:"mock_hardware"`
	DeviceDiscovery    bool          `json:"device_discovery"`
	WorkerPoolSize     int           `json:"worker_pool_size"`
}

// Load loads configuration from environment variables with sensible defaults
func Load() (*Config, error) {
	config := &Config{
		Port:                    getEnvInt("DEVICE_PORT", 8081),
		GRPCPort:               getEnvInt("DEVICE_GRPC_PORT", 9091),
		StreamingServiceURL:    getEnvString("STREAMING_SERVICE_URL", "http://localhost:8080"),
		PostgresURL:            getEnvString("POSTGRES_URL", "postgres://user:password@localhost:5432/plantos"),
		RedisURL:               getEnvString("REDIS_URL", "redis://localhost:6379"),
		PollingInterval:        getEnvDuration("POLLING_INTERVAL", 30*time.Second),
		MaxConcurrentPolls:     getEnvInt("MAX_CONCURRENT_POLLS", 50),
		DeviceTimeout:          getEnvDuration("DEVICE_TIMEOUT", 5*time.Second),
		RetryAttempts:          getEnvInt("RETRY_ATTEMPTS", 3),
		RetryDelay:             getEnvDuration("RETRY_DELAY", 1*time.Second),
		CircuitBreakerTimeout:  getEnvDuration("CIRCUIT_BREAKER_TIMEOUT", 60*time.Second),
		CircuitBreakerThreshold: getEnvInt("CIRCUIT_BREAKER_THRESHOLD", 10),
		MetricsEnabled:         getEnvBool("METRICS_ENABLED", true),
		LogLevel:               getEnvString("LOG_LEVEL", "info"),
		Environment:            getEnvString("ENVIRONMENT", "development"),
		ServiceName:            getEnvString("SERVICE_NAME", "plantos-device-manager"),
		ServiceVersion:         getEnvString("SERVICE_VERSION", "1.0.0"),
		HealthCheckPath:        getEnvString("HEALTH_CHECK_PATH", "/health"),
		MetricsPath:            getEnvString("METRICS_PATH", "/metrics"),
		ShutdownTimeout:        getEnvDuration("SHUTDOWN_TIMEOUT", 30*time.Second),
		HardwareInterface:      getEnvString("HARDWARE_INTERFACE", "mock"),
		MockHardware:           getEnvBool("MOCK_HARDWARE", true),
		DeviceDiscovery:        getEnvBool("DEVICE_DISCOVERY", true),
		WorkerPoolSize:         getEnvInt("WORKER_POOL_SIZE", 10),
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

	if c.PollingInterval <= 0 {
		return fmt.Errorf("polling interval must be positive: %v", c.PollingInterval)
	}

	if c.MaxConcurrentPolls <= 0 {
		return fmt.Errorf("max concurrent polls must be positive: %d", c.MaxConcurrentPolls)
	}

	if c.DeviceTimeout <= 0 {
		return fmt.Errorf("device timeout must be positive: %v", c.DeviceTimeout)
	}

	if c.RetryAttempts < 0 {
		return fmt.Errorf("retry attempts must be non-negative: %d", c.RetryAttempts)
	}

	if c.RetryDelay <= 0 {
		return fmt.Errorf("retry delay must be positive: %v", c.RetryDelay)
	}

	if c.WorkerPoolSize <= 0 {
		return fmt.Errorf("worker pool size must be positive: %d", c.WorkerPoolSize)
	}

	if c.StreamingServiceURL == "" {
		return fmt.Errorf("streaming service URL is required")
	}

	if c.PostgresURL == "" {
		return fmt.Errorf("postgres URL is required")
	}

	if c.RedisURL == "" {
		return fmt.Errorf("redis URL is required")
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