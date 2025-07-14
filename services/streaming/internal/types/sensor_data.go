package types

import (
	"fmt"
	"time"
)

// SensorData represents a sensor reading
type SensorData struct {
	ID          string    `json:"id"`
	PlantID     string    `json:"plant_id"`
	SensorID    string    `json:"sensor_id"`
	SensorType  string    `json:"sensor_type"`
	Value       float64   `json:"value"`
	Unit        string    `json:"unit"`
	Timestamp   time.Time `json:"timestamp"`
	Quality     string    `json:"quality"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	BatchID     string    `json:"batch_id,omitempty"`
	Source      string    `json:"source"`
	Environment string    `json:"environment"`
}

// SensorType constants
const (
	SensorTypeMoisture    = "moisture"
	SensorTypeTemperature = "temperature"
	SensorTypeHumidity    = "humidity"
	SensorTypeLight       = "light"
	SensorTypePH          = "ph"
	SensorTypeNutrients   = "nutrients"
	SensorTypeCamera      = "camera"
)

// Quality constants
const (
	QualityGood    = "good"
	QualityWarning = "warning"
	QualityError   = "error"
)

// Validate validates the sensor data
func (sd *SensorData) Validate() error {
	if sd.ID == "" {
		return fmt.Errorf("sensor data ID is required")
	}
	
	if sd.PlantID == "" {
		return fmt.Errorf("plant ID is required")
	}
	
	if sd.SensorID == "" {
		return fmt.Errorf("sensor ID is required")
	}
	
	if sd.SensorType == "" {
		return fmt.Errorf("sensor type is required")
	}
	
	if !isValidSensorType(sd.SensorType) {
		return fmt.Errorf("invalid sensor type: %s", sd.SensorType)
	}
	
	if sd.Unit == "" {
		return fmt.Errorf("unit is required")
	}
	
	if sd.Timestamp.IsZero() {
		return fmt.Errorf("timestamp is required")
	}
	
	if sd.Quality == "" {
		sd.Quality = QualityGood // Default to good quality
	}
	
	if !isValidQuality(sd.Quality) {
		return fmt.Errorf("invalid quality: %s", sd.Quality)
	}
	
	if sd.Source == "" {
		return fmt.Errorf("source is required")
	}
	
	// Validate value ranges based on sensor type
	if err := sd.validateValueRange(); err != nil {
		return err
	}
	
	return nil
}

// validateValueRange validates that the sensor value is within expected ranges
func (sd *SensorData) validateValueRange() error {
	switch sd.SensorType {
	case SensorTypeMoisture:
		if sd.Value < 0 || sd.Value > 100 {
			return fmt.Errorf("moisture value must be between 0 and 100, got %f", sd.Value)
		}
	case SensorTypeTemperature:
		if sd.Value < -40 || sd.Value > 80 {
			return fmt.Errorf("temperature value must be between -40 and 80, got %f", sd.Value)
		}
	case SensorTypeHumidity:
		if sd.Value < 0 || sd.Value > 100 {
			return fmt.Errorf("humidity value must be between 0 and 100, got %f", sd.Value)
		}
	case SensorTypeLight:
		if sd.Value < 0 || sd.Value > 100000 {
			return fmt.Errorf("light value must be between 0 and 100000, got %f", sd.Value)
		}
	case SensorTypePH:
		if sd.Value < 0 || sd.Value > 14 {
			return fmt.Errorf("pH value must be between 0 and 14, got %f", sd.Value)
		}
	}
	
	return nil
}

// isValidSensorType checks if the sensor type is valid
func isValidSensorType(sensorType string) bool {
	validTypes := []string{
		SensorTypeMoisture,
		SensorTypeTemperature,
		SensorTypeHumidity,
		SensorTypeLight,
		SensorTypePH,
		SensorTypeNutrients,
		SensorTypeCamera,
	}
	
	for _, validType := range validTypes {
		if sensorType == validType {
			return true
		}
	}
	
	return false
}

// isValidQuality checks if the quality is valid
func isValidQuality(quality string) bool {
	validQualities := []string{
		QualityGood,
		QualityWarning,
		QualityError,
	}
	
	for _, validQuality := range validQualities {
		if quality == validQuality {
			return true
		}
	}
	
	return false
}

// GetExpectedUnit returns the expected unit for a sensor type
func (sd *SensorData) GetExpectedUnit() string {
	switch sd.SensorType {
	case SensorTypeMoisture:
		return "%"
	case SensorTypeTemperature:
		return "°C"
	case SensorTypeHumidity:
		return "%"
	case SensorTypeLight:
		return "lux"
	case SensorTypePH:
		return "pH"
	case SensorTypeNutrients:
		return "ppm"
	case SensorTypeCamera:
		return "bytes"
	default:
		return ""
	}
}

// IsValidUnit checks if the unit is valid for this sensor type
func (sd *SensorData) IsValidUnit() bool {
	expectedUnit := sd.GetExpectedUnit()
	if expectedUnit == "" {
		return true // Unknown sensor type, accept any unit
	}
	return sd.Unit == expectedUnit
}

// ToMap converts SensorData to a map for easy serialization
func (sd *SensorData) ToMap() map[string]interface{} {
	result := map[string]interface{}{
		"id":          sd.ID,
		"plant_id":    sd.PlantID,
		"sensor_id":   sd.SensorID,
		"sensor_type": sd.SensorType,
		"value":       sd.Value,
		"unit":        sd.Unit,
		"timestamp":   sd.Timestamp,
		"quality":     sd.Quality,
		"source":      sd.Source,
		"environment": sd.Environment,
	}
	
	if sd.BatchID != "" {
		result["batch_id"] = sd.BatchID
	}
	
	if sd.Metadata != nil {
		result["metadata"] = sd.Metadata
	}
	
	return result
}

// StreamingMetadata represents metadata for streaming operations
type StreamingMetadata struct {
	ProcessedAt time.Time `json:"processed_at"`
	BatchID     string    `json:"batch_id"`
	RetryCount  int       `json:"retry_count"`
	Source      string    `json:"source"`
	Version     string    `json:"version"`
}

// AggregatedData represents aggregated sensor data
type AggregatedData struct {
	PlantID     string    `json:"plant_id"`
	SensorType  string    `json:"sensor_type"`
	MinValue    float64   `json:"min_value"`
	MaxValue    float64   `json:"max_value"`
	AvgValue    float64   `json:"avg_value"`
	Count       int       `json:"count"`
	StartTime   time.Time `json:"start_time"`
	EndTime     time.Time `json:"end_time"`
	Quality     string    `json:"quality"`
}

// StreamingStats represents streaming service statistics
type StreamingStats struct {
	ProcessedCount    int64     `json:"processed_count"`
	ErrorCount        int64     `json:"error_count"`
	LastProcessedAt   time.Time `json:"last_processed_at"`
	BufferSize        int       `json:"buffer_size"`
	ActiveSubscribers int       `json:"active_subscribers"`
	Uptime            time.Duration `json:"uptime"`
	AverageLatency    time.Duration `json:"average_latency"`
}