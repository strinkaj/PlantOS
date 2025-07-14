package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"plantos/streaming/internal/streaming"
	"plantos/streaming/internal/types"
)

// Handler handles HTTP requests for the streaming service
type Handler struct {
	streaming *streaming.Service
}

// NewHandler creates a new HTTP handler
func NewHandler(streamingService *streaming.Service) *Handler {
	return &Handler{
		streaming: streamingService,
	}
}

// Health handles health check requests
func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	response := map[string]interface{}{
		"status":    "healthy",
		"service":   "plantos-streaming",
		"timestamp": time.Now().UTC(),
		"version":   "1.0.0",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Metrics handles metrics requests
func (h *Handler) Metrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// TODO: Implement Prometheus metrics export
	w.Header().Set("Content-Type", "text/plain")
	w.Write([]byte("# Prometheus metrics will be implemented here\n"))
}

// IngestSensorData handles sensor data ingestion
func (h *Handler) IngestSensorData(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var sensorData types.SensorData
	if err := json.NewDecoder(r.Body).Decode(&sensorData); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Set default timestamp if not provided
	if sensorData.Timestamp.IsZero() {
		sensorData.Timestamp = time.Now().UTC()
	}

	// Set source if not provided
	if sensorData.Source == "" {
		sensorData.Source = "http-api"
	}

	// Set environment if not provided
	if sensorData.Environment == "" {
		sensorData.Environment = "production"
	}

	// Ingest the data
	if err := h.streaming.IngestSensorData(sensorData); err != nil {
		if err == streaming.ErrBufferFull {
			http.Error(w, "Service temporarily unavailable", http.StatusServiceUnavailable)
		} else {
			http.Error(w, fmt.Sprintf("Failed to ingest data: %v", err), http.StatusInternalServerError)
		}
		return
	}

	response := map[string]interface{}{
		"status":    "accepted",
		"timestamp": time.Now().UTC(),
		"data_id":   sensorData.ID,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(response)
}

// StreamSubscribe handles WebSocket subscriptions for real-time data
func (h *Handler) StreamSubscribe(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// For now, implement a simple long-polling mechanism
	// In a real implementation, this would use WebSockets
	subscriberID := r.URL.Query().Get("subscriber_id")
	if subscriberID == "" {
		http.Error(w, "subscriber_id is required", http.StatusBadRequest)
		return
	}

	// Subscribe to the stream
	ch := h.streaming.Subscribe(subscriberID)
	defer h.streaming.Unsubscribe(subscriberID)

	// Set headers for Server-Sent Events
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Send initial connection message
	fmt.Fprintf(w, "data: {\"type\":\"connected\",\"subscriber_id\":\"%s\"}\n\n", subscriberID)
	
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	// Stream data
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case data, ok := <-ch:
			if !ok {
				return // Channel closed
			}

			jsonData, err := json.Marshal(data)
			if err != nil {
				continue
			}

			fmt.Fprintf(w, "data: %s\n\n", jsonData)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}

		case <-ticker.C:
			// Send keepalive
			fmt.Fprintf(w, "data: {\"type\":\"keepalive\",\"timestamp\":\"%s\"}\n\n", time.Now().UTC().Format(time.RFC3339))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}

		case <-r.Context().Done():
			return
		}
	}
}

// BatchIngestSensorData handles batch sensor data ingestion
func (h *Handler) BatchIngestSensorData(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var sensorDataBatch []types.SensorData
	if err := json.NewDecoder(r.Body).Decode(&sensorDataBatch); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	if len(sensorDataBatch) == 0 {
		http.Error(w, "Empty batch", http.StatusBadRequest)
		return
	}

	if len(sensorDataBatch) > 1000 {
		http.Error(w, "Batch too large (max 1000 items)", http.StatusBadRequest)
		return
	}

	batchID := fmt.Sprintf("batch-%d", time.Now().UnixNano())
	successCount := 0
	errorCount := 0
	errors := make([]string, 0)

	for i, sensorData := range sensorDataBatch {
		// Set batch ID
		sensorData.BatchID = batchID

		// Set default timestamp if not provided
		if sensorData.Timestamp.IsZero() {
			sensorData.Timestamp = time.Now().UTC()
		}

		// Set source if not provided
		if sensorData.Source == "" {
			sensorData.Source = "http-batch-api"
		}

		// Set environment if not provided
		if sensorData.Environment == "" {
			sensorData.Environment = "production"
		}

		// Ingest the data
		if err := h.streaming.IngestSensorData(sensorData); err != nil {
			errorCount++
			errors = append(errors, fmt.Sprintf("Item %d: %v", i, err))
		} else {
			successCount++
		}
	}

	response := map[string]interface{}{
		"status":       "processed",
		"batch_id":     batchID,
		"total_items":  len(sensorDataBatch),
		"success_count": successCount,
		"error_count":  errorCount,
		"timestamp":    time.Now().UTC(),
	}

	if len(errors) > 0 {
		response["errors"] = errors
	}

	w.Header().Set("Content-Type", "application/json")
	
	if errorCount > 0 {
		w.WriteHeader(http.StatusPartialContent)
	} else {
		w.WriteHeader(http.StatusAccepted)
	}
	
	json.NewEncoder(w).Encode(response)
}

// GetStreamingStats returns streaming service statistics
func (h *Handler) GetStreamingStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// TODO: Implement actual statistics collection
	stats := types.StreamingStats{
		ProcessedCount:    12345,
		ErrorCount:        23,
		LastProcessedAt:   time.Now().UTC(),
		BufferSize:        100,
		ActiveSubscribers: 5,
		Uptime:            time.Hour * 24,
		AverageLatency:    time.Millisecond * 50,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}