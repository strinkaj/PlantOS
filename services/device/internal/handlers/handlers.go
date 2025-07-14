package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"plantos/device/internal/device"
	"plantos/device/internal/types"
)

// Handler handles HTTP requests for the device manager service
type Handler struct {
	deviceManager *device.Manager
}

// NewHandler creates a new HTTP handler
func NewHandler(deviceManager *device.Manager) *Handler {
	return &Handler{
		deviceManager: deviceManager,
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
		"service":   "plantos-device-manager",
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

// ListDevices handles requests to list all devices
func (h *Handler) ListDevices(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	devices := h.deviceManager.ListDevices()

	response := map[string]interface{}{
		"devices":   devices,
		"count":     len(devices),
		"timestamp": time.Now().UTC(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RegisterDevice handles device registration requests
func (h *Handler) RegisterDevice(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var device types.Device
	if err := json.NewDecoder(r.Body).Decode(&device); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Generate ID if not provided
	if device.ID == "" {
		device.ID = fmt.Sprintf("%s-%d", device.Type, time.Now().Unix())
	}

	if err := h.deviceManager.RegisterDevice(&device); err != nil {
		http.Error(w, fmt.Sprintf("Failed to register device: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"status":    "registered",
		"device_id": device.ID,
		"timestamp": time.Now().UTC(),
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// DeviceDetails handles requests for specific device information
func (h *Handler) DeviceDetails(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract device ID from path /api/v1/devices/{id}
	path := r.URL.Path
	parts := strings.Split(path, "/")
	if len(parts) < 5 {
		http.Error(w, "Device ID is required", http.StatusBadRequest)
		return
	}

	deviceID := parts[4]
	device, err := h.deviceManager.GetDevice(deviceID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Device not found: %v", err), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(device)
}

// PlantDevices handles requests for devices associated with a specific plant
func (h *Handler) PlantDevices(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract plant ID from path /api/v1/plants/{id}/devices
	path := r.URL.Path
	parts := strings.Split(path, "/")
	if len(parts) < 5 {
		http.Error(w, "Plant ID is required", http.StatusBadRequest)
		return
	}

	plantID := parts[4]
	devices := h.deviceManager.GetDevicesByPlant(plantID)

	response := map[string]interface{}{
		"plant_id":  plantID,
		"devices":   devices,
		"count":     len(devices),
		"timestamp": time.Now().UTC(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// StartPolling handles requests to start device polling
func (h *Handler) StartPolling(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if err := h.deviceManager.StartPolling(); err != nil {
		http.Error(w, fmt.Sprintf("Failed to start polling: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"status":    "polling_started",
		"timestamp": time.Now().UTC(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// StopPolling handles requests to stop device polling
func (h *Handler) StopPolling(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if err := h.deviceManager.StopPolling(); err != nil {
		http.Error(w, fmt.Sprintf("Failed to stop polling: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"status":    "polling_stopped",
		"timestamp": time.Now().UTC(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// PollingStatus handles requests for polling status
func (h *Handler) PollingStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	status := h.deviceManager.GetPollingStatus()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// GetDeviceStats returns device manager statistics
func (h *Handler) GetDeviceStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	stats := h.deviceManager.GetStats()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// ExecuteDeviceCommand handles device command execution
func (h *Handler) ExecuteDeviceCommand(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract device ID from path
	path := r.URL.Path
	parts := strings.Split(path, "/")
	if len(parts) < 6 {
		http.Error(w, "Device ID is required", http.StatusBadRequest)
		return
	}

	deviceID := parts[4]

	var command types.DeviceCommand
	if err := json.NewDecoder(r.Body).Decode(&command); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Verify device exists
	_, err := h.deviceManager.GetDevice(deviceID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Device not found: %v", err), http.StatusNotFound)
		return
	}

	// Set command metadata
	command.ID = fmt.Sprintf("cmd-%d", time.Now().UnixNano())
	command.DeviceID = deviceID
	command.Timestamp = time.Now().UTC()
	command.ExpiresAt = time.Now().UTC().Add(5 * time.Minute)
	command.Status = "queued"

	// TODO: Implement actual command execution
	// For now, just return success
	command.Status = "completed"

	response := map[string]interface{}{
		"command_id": command.ID,
		"status":     command.Status,
		"timestamp":  time.Now().UTC(),
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(response)
}

// UpdateDevice handles device updates
func (h *Handler) UpdateDevice(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract device ID from path
	path := r.URL.Path
	parts := strings.Split(path, "/")
	if len(parts) < 5 {
		http.Error(w, "Device ID is required", http.StatusBadRequest)
		return
	}

	deviceID := parts[4]

	// Get existing device
	existingDevice, err := h.deviceManager.GetDevice(deviceID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Device not found: %v", err), http.StatusNotFound)
		return
	}

	var updateData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&updateData); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Update allowed fields
	if name, ok := updateData["name"].(string); ok {
		existingDevice.Name = name
	}
	if plantID, ok := updateData["plant_id"].(string); ok {
		existingDevice.PlantID = plantID
	}
	if location, ok := updateData["location"].(string); ok {
		existingDevice.Location = location
	}
	if tags, ok := updateData["tags"].([]interface{}); ok {
		stringTags := make([]string, len(tags))
		for i, tag := range tags {
			if str, ok := tag.(string); ok {
				stringTags[i] = str
			}
		}
		existingDevice.Tags = stringTags
	}

	existingDevice.UpdatedAt = time.Now().UTC()

	// TODO: Save updated device to database

	response := map[string]interface{}{
		"status":    "updated",
		"device_id": deviceID,
		"timestamp": time.Now().UTC(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}