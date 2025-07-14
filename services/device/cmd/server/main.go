package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"plantos/device/internal/config"
	"plantos/device/internal/device"
	"plantos/device/internal/handlers"
)

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Create device manager
	deviceManager := device.NewManager(cfg)

	// Create HTTP handlers
	handler := handlers.NewHandler(deviceManager)

	// Setup HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/health", handler.Health)
	mux.HandleFunc("/metrics", handler.Metrics)
	mux.HandleFunc("/api/v1/devices", handler.ListDevices)
	mux.HandleFunc("/api/v1/devices/register", handler.RegisterDevice)
	mux.HandleFunc("/api/v1/devices/", handler.DeviceDetails) // Handle /devices/{id}
	mux.HandleFunc("/api/v1/plants/", handler.PlantDevices)  // Handle /plants/{id}/devices
	mux.HandleFunc("/api/v1/polling/start", handler.StartPolling)
	mux.HandleFunc("/api/v1/polling/stop", handler.StopPolling)
	mux.HandleFunc("/api/v1/polling/status", handler.PollingStatus)

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.Port),
		Handler: mux,
	}

	// Start device manager
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := deviceManager.Start(ctx); err != nil {
			log.Printf("Device manager error: %v", err)
		}
	}()

	// Start HTTP server
	go func() {
		log.Printf("Device manager listening on port %d", cfg.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down device manager...")

	// Shutdown with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}

	cancel() // Cancel device manager context
	log.Println("Device manager shutdown complete")
}