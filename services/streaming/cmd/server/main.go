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

	"plantos/streaming/internal/config"
	"plantos/streaming/internal/handlers"
	"plantos/streaming/internal/streaming"
)

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Create streaming service
	streamingService := streaming.NewService(cfg)

	// Create HTTP handlers
	handler := handlers.NewHandler(streamingService)

	// Setup HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/health", handler.Health)
	mux.HandleFunc("/metrics", handler.Metrics)
	mux.HandleFunc("/api/v1/sensor/data", handler.IngestSensorData)
	mux.HandleFunc("/api/v1/stream/subscribe", handler.StreamSubscribe)

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.Port),
		Handler: mux,
	}

	// Start streaming service
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := streamingService.Start(ctx); err != nil {
			log.Printf("Streaming service error: %v", err)
		}
	}()

	// Start HTTP server
	go func() {
		log.Printf("Streaming service listening on port %d", cfg.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down streaming service...")

	// Shutdown with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}

	cancel() // Cancel streaming service context
	log.Println("Streaming service shutdown complete")
}