# PlantOS Monitoring & Observability

## Overview
PlantOS uses Prometheus for metrics collection and Grafana for visualization, providing real-time insights into plant health, system performance, and hardware status.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   PlantOS App   │────▶│  Prometheus  │────▶│   Grafana   │
│  (Metrics Port) │     │   (Scraper)  │     │ (Dashboards)│
└─────────────────┘     └──────────────┘     └─────────────┘
         │                      │                     │
         │              ┌──────────────┐              │
         └─────────────▶│ Alert Manager│──────────────┘
                        └──────────────┘
```

## Metrics Strategy

### Application Metrics

```python
# src/infrastructure/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client import CollectorRegistry, generate_latest
from contextlib import contextmanager
import time

# Create custom registry
registry = CollectorRegistry()

# System Info
system_info = Info(
    'plantos_system_info',
    'PlantOS system information',
    registry=registry
)

# Plant Metrics
plant_count = Gauge(
    'plantos_plant_count',
    'Total number of plants in system',
    ['status', 'location'],
    registry=registry
)

watering_events = Counter(
    'plantos_watering_events_total',
    'Total watering events',
    ['plant_id', 'trigger_type', 'status'],
    registry=registry
)

water_dispensed = Counter(
    'plantos_water_dispensed_ml_total',
    'Total water dispensed in milliliters',
    ['plant_id', 'location'],
    registry=registry
)

# Sensor Metrics
sensor_readings = Gauge(
    'plantos_sensor_reading',
    'Current sensor reading value',
    ['sensor_id', 'sensor_type', 'plant_id', 'unit'],
    registry=registry
)

sensor_read_duration = Histogram(
    'plantos_sensor_read_duration_seconds',
    'Time to read sensor value',
    ['sensor_type', 'status'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry
)

sensor_errors = Counter(
    'plantos_sensor_errors_total',
    'Total sensor read errors',
    ['sensor_type', 'error_type'],
    registry=registry
)

# Hardware Metrics
pump_activations = Counter(
    'plantos_pump_activations_total',
    'Total pump activations',
    ['pump_id', 'reason'],
    registry=registry
)

pump_runtime = Counter(
    'plantos_pump_runtime_seconds_total',
    'Total pump runtime in seconds',
    ['pump_id'],
    registry=registry
)

# API Metrics
http_requests = Counter(
    'plantos_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

http_request_duration = Histogram(
    'plantos_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=registry
)

# Database Metrics
db_connections = Gauge(
    'plantos_db_connections',
    'Current database connections',
    ['state'],
    registry=registry
)

db_query_duration = Histogram(
    'plantos_db_query_duration_seconds',
    'Database query duration',
    ['operation', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry
)

# Business Metrics
plant_health_score = Gauge(
    'plantos_plant_health_score',
    'Plant health score (0-100)',
    ['plant_id', 'plant_name'],
    registry=registry
)

care_recommendations = Counter(
    'plantos_care_recommendations_total',
    'Care recommendations generated',
    ['plant_id', 'recommendation_type'],
    registry=registry
)

# Helper decorators
@contextmanager
def track_duration(metric, **labels):
    """Track duration of an operation."""
    start = time.time()
    try:
        yield
        labels['status'] = 'success'
    except Exception as e:
        labels['status'] = 'error'
        raise
    finally:
        duration = time.time() - start
        metric.labels(**labels).observe(duration)

def track_request(func):
    """Decorator to track HTTP requests."""
    async def wrapper(request, *args, **kwargs):
        method = request.method
        endpoint = request.url.path
        
        start = time.time()
        try:
            response = await func(request, *args, **kwargs)
            status = response.status_code
        except Exception as e:
            status = 500
            raise
        finally:
            duration = time.time() - start
            http_requests.labels(
                method=method,
                endpoint=endpoint,
                status_code=status
            ).inc()
            http_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
        
        return response
    
    return wrapper
```

### Metrics Implementation

```python
# src/infrastructure/monitoring/collectors.py
from prometheus_client import Gauge
from typing import Dict, Any
import asyncio
import logging

logger = logging.getLogger(__name__)

class PlantMetricsCollector:
    """Collects and updates plant-related metrics."""
    
    def __init__(self, plant_repository, metrics_registry):
        self.plant_repo = plant_repository
        self.registry = metrics_registry
        
    async def collect_metrics(self):
        """Collect current plant metrics."""
        try:
            # Count plants by status
            plants = await self.plant_repo.get_all()
            
            status_counts = {}
            for plant in plants:
                status = plant.health_status
                location = plant.location or "unknown"
                key = (status, location)
                status_counts[key] = status_counts.get(key, 0) + 1
            
            # Update metrics
            for (status, location), count in status_counts.items():
                plant_count.labels(
                    status=status,
                    location=location
                ).set(count)
                
            # Update health scores
            for plant in plants:
                if plant.health_score is not None:
                    plant_health_score.labels(
                        plant_id=plant.id,
                        plant_name=plant.name
                    ).set(plant.health_score)
                    
        except Exception as e:
            logger.error(f"Error collecting plant metrics: {e}")

class SensorMetricsCollector:
    """Collects sensor reading metrics."""
    
    def __init__(self, sensor_manager, metrics_registry):
        self.sensor_manager = sensor_manager
        self.registry = metrics_registry
    
    async def update_sensor_reading(
        self,
        sensor_id: str,
        sensor_type: str,
        plant_id: str,
        value: float,
        unit: str
    ):
        """Update metric for a sensor reading."""
        sensor_readings.labels(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            plant_id=plant_id,
            unit=unit
        ).set(value)
    
    async def track_sensor_read(self, sensor_type: str):
        """Context manager to track sensor read duration."""
        return track_duration(
            sensor_read_duration,
            sensor_type=sensor_type
        )

# Metrics endpoint for Prometheus
from fastapi import FastAPI, Response
from prometheus_client import generate_latest

app = FastAPI()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(registry),
        media_type="text/plain"
    )
```

### Hardware Metrics Export (C)

```c
// src/infrastructure/hardware/metrics_exporter.c
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "metrics_exporter.h"

typedef struct {
    char name[64];
    char labels[256];
    double value;
    metric_type_t type;
} metric_t;

static metric_t metrics[MAX_METRICS];
static int metric_count = 0;

int metrics_export_gauge(
    const char* name,
    const char* labels,
    double value
) {
    if (metric_count >= MAX_METRICS) {
        return -1;
    }
    
    metric_t* m = &metrics[metric_count++];
    strncpy(m->name, name, sizeof(m->name) - 1);
    strncpy(m->labels, labels, sizeof(m->labels) - 1);
    m->value = value;
    m->type = METRIC_GAUGE;
    
    return 0;
}

int metrics_export_counter_inc(
    const char* name,
    const char* labels,
    double increment
) {
    // Find existing metric or create new
    for (int i = 0; i < metric_count; i++) {
        if (strcmp(metrics[i].name, name) == 0 &&
            strcmp(metrics[i].labels, labels) == 0) {
            metrics[i].value += increment;
            return 0;
        }
    }
    
    // Create new counter
    if (metric_count >= MAX_METRICS) {
        return -1;
    }
    
    metric_t* m = &metrics[metric_count++];
    strncpy(m->name, name, sizeof(m->name) - 1);
    strncpy(m->labels, labels, sizeof(m->labels) - 1);
    m->value = increment;
    m->type = METRIC_COUNTER;
    
    return 0;
}

void metrics_format_prometheus(char* buffer, size_t buffer_size) {
    char* ptr = buffer;
    size_t remaining = buffer_size;
    
    for (int i = 0; i < metric_count; i++) {
        metric_t* m = &metrics[i];
        
        int written = snprintf(
            ptr, remaining,
            "%s{%s} %.4f\n",
            m->name, m->labels, m->value
        );
        
        if (written > 0 && written < remaining) {
            ptr += written;
            remaining -= written;
        }
    }
}

// Python binding
static PyObject* py_get_hardware_metrics(PyObject* self, PyObject* args) {
    char buffer[4096];
    metrics_format_prometheus(buffer, sizeof(buffer));
    return PyUnicode_FromString(buffer);
}
```

## Grafana Dashboards

### Main Dashboard Configuration

```yaml
# docker/grafana/dashboards/plantos-overview.yaml
apiVersion: 1
providers:
  - name: 'PlantOS'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /var/lib/grafana/dashboards
```

### Plant Health Dashboard

```json
{
  "dashboard": {
    "title": "PlantOS - Plant Health Overview",
    "uid": "plantos-health",
    "timezone": "browser",
    "panels": [
      {
        "title": "Plant Health Scores",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "plantos_plant_health_score",
            "legendFormat": "{{plant_name}}",
            "refId": "A"
          }
        ],
        "yaxes": [
          {
            "format": "percent",
            "max": 100,
            "min": 0
          }
        ]
      },
      {
        "title": "Plants by Status",
        "type": "piechart",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum by (status) (plantos_plant_count)",
            "legendFormat": "{{status}}",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Soil Moisture Levels",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "plantos_sensor_reading{sensor_type=\"moisture\"}",
            "format": "heatmap",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Watering Events (24h)",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16},
        "targets": [
          {
            "expr": "increase(plantos_watering_events_total[24h])",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Water Used Today",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16},
        "targets": [
          {
            "expr": "increase(plantos_water_dispensed_ml_total[24h])",
            "format": "time_series",
            "refId": "A"
          }
        ],
        "options": {
          "unit": "ml"
        }
      }
    ]
  }
}
```

### System Performance Dashboard

```json
{
  "dashboard": {
    "title": "PlantOS - System Performance",
    "uid": "plantos-performance",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(plantos_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.50, rate(plantos_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p50",
            "refId": "B"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum by (endpoint) (rate(plantos_http_requests_total[5m]))",
            "legendFormat": "{{endpoint}}",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Sensor Read Performance",
        "type": "table",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(plantos_sensor_read_duration_seconds_bucket[5m])) by (sensor_type)",
            "format": "table",
            "instant": true,
            "refId": "A"
          }
        ]
      },
      {
        "title": "Database Query Time",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(plantos_db_query_duration_seconds_bucket[5m])) by (operation)",
            "legendFormat": "{{operation}}",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

## Alerting Rules

### Prometheus Alert Configuration

```yaml
# docker/prometheus/alerts/plantos.rules.yml
groups:
  - name: plantos_plant_health
    interval: 30s
    rules:
      - alert: PlantCriticallyDry
        expr: plantos_sensor_reading{sensor_type="moisture"} < 20
        for: 30m
        labels:
          severity: critical
          category: plant_health
        annotations:
          summary: "Plant {{ $labels.plant_id }} is critically dry"
          description: "Moisture level is {{ $value }}%, immediate watering required"
      
      - alert: PlantHealthDegraded
        expr: plantos_plant_health_score < 50
        for: 1h
        labels:
          severity: warning
          category: plant_health
        annotations:
          summary: "Plant {{ $labels.plant_name }} health is degraded"
          description: "Health score is {{ $value }}%"
      
      - alert: NoRecentWatering
        expr: time() - plantos_last_watering_timestamp > 86400 * 3
        for: 1h
        labels:
          severity: warning
          category: maintenance
        annotations:
          summary: "Plant {{ $labels.plant_id }} hasn't been watered in 3 days"

  - name: plantos_system
    interval: 10s
    rules:
      - alert: SensorReadFailure
        expr: rate(plantos_sensor_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
          category: hardware
        annotations:
          summary: "Sensor {{ $labels.sensor_type }} experiencing failures"
          description: "Error rate: {{ $value }} errors/sec"
      
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(plantos_http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "API endpoint {{ $labels.endpoint }} has high latency"
          description: "95th percentile latency is {{ $value }}s"
      
      - alert: DatabaseConnectionPoolExhausted
        expr: plantos_db_connections{state="active"} / plantos_db_connections{state="max"} > 0.9
        for: 5m
        labels:
          severity: critical
          category: database
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "{{ $value }}% of connections in use"

  - name: plantos_hardware
    interval: 10s
    rules:
      - alert: PumpRunningTooLong
        expr: rate(plantos_pump_runtime_seconds_total[5m]) > 30
        for: 2m
        labels:
          severity: critical
          category: hardware
        annotations:
          summary: "Pump {{ $labels.pump_id }} running continuously"
          description: "Pump has been active for over 2 minutes"
      
      - alert: WaterReservoirLow
        expr: plantos_water_reservoir_level_percent < 10
        for: 5m
        labels:
          severity: warning
          category: maintenance
        annotations:
          summary: "Water reservoir is low"
          description: "Only {{ $value }}% remaining"
```

### AlertManager Configuration

```yaml
# docker/alertmanager/alertmanager.yml
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'plantos-alerts@example.com'
  smtp_auth_username: 'plantos-alerts@example.com'
  smtp_auth_password: 'app-specific-password'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical'
      repeat_interval: 1h
    
    - match:
        category: plant_health
      receiver: 'plant-health'
      repeat_interval: 4h

receivers:
  - name: 'default'
    email_configs:
      - to: 'admin@example.com'
        headers:
          Subject: 'PlantOS Alert: {{ .GroupLabels.alertname }}'

  - name: 'critical'
    email_configs:
      - to: 'admin@example.com'
        headers:
          Subject: 'CRITICAL PlantOS Alert: {{ .GroupLabels.alertname }}'
    webhook_configs:
      - url: 'http://plantos-api:8000/webhooks/alerts'
        send_resolved: true

  - name: 'plant-health'
    webhook_configs:
      - url: 'http://plantos-api:8000/webhooks/plant-health-alerts'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
```

## Docker Compose Setup

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: plantos-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus:/etc/prometheus
      - prometheus-data:/prometheus
    networks:
      - plantos-network

  grafana:
    image: grafana/grafana:latest
    container_name: plantos-grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=plantos123
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3000:3000"
    volumes:
      - ./docker/grafana/provisioning:/etc/grafana/provisioning
      - ./docker/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana-data:/var/lib/grafana
    networks:
      - plantos-network
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager:latest
    container_name: plantos-alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    ports:
      - "9093:9093"
    volumes:
      - ./docker/alertmanager:/etc/alertmanager
      - alertmanager-data:/alertmanager
    networks:
      - plantos-network

  node-exporter:
    image: prom/node-exporter:latest
    container_name: plantos-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - plantos-network

volumes:
  prometheus-data:
  grafana-data:
  alertmanager-data:

networks:
  plantos-network:
    external: true
```

## Prometheus Configuration

```yaml
# docker/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'plantos-monitor'

rule_files:
  - '/etc/prometheus/alerts/*.rules.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  - job_name: 'plantos-api'
    static_configs:
      - targets: ['plantos-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'plantos-hardware'
    static_configs:
      - targets: ['plantos-hardware:9101']
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

## Custom Metrics Dashboard

```python
# src/application/api/dashboard.py
from fastapi import APIRouter, Response
from prometheus_client import generate_latest
import json

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/metrics/plant/{plant_id}")
async def get_plant_metrics(plant_id: str):
    """Get detailed metrics for a specific plant."""
    metrics = {
        "current_moisture": sensor_readings.labels(
            sensor_type="moisture",
            plant_id=plant_id
        )._value.get(),
        "health_score": plant_health_score.labels(
            plant_id=plant_id
        )._value.get(),
        "watering_events_24h": watering_events.labels(
            plant_id=plant_id
        )._value.get(),
        "last_watered": await get_last_watered_time(plant_id),
        "recommendations": await get_active_recommendations(plant_id)
    }
    
    return metrics

@router.get("/export/prometheus")
async def export_prometheus_metrics():
    """Export all metrics in Prometheus format."""
    return Response(
        content=generate_latest(registry),
        media_type="text/plain"
    )
```