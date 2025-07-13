# PlantOS

## Overview
PlantOS is a production-grade automated plant care system demonstrating enterprise-level software engineering practices with a polyglot architecture optimized for IoT, real-time processing, and data engineering workflows.

## Technology Stack

### Core Languages
- **Python 3.12+**: API services, business logic, ML/AI plant health models, analytics
- **Go**: High-performance sensor data streaming, concurrent device management
- **Rust**: Complete hardware control, memory-safe sensor and actuator drivers
- **SQL**: Complex time-series queries with PostgreSQL/TimescaleDB

### Data Engineering Stack (Free for Personal Use)
- **Apache Spark/PySpark**: Historical data analysis, batch plant health reports (Apache 2.0)
- **Apache Kafka**: Real-time sensor event streaming (Apache 2.0)
  - *Alternative*: Redpanda CE (BSL - free personal, paid business)
- **Apache Flink**: Complex event processing for anomaly detection (Apache 2.0)
- **dbt Core**: Data transformation pipelines for analytics (Apache 2.0)
  - *Note*: dbt Cloud requires subscription for business use
- **Apache Iceberg**: Data lakehouse for long-term sensor data storage (Apache 2.0)
- **DuckDB**: In-process OLAP for edge analytics (MIT)

### Infrastructure & DevOps
- **OpenTofu**: Infrastructure as code (MPL 2.0 - free for all)
  - *Alternative*: Terraform (BSL - check license for business use)
- **K3s**: Lightweight Kubernetes for edge/IoT (Apache 2.0)
  - *Alternative*: MicroK8s (Apache 2.0)
- **ArgoCD**: GitOps continuous deployment (Apache 2.0)
- **Ansible**: Configuration management for edge devices (GPL)

### Monitoring & Observability
- **Prometheus**: Metrics collection (Apache 2.0)
- **Grafana OSS**: Dashboards and visualization (AGPL v3)
  - *Note*: Grafana Cloud requires subscription
- **OpenTelemetry**: Distributed tracing (Apache 2.0)
- **Vector**: High-performance log aggregation (MPL 2.0)
- **Apache Superset**: Self-service analytics dashboards (Apache 2.0)

### Configuration & Schema
- **YAML**: Application configuration
- **Protocol Buffers**: Service-to-service communication (BSD)
- **Apache Avro**: Schema evolution for event streams (Apache 2.0)
- **JSON Schema**: API validation (MIT)

## Licensing Summary

This project uses only software that is **free for personal use**. All core components use permissive licenses (MIT, Apache 2.0, BSD) that are also free for commercial use.

### Components Requiring Attention for Business Use:
1. **Redpanda Community Edition**: Business Source License (BSL) - requires license for commercial use
   - **Solution**: Use Apache Kafka instead (Apache 2.0)
2. **Terraform**: Recently moved to BSL - check licensing for your use case
   - **Solution**: Use OpenTofu instead (MPL 2.0)
3. **Docker Desktop**: Free for personal use, requires subscription for businesses >250 employees
   - **Solution**: Use Podman, containerd, or K3s builtin container runtime
4. **dbt Cloud**: SaaS platform requires subscription
   - **Solution**: Use dbt Core with your own orchestration (Apache 2.0)
5. **Grafana Cloud/Enterprise**: Requires subscription
   - **Solution**: Use self-hosted Grafana OSS (AGPL v3)

### Always Free Alternatives:
- All Apache Software Foundation projects (Kafka, Spark, Flink, Iceberg, Superset)
- All programming languages (Python, Go, Rust, C)
- PostgreSQL and TimescaleDB
- Prometheus, OpenTelemetry, Vector
- K3s, MicroK8s, ArgoCD
- DuckDB, FastAPI, pytest