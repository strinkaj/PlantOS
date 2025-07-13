# PlantOS - Getting Started

## Quick Environment Verification

```bash
# Test the current environment setup
./scripts/development/verify-environment.sh
```

## Development Environment Setup

### Option 1: Automated Setup (Recommended)
```bash
# Run the automated setup script
./scripts/development/setup-dev-environment.sh
```

### Option 2: Manual Setup

1. **Install Container Runtime**:
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install -y podman podman-compose python3-venv
   
   # macOS
   brew install podman podman-compose
   ```

2. **Setup Python Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   pre-commit install
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env file as needed
   ```

4. **Start Development Services**:
   ```bash
   podman-compose up -d
   ```

## Verify Installation

### Check Services are Running
```bash
# Check all services
podman-compose ps

# Test individual services
curl http://localhost:9090    # Prometheus
curl http://localhost:3000    # Grafana (admin/plantos123)
curl http://localhost:9000    # MinIO API
```

### Test Python Environment
```bash
source venv/bin/activate

# Test imports
python3 -c "
from src.shared.types import PlantID, SensorID
from src.infrastructure.hardware.interfaces import SensorType
print('✅ Python environment working')
"

# Run tests (when available)
pytest

# Check code quality
ruff check src/
mypy src/
```

## Development Workflow

### Daily Development
```bash
# Start services
podman-compose up -d

# Activate Python environment
source venv/bin/activate

# Start API server (when ready)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Service Access
- **Grafana Dashboard**: http://localhost:3000 (admin/plantos123)
- **Prometheus Metrics**: http://localhost:9090
- **MinIO Console**: http://localhost:9001 (plantos_admin/plantos_dev_password)
- **Spark Master UI**: http://localhost:8080
- **PostgreSQL**: localhost:5432 (plantos_user/plantos_dev_password)
- **Redis**: localhost:6379
- **Kafka**: localhost:9092

### Common Commands
```bash
# Stop all services
podman-compose down

# View service logs
podman-compose logs -f postgres
podman-compose logs -f redis

# Restart specific service
podman-compose restart postgres

# Run database migrations (when available)
alembic upgrade head

# Format code
black src/
ruff format src/

# Security scan
bandit -r src/
safety check
```

## Project Structure

```
PlantOS/
├── src/                          # Python application code
│   ├── core/                     # Business logic layer
│   │   ├── domain/               # Entities, value objects
│   │   └── use_cases/            # Application logic
│   ├── infrastructure/           # External interfaces
│   │   ├── database/             # Repository implementations
│   │   └── hardware/             # Hardware abstraction
│   ├── application/              # API, WebSocket, scheduling
│   └── shared/                   # Common utilities, types
├── services/                     # Go microservices
│   ├── streaming/                # Real-time data pipeline
│   └── device/                   # Device management
├── hardware/drivers/             # Rust hardware drivers
├── analytics/                    # Data processing
│   ├── spark/                    # Batch processing (PySpark)
│   ├── flink/                    # Stream processing
│   └── julia/                    # Scientific computing
├── scripts/                      # DevOps automation (Bash)
├── monitoring/                   # Prometheus, Grafana configs
├── docker/                       # Container configurations
└── docs/                         # Documentation
```

## Development Standards

### Type Safety
- **Python**: >95% mypy coverage, Pydantic validation
- **Go**: Interface-based design, comprehensive error handling
- **Rust**: Result<T,E> patterns, memory safety
- **Julia**: Type annotations, multiple dispatch

### Testing
```bash
# Run all tests
pytest --cov=src --cov-report=html

# Run specific test types
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests
pytest -m hardware       # Hardware tests (requires real sensors)
```

### Code Quality
```bash
# Full quality check
ruff check src/ && mypy src/ && bandit -r src/ && safety check
```

## Troubleshooting

### Container Issues
```bash
# Check container runtime
podman --version
podman-compose --version

# Check running containers
podman ps

# View container logs
podman logs plantos_postgres
```

### Database Issues
```bash
# Test database connection
psql -h localhost -U plantos_user -d plantos_dev

# Check TimescaleDB extension
psql -h localhost -U plantos_user -d plantos_dev -c "SELECT * FROM timescaledb_information.hypertables;"
```

### Python Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Next Steps

1. **Complete Phase 1**: FastAPI foundation with SQLAlchemy
2. **Start Phase 2**: Rust hardware drivers and Go streaming services
3. **Implement Phase 3**: Data engineering pipeline
4. **Deploy Phase 4**: Julia scientific computing
5. **Finalize Phase 5**: Production deployment automation

See `docs/todo.md` for detailed implementation roadmap and `docs/dev_log.md` for current progress tracking.