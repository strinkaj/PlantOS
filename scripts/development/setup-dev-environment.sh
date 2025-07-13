#!/bin/bash
set -euo pipefail

# PlantOS Development Environment Setup Script
# This script sets up the complete development environment for PlantOS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    log_info "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v lsb_release >/dev/null 2>&1; then
            OS_INFO=$(lsb_release -d | cut -f2)
            log_info "Detected: $OS_INFO"
        else
            log_info "Detected: Linux (unknown distribution)"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Detected: macOS"
    else
        log_warning "Unsupported OS: $OSTYPE. Proceeding anyway..."
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    if command -v apt >/dev/null 2>&1; then
        # Ubuntu/Debian
        log_info "Using apt package manager"
        
        # Check if we can run sudo
        if sudo -n true 2>/dev/null; then
            sudo apt update
            sudo apt install -y \
                python3 \
                python3-venv \
                python3-pip \
                podman \
                podman-compose \
                git \
                curl \
                wget \
                jq \
                make
            log_success "System dependencies installed via apt"
        else
            log_error "sudo access required for installing system packages"
            log_info "Please run: sudo apt install -y python3 python3-venv python3-pip podman podman-compose git curl wget jq make"
            exit 1
        fi
        
    elif command -v brew >/dev/null 2>&1; then
        # macOS with Homebrew
        log_info "Using Homebrew package manager"
        brew install python3 podman podman-compose git curl wget jq make
        log_success "System dependencies installed via Homebrew"
        
    else
        log_error "No supported package manager found (apt or brew)"
        log_info "Please install manually: python3, python3-venv, podman, podman-compose, git, curl, wget, jq, make"
        exit 1
    fi
}

# Setup Python virtual environment
setup_python_env() {
    log_info "Setting up Python virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_success "Python virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    
    log_success "Python dependencies installed"
}

# Setup pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    pre-commit install
    log_success "Pre-commit hooks installed"
}

# Create environment file
setup_env_file() {
    log_info "Setting up environment configuration..."
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -f ".env" ]]; then
        cp .env.example .env
        log_success "Environment file created from template"
        log_warning "Please review and modify .env file for your environment"
    else
        log_info "Environment file already exists"
    fi
}

# Start development services
start_services() {
    log_info "Starting development services..."
    
    cd "$PROJECT_ROOT"
    
    # Check if podman-compose is available
    if command -v podman-compose >/dev/null 2>&1; then
        podman-compose up -d
        log_success "Development services started with podman-compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        docker-compose up -d
        log_success "Development services started with docker-compose"
    else
        log_error "Neither podman-compose nor docker-compose found"
        exit 1
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    check_service_health
}

# Check service health
check_service_health() {
    log_info "Checking service health..."
    
    local services=(
        "postgres:5432"
        "redis:6379"
        "prometheus:9090"
        "grafana:3000"
        "kafka:9092"
        "minio:9000"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if nc -z localhost "$port" 2>/dev/null; then
            log_success "$name service is running on port $port"
        else
            log_warning "$name service is not responding on port $port"
        fi
    done
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Test Python imports
    python3 -c "
from src.shared.types import PlantID, SensorID
from src.infrastructure.hardware.interfaces import SensorType, HardwareManager
print('✅ Python imports successful')
" || {
        log_error "Python import test failed"
        exit 1
    }
    
    # Test database connection (if services are running)
    if nc -z localhost 5432 2>/dev/null; then
        log_info "Testing database connection..."
        # We'll add actual connection test later when we have SQLAlchemy setup
        log_success "Database port is accessible"
    fi
    
    log_success "Installation verification completed"
}

# Display access information
show_access_info() {
    log_info "Development environment is ready!"
    echo
    echo "Access URLs:"
    echo "  • Grafana Dashboard: http://localhost:3000 (admin/plantos123)"
    echo "  • Prometheus Metrics: http://localhost:9090"
    echo "  • MinIO Console: http://localhost:9001 (plantos_admin/plantos_dev_password)"
    echo "  • Spark Master UI: http://localhost:8080"
    echo
    echo "Development Commands:"
    echo "  • Activate Python env: source venv/bin/activate"
    echo "  • Run tests: pytest"
    echo "  • Check code quality: ruff check src/ && mypy src/"
    echo "  • Start API server: uvicorn src.main:app --reload"
    echo
    echo "Service Management:"
    echo "  • Stop services: podman-compose down"
    echo "  • View logs: podman-compose logs -f [service_name]"
    echo "  • Restart services: podman-compose restart"
    echo
}

# Main execution
main() {
    log_info "Starting PlantOS development environment setup..."
    
    check_os
    install_system_deps
    setup_python_env
    setup_precommit
    setup_env_file
    start_services
    verify_installation
    show_access_info
    
    log_success "PlantOS development environment setup completed!"
}

# Run main function
main "$@"