#!/bin/bash
set -euo pipefail

# PlantOS Environment Verification Script
# Tests the development environment setup without requiring full installation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Run test with output capture
run_test_with_output() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${BLUE}Testing $test_name...${NC}"
    
    if output=$(eval "$test_command" 2>&1); then
        echo -e "${GREEN}✅ PASS${NC}"
        echo "$output" | sed 's/^/  /'
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "$output" | sed 's/^/  /' >&2
        ((TESTS_FAILED++))
        return 1
    fi
}

echo -e "${BLUE}PlantOS Environment Verification${NC}"
echo "=================================="
echo

cd "$PROJECT_ROOT"

# Test 1: Project structure
run_test "Project structure" "
    [[ -d src/core/domain ]] && 
    [[ -d src/infrastructure/hardware ]] && 
    [[ -d services ]] && 
    [[ -d hardware ]] && 
    [[ -d analytics ]]
"

# Test 2: Configuration files
run_test "Configuration files" "
    [[ -f podman-compose.yml ]] && 
    [[ -f .env.example ]] && 
    [[ -f requirements.txt ]] && 
    [[ -f requirements-dev.txt ]]
"

# Test 3: Python availability
run_test "Python 3.10+" "python3 -c 'import sys; assert sys.version_info >= (3, 10)'"

# Test 4: Python module imports
run_test_with_output "Python module imports" "
    python3 -c '
import sys
sys.path.insert(0, \".\")
from src.shared.types import PlantID, SensorID
from src.infrastructure.hardware.interfaces import SensorType, ActuatorType
print(f\"Available sensor types: {len(list(SensorType))}\")
print(f\"Available actuator types: {len(list(ActuatorType))}\")
'
"

# Test 5: YAML configuration validation
run_test_with_output "YAML configuration validation" "
    python3 -c '
import yaml
with open(\"podman-compose.yml\", \"r\") as f:
    compose = yaml.safe_load(f)
with open(\"monitoring/prometheus/prometheus.yml\", \"r\") as f:
    prometheus = yaml.safe_load(f)
print(f\"Compose services: {len(compose.get(\"services\", {}))}\")
print(f\"Prometheus jobs: {len(prometheus.get(\"scrape_configs\", []))}\")
'
"

# Test 6: SQL initialization script
run_test "SQL initialization script" "
    grep -q 'CREATE EXTENSION.*timescaledb' docker/postgres/init/01-init-timescaledb.sql &&
    grep -q 'create_hypertable' docker/postgres/init/01-init-timescaledb.sql
"

# Test 7: Hardware abstraction interfaces
run_test_with_output "Hardware abstraction interfaces" "
    python3 -c '
import sys
sys.path.insert(0, \".\")
from src.infrastructure.hardware.interfaces import HardwareManager, SensorInterface, ActuatorInterface
import inspect
print(f\"HardwareManager methods: {len([m for m in dir(HardwareManager) if not m.startswith(\"_\")])}\")
print(f\"SensorInterface is protocol: {hasattr(SensorInterface, \"__protocol__\") or \"abstractmethod\" in str(inspect.getmembers(SensorInterface))}\")
'
"

# Test 8: Container runtime availability (optional)
echo -e "${BLUE}Testing container runtime availability...${NC}"
if command -v podman >/dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS${NC} - Podman available"
    ((TESTS_PASSED++))
elif command -v docker >/dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS${NC} - Docker available"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  OPTIONAL${NC} - No container runtime found (run setup script to install)"
fi

# Test 9: Development tools
echo -e "${BLUE}Testing development tools availability...${NC}"
tools=("git" "curl" "wget" "jq")
for tool in "${tools[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo -e "  $tool: ${GREEN}✅ Available${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "  $tool: ${YELLOW}⚠️  Missing${NC}"
    fi
done

echo
echo "=================================="
echo -e "${BLUE}Verification Summary${NC}"
echo "=================================="
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo
    echo -e "${GREEN}🎉 Environment verification successful!${NC}"
    echo "Your PlantOS development environment structure is correctly set up."
    echo
    echo "Next steps:"
    echo "  1. Run: ./scripts/development/setup-dev-environment.sh (to install dependencies)"
    echo "  2. Or manually install: podman, python3-venv, and run 'podman-compose up -d'"
    echo "  3. Create virtual environment: python3 -m venv venv && source venv/bin/activate"
    echo "  4. Install dependencies: pip install -r requirements.txt -r requirements-dev.txt"
    exit 0
else
    echo
    echo -e "${RED}❌ Environment verification failed!${NC}"
    echo "Please fix the failed tests before proceeding."
    exit 1
fi