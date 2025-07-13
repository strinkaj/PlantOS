"""
Hardware abstraction interfaces for PlantOS.

This module defines the contracts for hardware interactions,
enabling seamless switching between real hardware (Raspberry Pi)
and simulated hardware (development environment).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol
from enum import Enum
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass

from src.shared.types import PlantID, SensorID, PinNumber


class SensorType(str, Enum):
    """Types of sensors supported by the system."""
    MOISTURE = "moisture"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    LIGHT = "light"
    PH = "ph"
    CONDUCTIVITY = "conductivity"


class ActuatorType(str, Enum):
    """Types of actuators supported by the system."""
    PUMP = "pump"
    VALVE = "valve"
    FAN = "fan"
    LIGHT = "light"
    HEATER = "heater"


@dataclass
class SensorReading:
    """Represents a single sensor reading with metadata."""
    sensor_id: SensorID
    plant_id: Optional[PlantID]
    sensor_type: SensorType
    value: Decimal
    unit: str
    quality_score: float  # 0.0 to 1.0, where 1.0 is perfect quality
    timestamp: datetime
    metadata: Optional[Dict] = None


@dataclass
class ActuatorCommand:
    """Represents a command to control an actuator."""
    actuator_id: str
    actuator_type: ActuatorType
    action: str  # e.g., "on", "off", "set_speed", "set_position"
    value: Optional[float] = None  # For variable controls
    duration_seconds: Optional[int] = None
    safety_timeout: Optional[int] = None


class SensorInterface(Protocol):
    """Protocol for sensor implementations."""
    
    @abstractmethod
    async def read_value(self) -> SensorReading:
        """Read current sensor value."""
        ...
    
    @abstractmethod
    async def calibrate(self, reference_values: Dict[str, float]) -> bool:
        """Calibrate sensor with known reference values."""
        ...
    
    @abstractmethod
    async def get_status(self) -> Dict[str, any]:
        """Get sensor health and status information."""
        ...


class ActuatorInterface(Protocol):
    """Protocol for actuator implementations."""
    
    @abstractmethod
    async def execute_command(self, command: ActuatorCommand) -> bool:
        """Execute an actuator command."""
        ...
    
    @abstractmethod
    async def get_status(self) -> Dict[str, any]:
        """Get actuator status and health information."""
        ...
    
    @abstractmethod
    async def emergency_stop(self) -> bool:
        """Immediately stop actuator for safety."""
        ...


class GPIOInterface(Protocol):
    """Protocol for GPIO pin control."""
    
    @abstractmethod
    async def setup_pin(self, pin: PinNumber, mode: str) -> bool:
        """Setup GPIO pin for input or output."""
        ...
    
    @abstractmethod
    async def read_digital(self, pin: PinNumber) -> bool:
        """Read digital value from GPIO pin."""
        ...
    
    @abstractmethod
    async def write_digital(self, pin: PinNumber, value: bool) -> bool:
        """Write digital value to GPIO pin."""
        ...
    
    @abstractmethod
    async def read_analog(self, pin: PinNumber) -> float:
        """Read analog value from GPIO pin (0.0 to 1.0)."""
        ...
    
    @abstractmethod
    async def write_pwm(self, pin: PinNumber, duty_cycle: float) -> bool:
        """Write PWM signal to GPIO pin (0.0 to 1.0)."""
        ...


class I2CInterface(Protocol):
    """Protocol for I2C communication."""
    
    @abstractmethod
    async def read_byte(self, address: int, register: int) -> int:
        """Read single byte from I2C device."""
        ...
    
    @abstractmethod
    async def write_byte(self, address: int, register: int, value: int) -> bool:
        """Write single byte to I2C device."""
        ...
    
    @abstractmethod
    async def read_bytes(self, address: int, register: int, length: int) -> bytes:
        """Read multiple bytes from I2C device."""
        ...


class HardwareManager(ABC):
    """Abstract base class for hardware management."""
    
    def __init__(self, config: Dict[str, any]):
        """Initialize hardware manager with configuration."""
        self.config = config
        self.sensors: Dict[SensorID, SensorInterface] = {}
        self.actuators: Dict[str, ActuatorInterface] = {}
        self.gpio: Optional[GPIOInterface] = None
        self.i2c: Optional[I2CInterface] = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize all hardware components."""
        ...
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Safely shutdown all hardware components."""
        ...
    
    @abstractmethod
    async def register_sensor(
        self, 
        sensor_id: SensorID, 
        sensor_type: SensorType,
        config: Dict[str, any]
    ) -> bool:
        """Register a new sensor with the hardware manager."""
        ...
    
    @abstractmethod
    async def register_actuator(
        self, 
        actuator_id: str, 
        actuator_type: ActuatorType,
        config: Dict[str, any]
    ) -> bool:
        """Register a new actuator with the hardware manager."""
        ...
    
    async def read_all_sensors(self) -> List[SensorReading]:
        """Read values from all registered sensors."""
        readings = []
        for sensor in self.sensors.values():
            try:
                reading = await sensor.read_value()
                readings.append(reading)
            except Exception as e:
                # Log error but continue with other sensors
                # TODO: Add proper logging
                pass
        return readings
    
    async def get_system_status(self) -> Dict[str, any]:
        """Get overall system health status."""
        status = {
            "sensors": {},
            "actuators": {},
            "gpio_available": self.gpio is not None,
            "i2c_available": self.i2c is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Get sensor statuses
        for sensor_id, sensor in self.sensors.items():
            try:
                status["sensors"][str(sensor_id)] = await sensor.get_status()
            except Exception:
                status["sensors"][str(sensor_id)] = {"status": "error"}
        
        # Get actuator statuses
        for actuator_id, actuator in self.actuators.items():
            try:
                status["actuators"][actuator_id] = await actuator.get_status()
            except Exception:
                status["actuators"][actuator_id] = {"status": "error"}
        
        return status
    
    async def emergency_shutdown(self) -> bool:
        """Emergency stop all actuators and shutdown system."""
        success = True
        
        # Stop all actuators
        for actuator in self.actuators.values():
            try:
                await actuator.emergency_stop()
            except Exception:
                success = False
        
        # Shutdown system
        try:
            await self.shutdown()
        except Exception:
            success = False
        
        return success