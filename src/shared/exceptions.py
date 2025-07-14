"""
Custom exceptions for PlantOS application.

This module defines all custom exceptions used throughout the application,
providing clear error hierarchies and detailed error information for debugging.
"""

from typing import Optional, Dict, Any
from uuid import UUID


class PlantOSError(Exception):
    """Base exception for all PlantOS errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code


class ConfigurationError(PlantOSError):
    """Raised when there are configuration or setup issues."""
    pass


class ValidationError(PlantOSError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value


class ContractViolationError(PlantOSError):
    """Base class for contract programming violations."""
    pass


class PreconditionError(ContractViolationError):
    """Raised when a function precondition is violated."""
    pass


class PostconditionError(ContractViolationError):
    """Raised when a function postcondition is violated."""
    pass


class InvariantError(ContractViolationError):
    """Raised when a class invariant is violated."""
    pass


class DatabaseError(PlantOSError):
    """Raised when database operations fail."""
    pass


class HardwareError(PlantOSError):
    """Base class for hardware-related errors."""
    pass


class SensorError(HardwareError):
    """Raised when sensor operations fail."""
    
    def __init__(self, message: str, sensor_id: Optional[UUID] = None, sensor_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type


class ActuatorError(HardwareError):
    """Raised when actuator operations fail."""
    
    def __init__(self, message: str, actuator_id: Optional[UUID] = None, actuator_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.actuator_id = actuator_id
        self.actuator_type = actuator_type


class SafetyError(HardwareError):
    """Raised when safety limits are exceeded."""
    
    def __init__(self, message: str, safety_limit: Optional[str] = None, current_value: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.safety_limit = safety_limit
        self.current_value = current_value


class CommunicationError(PlantOSError):
    """Raised when communication with external services fails."""
    
    def __init__(self, message: str, service: Optional[str] = None, endpoint: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.service = service
        self.endpoint = endpoint


class AuthenticationError(PlantOSError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(PlantOSError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str, required_permission: Optional[str] = None, user_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.required_permission = required_permission
        self.user_id = user_id


class BusinessLogicError(PlantOSError):
    """Raised when business logic constraints are violated."""
    pass


class PlantNotFoundError(BusinessLogicError):
    """Raised when a plant is not found."""
    
    def __init__(self, plant_id: UUID, **kwargs):
        super().__init__(f"Plant not found: {plant_id}", **kwargs)
        self.plant_id = plant_id


class SpeciesNotFoundError(BusinessLogicError):
    """Raised when a plant species is not found."""
    
    def __init__(self, species_id: UUID, **kwargs):
        super().__init__(f"Plant species not found: {species_id}", **kwargs)
        self.species_id = species_id


class SensorNotFoundError(BusinessLogicError):
    """Raised when a sensor is not found."""
    
    def __init__(self, sensor_id: UUID, **kwargs):
        super().__init__(f"Sensor not found: {sensor_id}", **kwargs)
        self.sensor_id = sensor_id


class DuplicateResourceError(BusinessLogicError):
    """Raised when trying to create a resource that already exists."""
    
    def __init__(self, resource_type: str, identifier: str, **kwargs):
        super().__init__(f"Duplicate {resource_type}: {identifier}", **kwargs)
        self.resource_type = resource_type
        self.identifier = identifier


class CircuitBreakerOpenError(PlantOSError):
    """Raised when a circuit breaker is open."""
    
    def __init__(self, service: str, **kwargs):
        super().__init__(f"Circuit breaker open for service: {service}", **kwargs)
        self.service = service


class RateLimitExceededError(PlantOSError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, limit: int, window: int, **kwargs):
        super().__init__(f"Rate limit exceeded: {limit} requests per {window} seconds", **kwargs)
        self.limit = limit
        self.window = window


class ExternalServiceError(CommunicationError):
    """Raised when external services fail."""
    
    def __init__(self, service: str, status_code: Optional[int] = None, response_body: Optional[str] = None, **kwargs):
        message = f"External service error: {service}"
        if status_code:
            message += f" (HTTP {status_code})"
        super().__init__(message, service=service, **kwargs)
        self.status_code = status_code
        self.response_body = response_body


class TimeoutError(PlantOSError):
    """Raised when operations timeout."""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(f"Operation timed out: {operation} after {timeout_seconds}s", **kwargs)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class SecurityError(PlantOSError):
    """Raised when security violations are detected."""
    
    def __init__(self, message: str, severity: str = "medium", **kwargs):
        super().__init__(message, **kwargs)
        self.severity = severity


class DataIntegrityError(DatabaseError):
    """Raised when data integrity constraints are violated."""
    
    def __init__(self, constraint: str, table: Optional[str] = None, **kwargs):
        message = f"Data integrity violation: {constraint}"
        if table:
            message += f" in table {table}"
        super().__init__(message, **kwargs)
        self.constraint = constraint
        self.table = table


class ConcurrencyError(DatabaseError):
    """Raised when concurrent access causes conflicts."""
    
    def __init__(self, resource: str, operation: str, **kwargs):
        super().__init__(f"Concurrency conflict: {operation} on {resource}", **kwargs)
        self.resource = resource
        self.operation = operation


class MaintenanceModeError(PlantOSError):
    """Raised when the system is in maintenance mode."""
    
    def __init__(self, estimated_duration: Optional[int] = None, **kwargs):
        message = "System is in maintenance mode"
        if estimated_duration:
            message += f" (estimated duration: {estimated_duration} minutes)"
        super().__init__(message, **kwargs)
        self.estimated_duration = estimated_duration


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAP = {
    ValidationError: 400,
    PlantNotFoundError: 404,
    SpeciesNotFoundError: 404,
    SensorNotFoundError: 404,
    DuplicateResourceError: 409,
    AuthenticationError: 401,
    AuthorizationError: 403,
    RateLimitExceededError: 429,
    MaintenanceModeError: 503,
    SafetyError: 400,
    BusinessLogicError: 400,
    ConfigurationError: 500,
    DatabaseError: 500,
    HardwareError: 500,
    CommunicationError: 502,
    ExternalServiceError: 502,
    TimeoutError: 504,
    SecurityError: 403,
    DataIntegrityError: 500,
    ConcurrencyError: 409,
    CircuitBreakerOpenError: 503,
}


def get_http_status_code(exception: Exception) -> int:
    """Get appropriate HTTP status code for an exception."""
    exception_type = type(exception)
    return EXCEPTION_STATUS_MAP.get(exception_type, 500)


def is_retryable_error(exception: Exception) -> bool:
    """Determine if an error is retryable."""
    retryable_exceptions = (
        CommunicationError,
        TimeoutError,
        CircuitBreakerOpenError,
        DatabaseError,  # Some database errors are retryable
    )
    
    # Don't retry authentication, authorization, or validation errors
    non_retryable_exceptions = (
        AuthenticationError,
        AuthorizationError,
        ValidationError,
        SecurityError,
        BusinessLogicError,
    )
    
    if isinstance(exception, non_retryable_exceptions):
        return False
    
    return isinstance(exception, retryable_exceptions)


def should_log_error(exception: Exception) -> bool:
    """Determine if an error should be logged."""
    # Don't log expected business logic errors and user errors
    low_priority_exceptions = (
        ValidationError,
        PlantNotFoundError,
        SpeciesNotFoundError,
        SensorNotFoundError,
        AuthenticationError,
        AuthorizationError,
    )
    
    return not isinstance(exception, low_priority_exceptions)