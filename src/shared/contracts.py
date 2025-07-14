"""
Contract Programming implementation with preconditions and postconditions.

This module provides decorators and utilities for implementing Design by Contract
principles, ensuring robust and reliable code through explicit contracts.
"""

import functools
import inspect
from typing import Any, Callable, TypeVar, Optional, Union
from datetime import datetime, timedelta
import structlog

from src.shared.exceptions import ContractViolationError, PreconditionError, PostconditionError

logger = structlog.get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class ContractMeta:
    """Metadata for contract violations."""
    
    def __init__(self, function_name: str, contract_type: str, condition: str):
        self.function_name = function_name
        self.contract_type = contract_type
        self.condition = condition
        self.timestamp = datetime.utcnow()


def require(condition: Union[bool, Callable[..., bool]], message: str = "") -> Callable[[F], F]:
    """
    Precondition decorator - validates input parameters.
    
    Args:
        condition: Boolean expression or callable that takes function arguments
        message: Custom error message for contract violation
        
    Returns:
        Decorated function with precondition checking
        
    Raises:
        PreconditionError: If precondition is not met
    """
    def decorator(func: F) -> F:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature for parameter inspection
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Evaluate condition
            if callable(condition):
                try:
                    result = condition(*args, **kwargs)
                except Exception as e:
                    logger.error(
                        "Precondition evaluation failed",
                        function=func.__name__,
                        error=str(e)
                    )
                    raise PreconditionError(
                        f"Precondition evaluation error in {func.__name__}: {str(e)}"
                    )
            else:
                result = condition
                
            if not result:
                error_msg = message or f"Precondition failed in {func.__name__}"
                logger.warning(
                    "Precondition violation",
                    function=func.__name__,
                    message=error_msg,
                    args=str(bound_args.arguments)
                )
                raise PreconditionError(error_msg)
                
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


def ensure(condition: Union[bool, Callable[..., bool]], message: str = "") -> Callable[[F], F]:
    """
    Postcondition decorator - validates return values and state changes.
    
    Args:
        condition: Boolean expression or callable that takes function arguments and result
        message: Custom error message for contract violation
        
    Returns:
        Decorated function with postcondition checking
        
    Raises:
        PostconditionError: If postcondition is not met
    """
    def decorator(func: F) -> F:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)
            
            # Evaluate postcondition
            if callable(condition):
                try:
                    # Pass original arguments plus result to condition
                    condition_result = condition(result, *args, **kwargs)
                except Exception as e:
                    logger.error(
                        "Postcondition evaluation failed",
                        function=func.__name__,
                        error=str(e)
                    )
                    raise PostconditionError(
                        f"Postcondition evaluation error in {func.__name__}: {str(e)}"
                    )
            else:
                condition_result = condition
                
            if not condition_result:
                error_msg = message or f"Postcondition failed in {func.__name__}"
                logger.warning(
                    "Postcondition violation",
                    function=func.__name__,
                    message=error_msg,
                    result=str(result)
                )
                raise PostconditionError(error_msg)
                
            return result
            
        return wrapper
    return decorator


def invariant(condition: Callable[[Any], bool], message: str = "") -> Callable[[type], type]:
    """
    Class invariant decorator - validates object state consistency.
    
    Args:
        condition: Callable that takes self and returns boolean
        message: Custom error message for invariant violation
        
    Returns:
        Decorated class with invariant checking
        
    Raises:
        ContractViolationError: If invariant is violated
    """
    def class_decorator(cls):
        # Store original methods
        original_init = cls.__init__
        original_setattr = cls.__setattr__
        
        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _check_invariant(self, condition, message, cls.__name__)
            
        @functools.wraps(original_setattr)
        def new_setattr(self, name, value):
            original_setattr(self, name, value)
            _check_invariant(self, condition, message, cls.__name__)
            
        # Replace methods
        cls.__init__ = new_init
        cls.__setattr__ = new_setattr
        
        return cls
    return class_decorator


def _check_invariant(instance: Any, condition: Callable, message: str, class_name: str):
    """Check class invariant condition."""
    try:
        if not condition(instance):
            error_msg = message or f"Invariant violation in {class_name}"
            logger.warning(
                "Class invariant violation",
                class_name=class_name,
                message=error_msg
            )
            raise ContractViolationError(error_msg)
    except Exception as e:
        if isinstance(e, ContractViolationError):
            raise
        logger.error(
            "Invariant evaluation failed",
            class_name=class_name,
            error=str(e)
        )
        raise ContractViolationError(
            f"Invariant evaluation error in {class_name}: {str(e)}"
        )


# Common contract conditions for PlantOS domain

def positive(value: Union[int, float]) -> bool:
    """Check if value is positive."""
    return value > 0


def non_negative(value: Union[int, float]) -> bool:
    """Check if value is non-negative."""
    return value >= 0


def in_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> bool:
    """Check if value is within specified range."""
    return min_val <= value <= max_val


def valid_moisture_level(moisture: float) -> bool:
    """Check if moisture level is valid (0-100%)."""
    return 0 <= moisture <= 100


def valid_temperature(temp: float) -> bool:
    """Check if temperature is within reasonable range (-50°C to 80°C)."""
    return -50 <= temp <= 80


def valid_humidity(humidity: float) -> bool:
    """Check if humidity is valid (0-100%)."""
    return 0 <= humidity <= 100


def valid_ph_level(ph: float) -> bool:
    """Check if pH level is valid (0-14)."""
    return 0 <= ph <= 14


def valid_uuid_string(value: str) -> bool:
    """Check if string is a valid UUID format."""
    import uuid
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def non_empty_string(value: str) -> bool:
    """Check if string is non-empty after stripping whitespace."""
    return isinstance(value, str) and len(value.strip()) > 0


def valid_email(email: str) -> bool:
    """Check if string is a valid email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def future_datetime(dt: datetime) -> bool:
    """Check if datetime is in the future."""
    return dt > datetime.utcnow()


def recent_datetime(dt: datetime, max_age_hours: int = 24) -> bool:
    """Check if datetime is recent (within specified hours)."""
    return datetime.utcnow() - dt <= timedelta(hours=max_age_hours)


# Contract validation decorators for specific domain objects

def validate_plant_data(func: F) -> F:
    """Validate plant data in function arguments."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract plant data from arguments
        for arg_name, arg_value in kwargs.items():
            if arg_name == 'moisture_level' and arg_value is not None:
                if not valid_moisture_level(arg_value):
                    raise PreconditionError(f"Invalid moisture level: {arg_value}")
            elif arg_name == 'temperature' and arg_value is not None:
                if not valid_temperature(arg_value):
                    raise PreconditionError(f"Invalid temperature: {arg_value}")
            elif arg_name == 'humidity' and arg_value is not None:
                if not valid_humidity(arg_value):
                    raise PreconditionError(f"Invalid humidity: {arg_value}")
            elif arg_name == 'ph_level' and arg_value is not None:
                if not valid_ph_level(arg_value):
                    raise PreconditionError(f"Invalid pH level: {arg_value}")
                    
        return func(*args, **kwargs)
    return wrapper


def validate_sensor_reading(func: F) -> F:
    """Validate sensor reading data."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Validate sensor reading parameters
        if 'value' in kwargs and kwargs['value'] is not None:
            value = kwargs['value']
            sensor_type = kwargs.get('sensor_type', '')
            
            if sensor_type == 'moisture' and not valid_moisture_level(value):
                raise PreconditionError(f"Invalid moisture reading: {value}")
            elif sensor_type == 'temperature' and not valid_temperature(value):
                raise PreconditionError(f"Invalid temperature reading: {value}")
            elif sensor_type == 'humidity' and not valid_humidity(value):
                raise PreconditionError(f"Invalid humidity reading: {value}")
            elif sensor_type == 'ph' and not valid_ph_level(value):
                raise PreconditionError(f"Invalid pH reading: {value}")
                
        return func(*args, **kwargs)
    return wrapper