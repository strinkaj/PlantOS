"""
Timeout and retry patterns for external dependencies.

This module provides robust retry mechanisms with exponential backoff,
circuit breakers, and timeout handling for resilient service interactions.
"""

import asyncio
import time
import random
from typing import Any, Callable, Optional, Type, Union, List, Dict
from functools import wraps
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import structlog

from src.shared.exceptions import (
    TimeoutError, CircuitBreakerOpenError, CommunicationError,
    ExternalServiceError, is_retryable_error
)

logger = structlog.get_logger(__name__)


class RetryStrategy(Enum):
    """Available retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter: bool = True  # Add randomization to prevent thundering herd
    retryable_exceptions: tuple = (CommunicationError, TimeoutError, ExternalServiceError)
    timeout_seconds: Optional[float] = None
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        elif self.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.base_delay * self._fibonacci(attempt)
        else:
            delay = self.base_delay
            
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
            
        return max(0, delay)
    
    @staticmethod
    def _fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 3  # Number of successes needed to close when half-open
    timeout_seconds: float = 60.0  # Time to wait before trying half-open
    expected_exception_types: tuple = (CommunicationError, TimeoutError, ExternalServiceError)


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit breaker
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
        """
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker half-open", name=self.name)
                else:
                    logger.warning("Circuit breaker open", name=self.name)
                    raise CircuitBreakerOpenError(self.name)
                    
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            # Record success
            await self._record_success()
            return result
            
        except Exception as e:
            # Record failure if it's an expected exception type
            if isinstance(e, self.config.expected_exception_types):
                await self._record_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
            
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout_seconds
        
    async def _record_success(self):
        """Record successful execution."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker closed", name=self.name)
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
                
    async def _record_failure(self):
        """Record failed execution."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(
                        "Circuit breaker opened",
                        name=self.name,
                        failure_count=self.failure_count
                    )
                    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class ResilientClient:
    """Client wrapper with retry, timeout, and circuit breaker capabilities."""
    
    def __init__(
        self,
        name: str,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize resilient client.
        
        Args:
            name: Client identifier
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        self.name = name
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = None
        
        if circuit_breaker_config:
            self.circuit_breaker = CircuitBreaker(name, circuit_breaker_config)
            
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry and circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        last_exception = None
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                # Apply timeout if configured
                if self.retry_config.timeout_seconds:
                    result = await asyncio.wait_for(
                        self._execute_with_circuit_breaker(func, *args, **kwargs),
                        timeout=self.retry_config.timeout_seconds
                    )
                else:
                    result = await self._execute_with_circuit_breaker(func, *args, **kwargs)
                    
                logger.info(
                    "Function executed successfully",
                    client=self.name,
                    attempt=attempt,
                    function=func.__name__
                )
                return result
                
            except asyncio.TimeoutError:
                last_exception = TimeoutError(
                    operation=f"{self.name}.{func.__name__}",
                    timeout_seconds=self.retry_config.timeout_seconds
                )
                logger.warning(
                    "Function execution timed out",
                    client=self.name,
                    attempt=attempt,
                    timeout=self.retry_config.timeout_seconds
                )
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.warning(
                        "Non-retryable exception occurred",
                        client=self.name,
                        attempt=attempt,
                        exception=type(e).__name__,
                        error=str(e)
                    )
                    raise
                    
                logger.warning(
                    "Retryable exception occurred",
                    client=self.name,
                    attempt=attempt,
                    exception=type(e).__name__,
                    error=str(e)
                )
                
            # If this isn't the last attempt, wait before retrying
            if attempt < self.retry_config.max_attempts:
                delay = self.retry_config.calculate_delay(attempt)
                logger.info(
                    "Waiting before retry",
                    client=self.name,
                    attempt=attempt,
                    delay_seconds=delay
                )
                await asyncio.sleep(delay)
                
        # All retries exhausted
        logger.error(
            "All retry attempts exhausted",
            client=self.name,
            max_attempts=self.retry_config.max_attempts,
            last_exception=str(last_exception)
        )
        
        if last_exception:
            raise last_exception
        else:
            raise CommunicationError(f"All retry attempts failed for {self.name}")
            
    async def _execute_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker if configured."""
        if self.circuit_breaker:
            return await self.circuit_breaker.call(func, *args, **kwargs)
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        # Check against configured retryable exceptions
        if isinstance(exception, self.retry_config.retryable_exceptions):
            return True
            
        # Use the shared function for additional checks
        return is_retryable_error(exception)
        
    def get_status(self) -> Dict[str, Any]:
        """Get client status including circuit breaker state."""
        status = {
            "name": self.name,
            "retry_config": {
                "max_attempts": self.retry_config.max_attempts,
                "strategy": self.retry_config.strategy.value,
                "timeout_seconds": self.retry_config.timeout_seconds
            }
        }
        
        if self.circuit_breaker:
            status["circuit_breaker"] = self.circuit_breaker.get_status()
            
        return status


# Decorator for adding retry capabilities to functions
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    timeout_seconds: Optional[float] = None,
    retryable_exceptions: Optional[tuple] = None
):
    """
    Decorator to add retry capabilities to a function.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between attempts in seconds
        strategy: Retry strategy to use
        timeout_seconds: Timeout for each attempt
        retryable_exceptions: Tuple of exception types that should trigger retry
        
    Returns:
        Decorated function with retry capabilities
    """
    def decorator(func):
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            strategy=strategy,
            timeout_seconds=timeout_seconds,
            retryable_exceptions=retryable_exceptions or (CommunicationError, TimeoutError)
        )
        
        client = ResilientClient(
            name=f"{func.__module__}.{func.__name__}",
            retry_config=config
        )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await client.execute(func, *args, **kwargs)
            
        return wrapper
    return decorator


# Decorator for adding circuit breaker capabilities
def circuit_breaker(
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout_seconds: float = 60.0
):
    """
    Decorator to add circuit breaker capabilities to a function.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes needed to close circuit
        timeout_seconds: Time to wait before attempting half-open
        
    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func):
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds
        )
        
        client = ResilientClient(
            name=f"{func.__module__}.{func.__name__}",
            circuit_breaker_config=config
        )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await client.execute(func, *args, **kwargs)
            
        return wrapper
    return decorator


# Global registry for monitoring circuit breakers
class CircuitBreakerRegistry:
    """Registry for monitoring all circuit breakers in the application."""
    
    def __init__(self):
        """Initialize registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        
    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker."""
        self._breakers[breaker.name] = breaker
        
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }
        
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()