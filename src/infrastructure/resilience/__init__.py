"""
Resilience patterns for fault-tolerant systems.

This module provides retry mechanisms, circuit breakers, and timeout patterns
for building resilient applications that gracefully handle failures.
"""

from .retry import (
    RetryStrategy,
    RetryConfig,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreaker,
    ResilientClient,
    retry,
    circuit_breaker,
    CircuitBreakerRegistry,
    circuit_breaker_registry
)

__all__ = [
    "RetryStrategy",
    "RetryConfig",
    "CircuitBreakerState", 
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "ResilientClient",
    "retry",
    "circuit_breaker",
    "CircuitBreakerRegistry",
    "circuit_breaker_registry"
]