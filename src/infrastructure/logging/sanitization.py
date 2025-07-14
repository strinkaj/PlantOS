"""
Logging sanitization for sensitive data protection.

This module provides utilities to sanitize log output, ensuring that sensitive
information is never exposed in log files or monitoring systems.
"""

import re
import json
from typing import Any, Dict, List, Set, Union, Optional
from copy import deepcopy
import structlog


class LogSanitizer:
    """Sanitizes sensitive data from log output."""
    
    # Sensitive field patterns (case-insensitive)
    SENSITIVE_FIELD_PATTERNS = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'auth',
        'authorization', 'bearer', 'api_key', 'apikey', 'private_key',
        'access_token', 'refresh_token', 'session_id', 'session_key',
        'credit_card', 'card_number', 'cvv', 'ssn', 'social_security',
        'phone', 'email', 'address', 'location', 'gps', 'coordinates',
        'personal', 'private', 'confidential', 'secure'
    }
    
    # Sensitive value patterns (regex)
    SENSITIVE_VALUE_PATTERNS = [
        # Credit card numbers
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        # SSN patterns
        r'\b\d{3}-\d{2}-\d{4}\b',
        # Email addresses
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # Phone numbers
        r'\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        # JWT tokens (basic pattern)
        r'\beyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*\b',
        # API keys (common patterns)
        r'\b[A-Za-z0-9]{32,}\b',
        # UUID patterns (sometimes used as secrets)
        r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'
    ]
    
    # Replacement text for sanitized values
    REPLACEMENT_TEXT = "***REDACTED***"
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
        """
        Recursively sanitize a dictionary, removing sensitive data.
        
        Args:
            data: Dictionary to sanitize
            max_depth: Maximum recursion depth to prevent infinite loops
            
        Returns:
            Sanitized dictionary with sensitive fields redacted
        """
        if max_depth <= 0:
            return {"error": "max_depth_reached"}
            
        if not isinstance(data, dict):
            return cls._sanitize_value(data)
            
        sanitized = {}
        
        for key, value in data.items():
            sanitized_key = str(key).lower()
            
            # Check if key indicates sensitive data
            if any(pattern in sanitized_key for pattern in cls.SENSITIVE_FIELD_PATTERNS):
                sanitized[key] = cls.REPLACEMENT_TEXT
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value, max_depth - 1)
            elif isinstance(value, list):
                sanitized[key] = cls._sanitize_list(value, max_depth - 1)
            else:
                sanitized[key] = cls._sanitize_value(value)
                
        return sanitized
    
    @classmethod
    def _sanitize_list(cls, data: List[Any], max_depth: int) -> List[Any]:
        """Sanitize a list of values."""
        if max_depth <= 0:
            return ["max_depth_reached"]
            
        sanitized = []
        for item in data:
            if isinstance(item, dict):
                sanitized.append(cls.sanitize_dict(item, max_depth - 1))
            elif isinstance(item, list):
                sanitized.append(cls._sanitize_list(item, max_depth - 1))
            else:
                sanitized.append(cls._sanitize_value(item))
                
        return sanitized
    
    @classmethod
    def _sanitize_value(cls, value: Any) -> Any:
        """Sanitize a single value based on content patterns."""
        if not isinstance(value, str):
            return value
            
        # Check for sensitive patterns in the value
        for pattern in cls.SENSITIVE_VALUE_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return cls.REPLACEMENT_TEXT
                
        return value
    
    @classmethod
    def sanitize_string(cls, text: str) -> str:
        """
        Sanitize a string by replacing sensitive patterns.
        
        Args:
            text: String to sanitize
            
        Returns:
            Sanitized string with sensitive data redacted
        """
        if not isinstance(text, str):
            return str(text)
            
        sanitized = text
        
        # Replace sensitive patterns
        for pattern in cls.SENSITIVE_VALUE_PATTERNS:
            sanitized = re.sub(pattern, cls.REPLACEMENT_TEXT, sanitized, flags=re.IGNORECASE)
            
        return sanitized
    
    @classmethod
    def sanitize_url(cls, url: str) -> str:
        """
        Sanitize URL by removing sensitive query parameters and credentials.
        
        Args:
            url: URL to sanitize
            
        Returns:
            Sanitized URL
        """
        if not isinstance(url, str):
            return str(url)
            
        # Remove credentials from URL (user:pass@host)
        sanitized = re.sub(r'://[^@/]+@', '://***:***@', url)
        
        # Remove sensitive query parameters
        sensitive_params = ['token', 'key', 'secret', 'password', 'auth', 'api_key']
        for param in sensitive_params:
            pattern = rf'[?&]{param}=[^&]*'
            sanitized = re.sub(pattern, f'&{param}={cls.REPLACEMENT_TEXT}', sanitized, flags=re.IGNORECASE)
            
        return sanitized


class StructlogSanitizer:
    """Structlog processor for sanitizing log events."""
    
    def __init__(self, sanitizer: Optional[LogSanitizer] = None):
        """Initialize with optional custom sanitizer."""
        self.sanitizer = sanitizer or LogSanitizer()
        
    def __call__(self, logger, method_name, event_dict):
        """
        Structlog processor that sanitizes event data.
        
        Args:
            logger: Logger instance
            method_name: Logging method name
            event_dict: Event dictionary to sanitize
            
        Returns:
            Sanitized event dictionary
        """
        try:
            # Create a deep copy to avoid modifying original data
            sanitized_event = deepcopy(event_dict)
            
            # Sanitize the entire event dictionary
            sanitized_event = self.sanitizer.sanitize_dict(sanitized_event)
            
            # Special handling for common fields
            if 'url' in sanitized_event:
                sanitized_event['url'] = self.sanitizer.sanitize_url(sanitized_event['url'])
                
            if 'error' in sanitized_event and isinstance(sanitized_event['error'], str):
                sanitized_event['error'] = self.sanitizer.sanitize_string(sanitized_event['error'])
                
            if 'message' in sanitized_event and isinstance(sanitized_event['message'], str):
                sanitized_event['message'] = self.sanitizer.sanitize_string(sanitized_event['message'])
                
            return sanitized_event
            
        except Exception as e:
            # If sanitization fails, log the error but don't expose the original data
            return {
                "event": "log_sanitization_error",
                "error": str(e),
                "original_event_type": type(event_dict).__name__
            }


def create_sanitized_logger(logger_name: str) -> structlog.stdlib.BoundLogger:
    """
    Create a structlog logger with sanitization processor.
    
    Args:
        logger_name: Name for the logger
        
    Returns:
        Configured logger with sanitization
    """
    # Configure structlog with sanitization
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.add_logger_name,
            structlog.processors.TimeStamper(fmt="ISO"),
            StructlogSanitizer(),  # Add our sanitization processor
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(logger_name)


def sanitize_exception_args(exc: Exception) -> Exception:
    """
    Sanitize exception arguments to remove sensitive data.
    
    Args:
        exc: Exception to sanitize
        
    Returns:
        Exception with sanitized arguments
    """
    if not exc.args:
        return exc
        
    sanitized_args = []
    
    for arg in exc.args:
        if isinstance(arg, str):
            sanitized_args.append(LogSanitizer.sanitize_string(arg))
        elif isinstance(arg, dict):
            sanitized_args.append(LogSanitizer.sanitize_dict(arg))
        else:
            sanitized_args.append(arg)
            
    # Create new exception with sanitized args
    exc_type = type(exc)
    try:
        sanitized_exc = exc_type(*sanitized_args)
        # Copy other attributes if they exist
        for attr in ['details', 'error_code', 'field', 'value']:
            if hasattr(exc, attr):
                value = getattr(exc, attr)
                if isinstance(value, (str, dict)):
                    if isinstance(value, str):
                        setattr(sanitized_exc, attr, LogSanitizer.sanitize_string(value))
                    else:
                        setattr(sanitized_exc, attr, LogSanitizer.sanitize_dict(value))
                else:
                    setattr(sanitized_exc, attr, value)
        return sanitized_exc
    except Exception:
        # If we can't create a sanitized version, return a generic error
        return exc_type("Exception details redacted for security")


# Configuration for different environments
class LogSanitizationConfig:
    """Configuration for log sanitization behavior."""
    
    def __init__(self, environment: str = "production"):
        """
        Initialize sanitization config.
        
        Args:
            environment: Environment name (development, staging, production)
        """
        self.environment = environment.lower()
        
    @property
    def should_sanitize(self) -> bool:
        """Whether sanitization should be enabled."""
        # Always sanitize in production and staging
        return self.environment in ['production', 'staging']
        
    @property
    def sanitization_level(self) -> str:
        """Level of sanitization to apply."""
        if self.environment == 'production':
            return 'strict'
        elif self.environment == 'staging':
            return 'moderate'
        else:
            return 'minimal'
            
    def get_sanitizer(self) -> LogSanitizer:
        """Get configured sanitizer for the environment."""
        sanitizer = LogSanitizer()
        
        if self.sanitization_level == 'minimal':
            # Only sanitize obvious secrets in development
            sanitizer.SENSITIVE_FIELD_PATTERNS = {
                'password', 'secret', 'token', 'key', 'private_key'
            }
            
        return sanitizer


# Default sanitized logger instance
default_config = LogSanitizationConfig()
sanitized_logger = create_sanitized_logger("plantos")