"""
Input validation and sanitization framework.

This module provides comprehensive input validation and sanitization utilities
to ensure data integrity and security across all API endpoints.
"""

import re
import html
import urllib.parse
import functools
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from uuid import UUID
import bleach
import structlog

from src.shared.exceptions import ValidationError, SecurityError
from src.shared.contracts import (
    valid_moisture_level, valid_temperature, valid_humidity, 
    valid_ph_level, non_empty_string, valid_email
)

logger = structlog.get_logger(__name__)


class ValidationResult:
    """Result of validation with details about errors."""
    
    def __init__(self, is_valid: bool = True, errors: Optional[List[str]] = None, sanitized_value: Any = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.sanitized_value = sanitized_value
        
    def add_error(self, error: str):
        """Add an error message."""
        self.is_valid = False
        self.errors.append(error)


class InputSanitizer:
    """Comprehensive input sanitization utilities."""
    
    # Allowed HTML tags for rich text fields (very restrictive)
    ALLOWED_HTML_TAGS = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
    ALLOWED_HTML_ATTRIBUTES = {}
    
    # Regular expressions for common patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\bSCRIPT\b)",
    ]
    
    XSS_PATTERNS = [
        r"(<script[^>]*>.*?</script>)",
        r"(javascript:|vbscript:|onload=|onerror=|onclick=)",
        r"(<iframe[^>]*>.*?</iframe>)",
        r"(<object[^>]*>.*?</object>)",
        r"(<embed[^>]*>.*?</embed>)",
    ]
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: Optional[int] = None, allow_html: bool = False) -> str:
        """
        Sanitize string input by removing dangerous content.
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow safe HTML tags
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If dangerous content is detected
        """
        if not isinstance(value, str):
            raise ValidationError("Value must be a string")
            
        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning("SQL injection attempt detected", pattern=pattern, value=value[:100])
                raise SecurityError(f"Potentially dangerous SQL pattern detected")
                
        # Check for XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning("XSS attempt detected", pattern=pattern, value=value[:100])
                raise SecurityError(f"Potentially dangerous XSS pattern detected")
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', value.strip())
        
        # Handle HTML content
        if allow_html:
            # Use bleach to sanitize HTML
            sanitized = bleach.clean(
                sanitized,
                tags=cls.ALLOWED_HTML_TAGS,
                attributes=cls.ALLOWED_HTML_ATTRIBUTES,
                strip=True
            )
        else:
            # Escape HTML entities
            sanitized = html.escape(sanitized)
            
        # Apply length limit
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            
        return sanitized
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal and other attacks.
        
        Args:
            filename: Original filename
            
        Returns:
            Safe filename
        """
        if not filename:
            raise ValidationError("Filename cannot be empty")
            
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure it's not empty after sanitization
        if not filename:
            raise ValidationError("Filename invalid after sanitization")
            
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = f"{name[:250]}.{ext}" if ext else filename[:255]
            
        return filename
    
    @classmethod
    def sanitize_url(cls, url: str) -> str:
        """
        Sanitize URL to prevent SSRF and other attacks.
        
        Args:
            url: URL to sanitize
            
        Returns:
            Sanitized URL
            
        Raises:
            SecurityError: If URL is potentially dangerous
        """
        if not url:
            raise ValidationError("URL cannot be empty")
            
        # Parse URL
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            raise ValidationError("Invalid URL format")
            
        # Check scheme
        allowed_schemes = ['http', 'https']
        if parsed.scheme.lower() not in allowed_schemes:
            raise SecurityError(f"URL scheme '{parsed.scheme}' not allowed")
            
        # Check for localhost and private IP ranges
        if parsed.hostname:
            hostname = parsed.hostname.lower()
            
            # Block localhost
            if hostname in ['localhost', '127.0.0.1', '::1']:
                raise SecurityError("Localhost URLs not allowed")
                
            # Block private IP ranges (basic check)
            if any(hostname.startswith(prefix) for prefix in ['10.', '192.168.', '172.']):
                raise SecurityError("Private IP ranges not allowed")
                
        return url


class InputValidator:
    """Input validation utilities with domain-specific rules."""
    
    @staticmethod
    def validate_plant_name(name: str) -> ValidationResult:
        """Validate plant name."""
        result = ValidationResult()
        
        try:
            sanitized = InputSanitizer.sanitize_string(name, max_length=100)
            
            if not non_empty_string(sanitized):
                result.add_error("Plant name cannot be empty")
            elif len(sanitized) < 2:
                result.add_error("Plant name must be at least 2 characters")
            elif not re.match(r'^[a-zA-Z0-9\s\-_\'\.]+$', sanitized):
                result.add_error("Plant name contains invalid characters")
            else:
                result.sanitized_value = sanitized
                
        except (ValidationError, SecurityError) as e:
            result.add_error(str(e))
            
        return result
    
    @staticmethod
    def validate_sensor_reading(value: float, sensor_type: str) -> ValidationResult:
        """Validate sensor reading based on type."""
        result = ValidationResult()
        
        if not isinstance(value, (int, float)):
            result.add_error("Sensor value must be numeric")
            return result
            
        # Type-specific validation
        if sensor_type == 'moisture':
            if not valid_moisture_level(value):
                result.add_error("Moisture level must be between 0 and 100")
            else:
                result.sanitized_value = round(float(value), 2)
        elif sensor_type == 'temperature':
            if not valid_temperature(value):
                result.add_error("Temperature must be between -50°C and 80°C")
            else:
                result.sanitized_value = round(float(value), 1)
        elif sensor_type == 'humidity':
            if not valid_humidity(value):
                result.add_error("Humidity must be between 0 and 100")
            else:
                result.sanitized_value = round(float(value), 2)
        elif sensor_type == 'ph':
            if not valid_ph_level(value):
                result.add_error("pH level must be between 0 and 14")
            else:
                result.sanitized_value = round(float(value), 2)
        elif sensor_type == 'light':
            if value < 0 or value > 100000:  # Lux range
                result.add_error("Light level must be between 0 and 100000 lux")
            else:
                result.sanitized_value = round(float(value), 0)
        else:
            result.add_error(f"Unknown sensor type: {sensor_type}")
            
        return result
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        """Validate email address."""
        result = ValidationResult()
        
        try:
            sanitized = InputSanitizer.sanitize_string(email.lower(), max_length=254)
            
            if not valid_email(sanitized):
                result.add_error("Invalid email format")
            else:
                result.sanitized_value = sanitized
                
        except (ValidationError, SecurityError) as e:
            result.add_error(str(e))
            
        return result
    
    @staticmethod
    def validate_uuid(uuid_str: str) -> ValidationResult:
        """Validate UUID string."""
        result = ValidationResult()
        
        try:
            uuid_obj = UUID(uuid_str)
            result.sanitized_value = str(uuid_obj)
        except (ValueError, TypeError):
            result.add_error("Invalid UUID format")
            
        return result
    
    @staticmethod
    def validate_positive_number(value: Union[int, float, str], field_name: str = "value") -> ValidationResult:
        """Validate positive number."""
        result = ValidationResult()
        
        try:
            if isinstance(value, str):
                # Try to convert string to number
                if '.' in value:
                    num_value = float(value)
                else:
                    num_value = int(value)
            else:
                num_value = value
                
            if num_value <= 0:
                result.add_error(f"{field_name} must be positive")
            else:
                result.sanitized_value = num_value
                
        except (ValueError, TypeError):
            result.add_error(f"{field_name} must be a valid number")
            
        return result
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float], field_name: str = "value") -> ValidationResult:
        """Validate value within range."""
        result = ValidationResult()
        
        if not isinstance(value, (int, float)):
            result.add_error(f"{field_name} must be numeric")
            return result
            
        if value < min_val or value > max_val:
            result.add_error(f"{field_name} must be between {min_val} and {max_val}")
        else:
            result.sanitized_value = value
            
        return result
    
    @staticmethod
    def validate_datetime_string(dt_str: str) -> ValidationResult:
        """Validate datetime string in ISO format."""
        result = ValidationResult()
        
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            result.sanitized_value = dt
        except ValueError:
            result.add_error("Invalid datetime format. Use ISO 8601 format")
            
        return result


class RequestValidator:
    """High-level request validation for API endpoints."""
    
    @staticmethod
    def validate_plant_creation_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate plant creation request data.
        
        Args:
            data: Request data dictionary
            
        Returns:
            Sanitized and validated data
            
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        sanitized_data = {}
        
        # Validate required fields
        required_fields = ['name', 'species_id']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                
        # Validate plant name
        if 'name' in data:
            name_result = InputValidator.validate_plant_name(data['name'])
            if name_result.is_valid:
                sanitized_data['name'] = name_result.sanitized_value
            else:
                errors.extend(name_result.errors)
                
        # Validate species ID
        if 'species_id' in data:
            species_result = InputValidator.validate_uuid(str(data['species_id']))
            if species_result.is_valid:
                sanitized_data['species_id'] = UUID(species_result.sanitized_value)
            else:
                errors.extend([f"species_id: {e}" for e in species_result.errors])
                
        # Validate optional location data
        if 'location' in data and data['location']:
            location_data = data['location']
            sanitized_location = {}
            
            # Validate room
            if 'room' in location_data:
                try:
                    room = InputSanitizer.sanitize_string(location_data['room'], max_length=50)
                    if non_empty_string(room):
                        sanitized_location['room'] = room
                    else:
                        errors.append("Room name cannot be empty")
                except (ValidationError, SecurityError) as e:
                    errors.append(f"Room: {str(e)}")
                    
            # Validate position
            if 'position' in location_data:
                try:
                    position = InputSanitizer.sanitize_string(location_data['position'], max_length=100)
                    sanitized_location['position'] = position
                except (ValidationError, SecurityError) as e:
                    errors.append(f"Position: {str(e)}")
                    
            # Validate light level
            if 'light_level' in location_data:
                light_result = InputValidator.validate_range(
                    location_data['light_level'], 0, 10, "light_level"
                )
                if light_result.is_valid:
                    sanitized_location['light_level'] = light_result.sanitized_value
                else:
                    errors.extend([f"light_level: {e}" for e in light_result.errors])
                    
            if sanitized_location:
                sanitized_data['location'] = sanitized_location
                
        # Validate optional notes
        if 'notes' in data and data['notes']:
            try:
                notes = InputSanitizer.sanitize_string(data['notes'], max_length=1000, allow_html=True)
                sanitized_data['notes'] = notes
            except (ValidationError, SecurityError) as e:
                errors.append(f"Notes: {str(e)}")
                
        if errors:
            raise ValidationError(f"Validation failed: {'; '.join(errors)}")
            
        return sanitized_data
    
    @staticmethod
    def validate_sensor_reading_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate sensor reading request data.
        
        Args:
            data: Request data dictionary
            
        Returns:
            Sanitized and validated data
            
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        sanitized_data = {}
        
        # Validate required fields
        required_fields = ['sensor_id', 'value', 'sensor_type']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                
        # Validate sensor ID
        if 'sensor_id' in data:
            sensor_result = InputValidator.validate_uuid(str(data['sensor_id']))
            if sensor_result.is_valid:
                sanitized_data['sensor_id'] = UUID(sensor_result.sanitized_value)
            else:
                errors.extend([f"sensor_id: {e}" for e in sensor_result.errors])
                
        # Validate sensor type
        if 'sensor_type' in data:
            allowed_types = ['moisture', 'temperature', 'humidity', 'ph', 'light']
            sensor_type = str(data['sensor_type']).lower()
            if sensor_type in allowed_types:
                sanitized_data['sensor_type'] = sensor_type
            else:
                errors.append(f"Invalid sensor type. Must be one of: {', '.join(allowed_types)}")
                
        # Validate sensor value
        if 'value' in data and 'sensor_type' in sanitized_data:
            value_result = InputValidator.validate_sensor_reading(
                data['value'], sanitized_data['sensor_type']
            )
            if value_result.is_valid:
                sanitized_data['value'] = value_result.sanitized_value
            else:
                errors.extend([f"value: {e}" for e in value_result.errors])
                
        # Validate optional timestamp
        if 'timestamp' in data:
            ts_result = InputValidator.validate_datetime_string(data['timestamp'])
            if ts_result.is_valid:
                sanitized_data['timestamp'] = ts_result.sanitized_value
            else:
                errors.extend([f"timestamp: {e}" for e in ts_result.errors])
                
        if errors:
            raise ValidationError(f"Validation failed: {'; '.join(errors)}")
            
        return sanitized_data


def validate_and_sanitize(validator_func: Callable) -> Callable:
    """
    Decorator to apply validation and sanitization to request data.
    
    Args:
        validator_func: Function that validates and sanitizes data
        
    Returns:
        Decorator function
    """
    def decorator(endpoint_func):
        @functools.wraps(endpoint_func)
        async def wrapper(*args, **kwargs):
            # Find request data in kwargs
            request_data = None
            for key, value in kwargs.items():
                if hasattr(value, 'dict') and callable(value.dict):  # Pydantic model
                    request_data = value.dict()
                    break
                elif isinstance(value, dict):
                    request_data = value
                    break
                    
            if request_data:
                try:
                    # Validate and sanitize
                    sanitized_data = validator_func(request_data)
                    
                    # Replace original data with sanitized version
                    for key, value in kwargs.items():
                        if hasattr(value, 'dict') and callable(value.dict):
                            # Update Pydantic model fields
                            for field, sanitized_value in sanitized_data.items():
                                if hasattr(value, field):
                                    setattr(value, field, sanitized_value)
                            break
                        elif isinstance(value, dict):
                            kwargs[key] = sanitized_data
                            break
                            
                except ValidationError as e:
                    logger.warning("Request validation failed", error=str(e))
                    raise
                    
            return await endpoint_func(*args, **kwargs)
        return wrapper
    return decorator