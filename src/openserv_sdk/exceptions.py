"""
Custom exceptions for the OpenServ Agent library.
"""

class OpenServError(Exception):
    """Base exception class for OpenServ SDK."""
    pass

class ConfigurationError(OpenServError):
    """Raised when there is a configuration error."""
    pass

class APIError(OpenServError):
    """Raised when there is an API error."""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class AuthenticationError(APIError):
    """Raised when there is an authentication error."""
    pass

class ToolError(OpenServError):
    """Raised when there is an error executing a tool."""
    def __init__(self, tool_name: str, message: str, original_error: Exception = None):
        super().__init__(f"Error in tool '{tool_name}': {message}")
        self.tool_name = tool_name
        self.original_error = original_error

class ValidationError(OpenServError):
    """Raised when there is a validation error."""
    pass

class RuntimeError(OpenServError):
    """Raised when there is a runtime error."""
    pass 
