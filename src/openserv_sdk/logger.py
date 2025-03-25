import os
import logging
import sys

def create_logger() -> logging.Logger:
    """
    Create a new logger instance.
    Matches the TypeScript pino logger implementation exactly.
    
    Returns:
        A configured logger instance
    """
    # Create logger with name 'openserv-agent'
    logger = logging.getLogger('openserv-agent')
    
    # Set level from env var or default to 'info'
    logger.setLevel(getattr(logging, (os.environ.get('LOG_LEVEL') or 'info').upper()))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Create default logger instance
logger = create_logger()