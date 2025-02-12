import os
import logging

def create_logger() -> logging.Logger:
    """Create a logger with the level specified in LOG_LEVEL environment variable."""
    logger = logging.getLogger("openserv-agent")
    
    # Get log level from environment variable, default to DEBUG during troubleshooting
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    
    # Set log level
    logger.setLevel(getattr(logging, log_level, logging.DEBUG))
    
    # Create console handler with formatter
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

# Create default logger instance
logger = create_logger() 
