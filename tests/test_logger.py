import pytest
import logging
from unittest.mock import patch
from src.logger import create_logger, logger

def test_create_logger_with_default_level():
    with patch.dict('os.environ', {}, clear=True):  # Clear LOG_LEVEL from env
        test_logger = create_logger()
        assert test_logger.level == logging.INFO

def test_create_logger_with_custom_level():
    with patch.dict('os.environ', {'LOG_LEVEL': 'DEBUG'}, clear=True):
        test_logger = create_logger()
        assert test_logger.level == logging.DEBUG

def test_create_logger_with_null_level():
    with patch.dict('os.environ', {}, clear=True):  # Clear LOG_LEVEL from env
        test_logger = create_logger()
        assert test_logger.level == logging.INFO

def test_default_logger_instance():
    assert logger is not None
    assert hasattr(logger, 'info')
    assert hasattr(logger, 'error')
    assert hasattr(logger, 'warning')
    assert hasattr(logger, 'debug')
    assert callable(logger.info)
    assert callable(logger.error)
    assert callable(logger.warning)
    assert callable(logger.debug) 
