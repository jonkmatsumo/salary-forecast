import unittest
import logging
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.utils.logger import setup_logging, get_logger


class TestSetupLogging(unittest.TestCase):
    """Tests for setup_logging function."""
    
    def setUp(self):
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)
    
    def test_setup_logging_default_level(self):
        """Verify default logging configuration works for standard application use."""
        setup_logging()
        
        self.assertEqual(logging.root.level, logging.INFO)
        
        handlers = logging.root.handlers
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in handlers))
    
    def test_setup_logging_custom_level(self):
        """Verify logging level can be customized for different environments."""
        setup_logging(level=logging.DEBUG)
        self.assertEqual(logging.root.level, logging.DEBUG)
        
        setup_logging(level=logging.WARNING)
        self.assertEqual(logging.root.level, logging.WARNING)
    
    def test_setup_logging_with_file(self):
        """Verify file logging works for persistent log storage."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp_file:
            log_file_path = tmp_file.name
        
        try:
            setup_logging(log_file=log_file_path)
            
            handlers = logging.root.handlers
            has_stream = any(isinstance(h, logging.StreamHandler) for h in handlers)
            has_file = any(isinstance(h, logging.FileHandler) for h in handlers)
            
            self.assertTrue(has_stream, "StreamHandler should be present")
            self.assertTrue(has_file, "FileHandler should be present")
            
            file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
            self.assertEqual(file_handlers[0].baseFilename, log_file_path)
        finally:
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)
    
    def test_setup_logging_force_reconfigure(self):
        """Verify logging can be reconfigured when application settings change."""
        setup_logging(level=logging.INFO)
        initial_handlers_count = len(logging.root.handlers)
        
        setup_logging(level=logging.DEBUG)
        
        self.assertEqual(logging.root.level, logging.DEBUG)
    
    def test_setup_logging_format(self):
        """Verify logging format enables proper log message structure."""
        setup_logging()
        
        logger = logging.getLogger("test_logger")
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        logger.info("Test message")
        
        self.assertTrue(logger.isEnabledFor(logging.INFO))


class TestGetLogger(unittest.TestCase):
    """Tests for get_logger function."""
    
    def test_get_logger_returns_logger(self):
        """Verify get_logger returns properly configured logger instances."""
        logger = get_logger("test_module")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_module")
    
    def test_get_logger_different_names(self):
        """Verify separate loggers are created for different modules."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        self.assertNotEqual(logger1, logger2)
        self.assertEqual(logger1.name, "module1")
        self.assertEqual(logger2.name, "module2")
    
    def test_get_logger_same_name_returns_same_logger(self):
        """Verify singleton behavior prevents duplicate logger instances."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        
        self.assertIs(logger1, logger2)
    
    def test_get_logger_inherits_root_config(self):
        """Verify child loggers inherit root logger configuration."""
        setup_logging(level=logging.DEBUG)
        
        logger = get_logger("test_module")
        
        self.assertTrue(logger.isEnabledFor(logging.DEBUG))
    
    def test_get_logger_can_log_messages(self):
        """Verify returned logger is functional and doesn't raise exceptions."""
        setup_logging(level=logging.INFO)
        
        logger = get_logger("test_module")
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        self.assertTrue(logger.isEnabledFor(logging.INFO))


class TestLoggerIntegration(unittest.TestCase):
    """Integration tests for logger module."""
    
    def setUp(self):
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)
    
    def test_setup_and_get_logger_work_together(self):
        """Verify setup and get functions integrate correctly for application initialization."""
        setup_logging(level=logging.INFO)
        logger = get_logger("integration_test")
        
        self.assertTrue(logger.isEnabledFor(logging.INFO))
        self.assertFalse(logger.isEnabledFor(logging.DEBUG))
    
    def test_multiple_loggers_with_file(self):
        """Verify multiple loggers can write to the same log file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp_file:
            log_file_path = tmp_file.name
        
        try:
            setup_logging(level=logging.INFO, log_file=log_file_path)
            
            logger1 = get_logger("module1")
            logger2 = get_logger("module2")
            
            logger1.info("Message from module1")
            logger2.info("Message from module2")
            
            self.assertTrue(os.path.exists(log_file_path))
            with open(log_file_path, 'r') as f:
                content = f.read()
                self.assertIn("module1", content)
                self.assertIn("module2", content)
        finally:
            if os.path.exists(log_file_path):
                os.unlink(log_file_path)

