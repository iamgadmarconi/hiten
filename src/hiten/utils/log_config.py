"""Logging configuration utilities for the hiten package.

This module provides centralized logging configuration for the entire hiten
package, including console and file output with timestamped log files.

Notes
-----
Logging is automatically configured when this module is imported.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(level: int = logging.INFO, 
                  format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                  save_to_file: bool = True, 
                  log_dir: str = r'results/logs') -> None:
    """
    Configure logging to stdout and optionally to a file.
    
    This function sets up the root logger with both console and file handlers.
    It clears any existing handlers to avoid duplicates and creates timestamped
    log files when file logging is enabled.
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR).
    format_string : str, default '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        Format string for log messages using standard logging format codes.
    save_to_file : bool, default True
        Whether to save logs to a file in addition to console output.
    log_dir : str, default 'results/logs'
        Directory path where log files will be saved. Will be created if it doesn't exist.
        
    Notes
    -----
    The function automatically creates the log directory if it doesn't exist.
    Log files are named with timestamps in the format 'run_YYYYMMDD_HHMMSS.txt'.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if save_to_file:
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"run_{timestamp}.txt"
        log_filepath = os.path.join(log_dir, log_filename)
        
        # Create file handler
        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Log the log file location
        root_logger.info(f"Log file created: {log_filepath}")

# Setup logging when this module is imported
# Check environment variable to control file logging
save_to_file = os.getenv('HITEN_LOG_TO_FILE', 'false').lower() in ('true', '1', 'yes')
setup_logging(save_to_file=save_to_file)

# Create a logger instance for other modules to import
logger = logging.getLogger(__name__)
"""Logger instance for use throughout the hiten package.

This logger is automatically configured with the settings from setup_logging()
and can be imported by other modules for consistent logging across the package.
"""
