import logging
import os
import sys
from datetime import datetime

from hiten.utils.files import _ensure_dir


def setup_logging(level=logging.INFO, format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                  save_to_file=True, log_dir=r'results/logs'):
    """
    Configures logging to stdout and optionally to a file.
    
    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO)
    format_string : str
        Format string for log messages
    save_to_file : bool
        Whether to save logs to a file (default: True)
    log_dir : str
        Directory to save log files (default: 'results')
    """
    _ensure_dir(log_dir)

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
        log_filename = f"run_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        
        # Create file handler
        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Log the log file location
        root_logger.info(f"Log file created: {log_filepath}")

# Setup logging when this module is imported
setup_logging()

# Create a logger instance for other modules to import
logger = logging.getLogger(__name__)
