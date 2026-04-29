"""Logging utilities for project scripts."""

import os
import logging
from datetime import datetime

# Module-level logger (will be configured when setup_logging is called)
logger = logging.getLogger(__name__)


def setup_logging(script_path, log_level=logging.INFO):
    """Setup logging configuration with file and console handlers.
    
    Args:
        script_path: Path to the script file (typically __file__)
        log_level: Logging level (default: logging.INFO)
    
    Returns:
        str: Path to the log file
    """
    script_dir = os.path.dirname(os.path.abspath(script_path))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,  # Root logger level matches console output level
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ],
        force=True  # Override any existing configuration
    )
    # Configure the module-level logger
    logger.setLevel(log_level)
    logger.info("Logging to: %s", log_file_path)
    
    return log_file_path

