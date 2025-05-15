import logging
import os
from logging.handlers import RotatingFileHandler

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(log_name="training_log", log_dir="logs", sampled=False, maxBytes=10*1024*1024, backupCount=5):
    """
    Sets up a logger with a rotating file handler.

    Args:
        log_name (str): Base name for the logger.
        log_dir (str): Directory where the log file will be saved.
        sampled (bool): If True, appends '_sampled' to the log name.
        maxBytes (int): Maximum file size (in bytes) before rotation (default 10 MB).
        backupCount (int): Number of backup files to keep (default 5).

    Returns:
        logger: The configured logger instance.
    """
    if sampled:
        log_name = f"{log_name}_sampled"
        
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}.log")
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers (if re-running in an interactive session)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console Handler (logs INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    
    # Rotating File Handler (logs DEBUG and above)
    file_handler = RotatingFileHandler(log_file, maxBytes=maxBytes, backupCount=backupCount)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    
    # Add Handlers to Logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
