import logging
import sys
from pathlib import Path

def setup_logger():
    """Sets up a singleton logger instance."""
    # Define log path relative to this file: root/logs/app.log
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"

    logger_instance = logging.getLogger("smart_chatbot")
    
    # Avoid adding handlers multiple times if the function is called twice
    if not logger_instance.handlers:
        logger_instance.setLevel(logging.INFO)

        # Standard English format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File Handler
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger_instance.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")

        # Console Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger_instance.addHandler(stream_handler)
    
    return logger_instance

logger = setup_logger()