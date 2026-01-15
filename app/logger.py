from pathlib import Path
import os 
import logging

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = Path(root) / 'logs'
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str, log_file: str = 'app.log') -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )

    file_handler = logging.FileHandler(LOG_DIR / log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger