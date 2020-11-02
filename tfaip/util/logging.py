import os
import logging

# Initialize logging
FORMAT = '{levelname:<8s} {asctime} {name:>30.30s}: {message}'
formatter = logging.Formatter(FORMAT, style='{')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(formatter)

for handler in logging.getLogger("tensorflow").handlers:
    handler.setFormatter(formatter)


def setup_log(log_dir, append, log_name='train.log'):
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, log_name)
    file_handler = logging.FileHandler(filename, 'a' if append else 'w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=logging.INFO)
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging training progress to '{filename}'")
