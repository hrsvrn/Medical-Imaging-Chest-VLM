from loguru import logger
import os

def setup_logger():
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/inference.log", rotation="1 MB")
    return logger
