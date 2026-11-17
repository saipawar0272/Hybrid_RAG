import os
import logging

from pathlib import Path
from logging.handlers import RotatingFileHandler

from src.config import ROOT_DIR

#------------------

#Logging configuration

#----------------------

LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

APP_LOGGER_NAME = "hybrid_rag"


def configure_logging():

    """ 
    configure centralized application logging.

    Logs are written to both:
    1. Console
    2. Rotating log files
    """

    # 1.creats log directory if it doesn't exists

    LOG_DIR.mkdir(parents=True, exist_ok =True)

    # 2. Read logging level from environment

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()

    log_level =  getattr(logging, log_level_name, None)

    if not isinstance(log_level, int) :
        raise ValueError(
            f"INVALID_LOG_LEVEL: {log_level_name}"
        )

    # 3. create application logger

    logger = logging.getLogger(APP_LOGGER_NAME)

    logger.setLevel(log_level)

    # prevent duplicate logging through logger

    logger.propagate = False


    # 4. Configure handler only once


# 4. Configure application handlers independently.
# Do not mistake pytest's capture handlers for application handlers.

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add console handler only if our application console handler is absent.
    has_console_handler = any(
        type(handler) is logging.StreamHandler
        for handler in logger.handlers
    )

    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add rotating file handler only if one is absent.
    has_file_handler = any(
        isinstance(handler, RotatingFileHandler)
        for handler in logger.handlers
    )

    if not has_file_handler:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


    return logger


def get_logger(name: str):

    """
    Return a named child logger for an application module.

    """

    configure_logging()

    return logging.getLogger(
        f"{APP_LOGGER_NAME}.{name}"
    )


