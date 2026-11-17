
import os
import logging

from logging.handlers import RotatingFileHandler

from src.config import ROOT_DIR


LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

APP_LOGGER_NAME = "hybrid_rag"


class ApplicationConsoleHandler(logging.StreamHandler):
    """Dedicated console handler owned by the Hybrid RAG application."""


def configure_logging():
    """
    Configure centralized application logging.

    Logs are written to:
    1. Console
    2. Rotating log files
    """

    # 1. Create log directory.
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Read logging level.
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, None)

    if not isinstance(log_level, int):
        raise ValueError(f"INVALID_LOG_LEVEL: {log_level_name}")

    # 3. Create application logger.
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(log_level)
    logger.propagate = False

    # 4. Configure formatter.
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 5. Identify only our application console handler.
    has_console_handler = any(
        isinstance(handler, ApplicationConsoleHandler)
        for handler in logger.handlers
    )

    if not has_console_handler:
        console_handler = ApplicationConsoleHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 6. Identify rotating file handler.
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
    """Return a named child logger for an application module."""

    configure_logging()

    return logging.getLogger(
        f"{APP_LOGGER_NAME}.{name}"
    )

