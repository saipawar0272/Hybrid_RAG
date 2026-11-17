import logging

from logging.handlers import RotatingFileHandler

import pytest

import src.logging_config as logging_config


@pytest.fixture
def isolated_logger(monkeypatch, tmp_path):
    """Provide an isolated logger configuration for each test."""

    logger = logging.getLogger(logging_config.APP_LOGGER_NAME)

    #save existing handler

    original_handlers = logger.handlers[:]

    #remove existing handlers
    for handler in original_handlers:
        logger.removeHandler(handler)


    #Use temporary log directory
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(
        logging_config,
        "LOG_FILE",
        tmp_path / "test.log"
    )

    yield logger

    #clean up handlers created during the test
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


    #restore original handler

    for handler in original_handlers:
        logger.addHandler(handler)


def test_configure_logging_creates_log_directory(
        isolated_logger,
        tmp_path,
        monkeypatch
):
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    logging_config.configure_logging()


    assert tmp_path.exists()
    assert isolated_logger.level == logging.INFO


def test_configure_logging_attaches_console_handler(
        isolated_logger,
        monkeypatch
):
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    logger = logging_config.configure_logging()

    console_handlers = [
        handler
        for handler in logger.handlers
        if type(handler) is logging.StreamHandler
        # if isinstance(handler, logging.StreamHandler)
        # and not isinstance(handler, RotatingFileHandler)
    ]

    assert len(console_handlers) == 1


def test_configure_logging_attaches_rotating_file_handler(
        isolated_logger,
        monkeypatch
):
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    logger = logging_config.configure_logging()

    file_handlers = [
        handler 
        for handler in logger.handlers
        if isinstance(handler, RotatingFileHandler)
    ]

    assert len(file_handlers) == 1


def test_configure_logging_does_not_duplicate_handlers(
        isolated_logger,
        monkeypatch
):
    monkeypatch.setenv("LOG_LEVEL", "INFO")


    logging_config.configure_logging()
    logging_config.configure_logging()

    application_handlers = [
        handler 
        for handler in isolated_logger.handlers
        if (
            type(handler) is logging.StreamHandler
            or isinstance(handler, RotatingFileHandler)
        )
    ]

    assert len(application_handlers) ==2


def test_configure_logging_rejects_invalid_level(
        isolated_logger,
        monkeypatch
):
    monkeypatch.setenv("LOG_LEVEL", "INVALID")

    with pytest.raises(ValueError, match="INVALID_LOG_LEVEL") :
        logging_config.configure_logging()


def test_get_logger_returns_named_child_logger(
        isolated_logger,
        monkeypatch
):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    logger = logging_config.get_logger("test_module")

    assert logger.name == "hybrid_rag.test_module"
    assert logger.getEffectiveLevel() == logging.DEBUG