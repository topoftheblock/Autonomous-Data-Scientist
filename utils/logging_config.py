# agent/utils/logging_config.py
"""
Centralized logging configuration for the Autonomous Data Science Agent.
Call `setup_logging()` early in `main.py` to apply settings from config or env.
"""

import logging
import sys
import os
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_format: Standard logging format string.
        log_file: Optional path to a log file; logs to stderr by default.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Remove any existing handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Prepare formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)

    # Optional file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

    logging.root.setLevel(numeric_level)

    # Quiet noisy third‑party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging configured: level=%s, file=%s", level, log_file or "stderr")


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (convenience wrapper)."""
    return logging.getLogger(name)