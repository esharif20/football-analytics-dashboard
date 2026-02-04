"""Centralized logging configuration for the football analysis pipeline.

Usage:
    from utils.logging_config import get_logger, setup_logging

    # At application start (optional, auto-configured on first get_logger call)
    setup_logging(level="INFO", log_file="pipeline.log")

    # In any module
    logger = get_logger(__name__)
    logger.info("Processing frame %d", frame_idx)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO

# Singleton state
_logging_configured = False
_log_file_path: Optional[Path] = None


# =============================================================================
# Custom Formatter
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels for terminal output."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str = DEFAULT_FORMAT, datefmt: str = DEFAULT_DATE_FORMAT, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# =============================================================================
# Setup Functions
# =============================================================================

def setup_logging(
    level: str | int = DEFAULT_LEVEL,
    log_file: Optional[str | Path] = None,
    format_string: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    use_colors: bool = True,
) -> None:
    """Configure the root logger for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console.
        format_string: Log message format string.
        date_format: Date format string.
        use_colors: Whether to use colored output in terminal.
    """
    global _logging_configured, _log_file_path

    if _logging_configured:
        return

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LEVEL)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(format_string, date_format, use_colors))
    root_logger.addHandler(console_handler)

    # File handler (no colors)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string, date_format))
        root_logger.addHandler(file_handler)
        _log_file_path = log_path

    # Suppress noisy third-party loggers
    for noisy_logger in [
        "urllib3",
        "httpx",
        "httpcore",
        "transformers",
        "ultralytics",
        "PIL",
        "matplotlib",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Auto-configure if not done yet
    if not _logging_configured:
        setup_logging()

    # Shorten module paths for cleaner output
    if name.startswith("src."):
        name = name[4:]

    return logging.getLogger(name)


def get_log_file_path() -> Optional[Path]:
    """Get the current log file path, if configured."""
    return _log_file_path


# =============================================================================
# Context Managers
# =============================================================================

class LogContext:
    """Context manager for logging with timing and context info.

    Usage:
        with LogContext(logger, "Processing video", video_name=video):
            # ... processing code ...
    """

    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: Optional[datetime] = None

    def __enter__(self):
        self.start_time = datetime.now()
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        if context_str:
            self.logger.info(f"Starting: {self.operation} ({context_str})")
        else:
            self.logger.info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} (took {elapsed.total_seconds():.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} after {elapsed.total_seconds():.2f}s - {exc_val}")
        return False  # Don't suppress exceptions


# =============================================================================
# Utility Functions
# =============================================================================

def log_config_values(logger: logging.Logger, config_dict: dict, title: str = "Configuration") -> None:
    """Log configuration values in a formatted way.

    Args:
        logger: Logger instance
        config_dict: Dictionary of configuration values
        title: Title for the config block
    """
    logger.info(f"=== {title} ===")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")


def log_metrics(logger: logging.Logger, metrics: dict, title: str = "Metrics") -> None:
    """Log metrics in a formatted way.

    Args:
        logger: Logger instance
        metrics: Dictionary of metric values
        title: Title for the metrics block
    """
    logger.info(f"--- {title} ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
