"""Professional pipeline output formatting."""

from utils.logging_config import get_logger, LogContext

# ANSI escape codes for styling
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"

logger = get_logger("pipeline")


def banner(title: str) -> None:
    """Print a stage banner with box drawing."""
    width = 60
    logger.info("")
    logger.info(f"{CYAN}{'━' * width}{RESET}")
    logger.info(f"{CYAN}┃{RESET} {BOLD}{WHITE}{title.upper()}{RESET}")
    logger.info(f"{CYAN}{'━' * width}{RESET}")


def stage(name: str) -> LogContext:
    """Return a LogContext for timing a pipeline stage."""
    return LogContext(logger, name)


def config_table(title: str, items: dict) -> None:
    """Print a formatted config/info table."""
    logger.info(f"  {DIM}┌─ {title}{RESET}")
    for key, val in items.items():
        logger.info(f"  {DIM}│{RESET}  {key:<22} {BOLD}{val}{RESET}")
    logger.info(f"  {DIM}└{'─' * 30}{RESET}")


def metric(label: str, value, unit: str = "") -> None:
    """Print a single metric line."""
    if isinstance(value, float):
        logger.info(f"  {GREEN}✓{RESET} {label}: {BOLD}{value:.2f}{RESET} {unit}")
    else:
        logger.info(f"  {GREEN}✓{RESET} {label}: {BOLD}{value}{RESET} {unit}")


def warn(msg: str) -> None:
    """Print a warning."""
    logger.warning(f"{YELLOW}⚠{RESET} {msg}")


def error(msg: str) -> None:
    """Print an error."""
    logger.error(f"✗ {msg}")


def divider() -> None:
    """Print a thin divider."""
    logger.info(f"  {DIM}{'─' * 50}{RESET}")
