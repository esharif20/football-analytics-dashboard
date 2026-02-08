"""Professional pipeline output formatting."""

import sys
import time as _time

from tqdm import tqdm as _tqdm
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

# Detect if we're running in a real terminal vs a subprocess pipe.
# In a pipe (worker subprocess), tqdm can't do in-place updates, so every
# progress tick becomes a new log line (750 lines for ball detection!).
# Use a sparse log-based reporter instead.
_IS_TTY = sys.stdout.isatty()

# Visual tqdm bar format — only used in interactive terminals
_TTY_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%"
    f"{DIM}|{RESET}{{bar:30}}{DIM}|{RESET} "
    "{n_fmt}/{total_fmt} "
    f"{DIM}[{{elapsed}}<{{remaining}}, {{rate_fmt}}]{RESET}"
)


def _fmt_time(secs: float) -> str:
    """Format seconds into MM:SS or H:MM:SS."""
    m, s = divmod(int(secs), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class _LogProgress:
    """Milestone-based progress for non-TTY environments (subprocess pipes).

    Instead of printing 750 tqdm lines, logs only at 10% milestones plus a
    clean completion summary. Supports both iterator and context-manager usage.
    """

    _INTERVAL = 10  # percent between log lines

    def __init__(self, iterable=None, *, desc="Processing", unit="it", total=None, **_kw):
        self.iterable = iterable
        self.desc = desc.strip()
        self.unit = unit
        self.total = total
        if self.total is None and iterable is not None and hasattr(iterable, "__len__"):
            self.total = len(iterable)
        self.n = 0
        self._start = None
        self._last_milestone = -1

    # -- context manager (manual .update()) -----------------------------------

    def __enter__(self):
        self._start = _time.time()
        return self

    def __exit__(self, *exc):
        self._log_complete()
        return False

    # -- iterator -------------------------------------------------------------

    def __iter__(self):
        self._start = _time.time()
        for item in self.iterable:
            yield item
            self.n += 1
            self._check_milestone()
        self._log_complete()

    def __len__(self):
        return self.total or 0

    # -- tqdm-compatible API --------------------------------------------------

    def update(self, n=1):
        self.n += n
        self._check_milestone()

    def set_postfix(self, **_kw):
        pass  # no-op in log mode

    def close(self):
        pass

    # -- internal -------------------------------------------------------------

    def _check_milestone(self):
        if not self.total:
            return
        pct = int(100 * self.n / self.total)
        milestone = (pct // self._INTERVAL) * self._INTERVAL
        if milestone > self._last_milestone and milestone < 100:
            self._last_milestone = milestone
            elapsed = _time.time() - self._start if self._start else 0
            rate = self.n / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.n) / rate if rate > 0 else 0
            logger.info(
                f"  {self.desc}: {DIM}{milestone:3d}%{RESET}  "
                f"{DIM}({self.n}/{self.total}) "
                f"[{_fmt_time(elapsed)}<{_fmt_time(remaining)}, "
                f"{rate:.1f} {self.unit}/s]{RESET}"
            )

    def _log_complete(self):
        elapsed = _time.time() - self._start if self._start else 0
        rate = self.n / elapsed if elapsed > 0 else 0
        logger.info(
            f"  {GREEN}✓{RESET} {self.desc}: "
            f"{BOLD}{self.n}{RESET}/{self.total} "
            f"{DIM}in {_fmt_time(elapsed)} ({rate:.1f} {self.unit}/s){RESET}"
        )


def progress(iterable=None, *, desc="  Processing", unit="it", total=None, **kwargs):
    """Create a progress reporter.

    In a real terminal: visual tqdm bar with in-place updates.
    In a subprocess pipe: sparse log-only output at 10% milestones.
    """
    if _IS_TTY:
        return _tqdm(
            iterable,
            desc=desc,
            unit=unit,
            total=total,
            bar_format=_TTY_BAR_FORMAT,
            delay=0.5,
            **kwargs,
        )
    return _LogProgress(iterable, desc=desc, unit=unit, total=total, **kwargs)


# Kept for backwards compat — prefer progress()
BAR_FORMAT = _TTY_BAR_FORMAT


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
