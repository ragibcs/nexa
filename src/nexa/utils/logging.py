"""Rich-based coloured logging + FFmpeg availability check."""

from __future__ import annotations

import shutil

from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the ``nexa`` logger with Rich output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logger = logging.getLogger("nexa")
    logger.setLevel(level)
    return logger


def check_ffmpeg() -> bool:
    """Print FFmpeg status and return availability."""
    available = shutil.which("ffmpeg") is not None
    if available:
        console.print("[green]FFmpeg found.[/]")
    else:
        console.print(
            "[yellow]FFmpeg not found — video processing will be limited.[/]"
        )
    return available
