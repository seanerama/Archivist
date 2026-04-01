"""Structured logging configuration for Archivist."""

from __future__ import annotations

import sys

import structlog
from structlog._log_levels import NAME_TO_LEVEL

_LEVEL_MAP: dict[str, int] = {k.upper(): v for k, v in NAME_TO_LEVEL.items()}


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with rich console output (TTY) or JSON (non-TTY).

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    log_level = _LEVEL_MAP.get(level.upper(), 20)  # default to INFO

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if sys.stderr.isatty():
        # Pretty colored output for interactive terminals
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON lines for piped/CI output
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger, optionally bound to a name.

    Args:
        name: Optional logger name for context.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    if name:
        logger = logger.bind(component=name)
    return logger
