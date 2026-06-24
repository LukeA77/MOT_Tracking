"""Centralised stdlib logging configuration.

Library code must never call ``print()``; it should call :func:`get_logger`
and log through the returned logger so verbosity is controlled in one place.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED_LOGGERS: set[str] = set()
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger, attaching a stdout handler exactly once.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).

    Returns:
        A :class:`logging.Logger` with a single stream handler attached.
    """
    logger = logging.getLogger(name)
    if name not in _CONFIGURED_LOGGERS:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)
    logger.setLevel(level.upper())
    return logger
