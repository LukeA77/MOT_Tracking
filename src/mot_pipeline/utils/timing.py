"""Lightweight timing utilities for the speed-benchmark harness."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Elapsed:
    """Mutable container for the elapsed wall-clock time of a ``with`` block, in seconds."""

    seconds: float = 0.0


@contextmanager
def timer() -> Iterator[Elapsed]:
    """Context manager measuring wall-clock elapsed time of the enclosed block.

    Yields:
        An :class:`Elapsed` instance whose ``seconds`` field is populated
        once the ``with`` block exits.
    """
    result = Elapsed()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start
