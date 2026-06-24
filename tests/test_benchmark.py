"""Tests for the runtime-agnostic inference speed benchmarking harness."""

from __future__ import annotations

import pytest

from mot_pipeline.benchmark.speed import benchmark_callable


def test_benchmark_callable_runs_warmup_and_returns_sane_result() -> None:
    calls: list[int] = []

    result = benchmark_callable(
        calls.append, frames=list(range(5)), warmup=2, runtime="dummy", device="cpu"
    )

    assert calls == [0, 1, 2, 3, 4]  # warmup frames still run, just aren't timed
    assert result.runtime == "dummy"
    assert result.device == "cpu"
    assert result.avg_ms_per_frame >= 0.0
    assert result.fps > 0.0


def test_benchmark_callable_times_only_post_warmup_frames() -> None:
    calls: list[int] = []

    benchmark_callable(
        calls.append, frames=list(range(10)), warmup=7, runtime="dummy", device="cpu"
    )

    assert len(calls) == 10


def test_benchmark_callable_requires_more_frames_than_warmup() -> None:
    with pytest.raises(ValueError, match="more than"):
        benchmark_callable(
            lambda frame: None, frames=[1, 2], warmup=2, runtime="dummy", device="cpu"
        )
