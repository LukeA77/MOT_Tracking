"""Shared pytest fixtures: repo paths and a synthetic MOT17 sequence."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def configs_dir(repo_root: Path) -> Path:
    """Absolute path to the ``configs/`` directory."""
    return repo_root / "configs"
