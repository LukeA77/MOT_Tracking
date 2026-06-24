"""MLflow run logging shared by the training, benchmark, and evaluation stages.

Every run logs the full resolved config, the current git commit SHA, and key
dependency versions, per the reproducibility requirements in the spec.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

from mot_pipeline.config import Config

_TRACKED_PACKAGES = ("ultralytics", "torch", "onnx", "onnxruntime", "opencv-python", "numpy")


def get_git_sha(repo_root: Path | None = None) -> str:
    """Return the current git commit SHA.

    Args:
        repo_root: Directory to run ``git`` in. Defaults to the current
            working directory.

    Returns:
        The full commit SHA, or ``"unknown"`` if not run inside a git
        repository (e.g. a fresh clone before the first commit, or git
        unavailable on the host).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def dependency_versions() -> dict[str, str]:
    """Return installed versions of the key runtime dependencies.

    Returns:
        A mapping from package name to version string, or ``"not installed"``
        for any tracked package missing from the current environment.
    """
    versions: dict[str, str] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


@contextmanager
def mlflow_run(run_name: str, config: Config) -> Iterator[Any]:
    """Start an MLflow run, logging the resolved config, git SHA, and dependency versions.

    Uses MLflow's local file backend (``./mlruns``) unless ``MLFLOW_TRACKING_URI``
    is set in the environment.

    Args:
        run_name: Human-readable run name.
        config: The resolved, validated pipeline configuration for this run.

    Yields:
        The active MLflow run, for callers that want to log additional
        metrics or artefacts inside the ``with`` block.
    """
    import mlflow

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten_config(config))
        mlflow.set_tag("git_sha", get_git_sha())
        for package, version in dependency_versions().items():
            mlflow.set_tag(f"dep.{package}", version)
        yield run


def _flatten_config(config: Config) -> dict[str, str]:
    """Flatten a nested config dict into dotted-key strings for ``mlflow.log_params``."""
    flat: dict[str, str] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                _walk(f"{prefix}.{key}" if prefix else str(key), child)
        else:
            flat[prefix] = str(value)

    _walk("", config.model_dump(mode="json"))
    return flat
