"""Read/merge/write the consolidated ``outputs/metrics.json`` file.

Detection, benchmark, and evaluation stages each contribute one top-level
section to the same file rather than overwriting each other's results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> dict[str, Any]:
    """Load the consolidated metrics file.

    Args:
        path: Path to ``outputs/metrics.json``.

    Returns:
        The parsed contents, or an empty dict if the file does not exist yet.
    """
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_metrics(path: Path, section: str, data: Any) -> dict[str, Any]:
    """Merge ``data`` into ``metrics[section]`` and rewrite the metrics file.

    Args:
        path: Path to ``outputs/metrics.json``.
        section: Top-level key to write, e.g. ``"detection"``, ``"speed"``.
        data: New, JSON-serialisable content for that section (a dict or a
            list, depending on the section), replacing any existing value.

    Returns:
        The full, updated metrics dict.
    """
    metrics = load_metrics(path)
    metrics[section] = data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics
