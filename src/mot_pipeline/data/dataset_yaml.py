"""Generate the Ultralytics ``dataset.yaml`` for the converted YOLO dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def build_dataset_yaml_content(converted_dir: Path, class_names: list[str]) -> dict[str, Any]:
    """Build the dataset.yaml content for an Ultralytics detection dataset.

    Args:
        converted_dir: Root of the converted dataset, containing
            ``images/train``, ``images/val``, ``labels/train``, ``labels/val``.
        class_names: Ordered class names; index in this list is the class id.

    Returns:
        A dict ready to be YAML-serialised, matching Ultralytics' expected schema.

    Raises:
        ValueError: If ``class_names`` is empty.
    """
    if not class_names:
        raise ValueError("class_names must not be empty")
    return {
        "path": str(converted_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": list(class_names),
    }


def write_dataset_yaml(converted_dir: Path, class_names: list[str]) -> Path:
    """Write ``dataset.yaml`` into the converted dataset directory.

    Args:
        converted_dir: Root of the converted dataset.
        class_names: Ordered class names.

    Returns:
        Path to the written ``dataset.yaml``.
    """
    content = build_dataset_yaml_content(converted_dir, class_names)
    out_path = converted_dir / "dataset.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return out_path
