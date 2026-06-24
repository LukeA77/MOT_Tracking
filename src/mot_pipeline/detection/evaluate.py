"""Standalone detection mAP evaluation on the validation split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mot_pipeline.config import Config


@dataclass(frozen=True)
class DetectionMetrics:
    """Detection accuracy on the configured validation split."""

    map50: float
    map50_95: float


def evaluate_detector(config: Config, weights_path: Path) -> DetectionMetrics:
    """Evaluate a trained detector checkpoint on the validation split.

    Args:
        config: Validated pipeline configuration.
        weights_path: Path to a ``.pt`` (or exported) weights file to evaluate.

    Returns:
        mAP@0.5 and mAP@0.5:0.95 on the configured validation split.

    Raises:
        FileNotFoundError: If ``weights_path`` or ``dataset.yaml`` does not exist.
    """
    from ultralytics import YOLO

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    dataset_yaml = config.paths.converted_dir / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"dataset.yaml not found at {dataset_yaml}; run prepare_dataset first."
        )

    model = YOLO(str(weights_path))
    results = model.val(
        data=str(dataset_yaml),
        imgsz=config.detection.imgsz,
        device=config.detection.device,
        split="val",
    )
    return DetectionMetrics(map50=float(results.box.map50), map50_95=float(results.box.map))
