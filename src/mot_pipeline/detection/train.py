"""Fine-tune the detector on the converted MOT17 YOLO dataset."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from mot_pipeline.config import Config
from mot_pipeline.detection.evaluate import evaluate_detector
from mot_pipeline.utils.experiment import mlflow_run
from mot_pipeline.utils.logging import get_logger
from mot_pipeline.utils.metrics_io import update_metrics
from mot_pipeline.utils.seed import set_seed


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a detector fine-tuning run."""

    best_weights_path: Path
    map50: float
    map50_95: float


def train_detector(config: Config) -> TrainResult:
    """Fine-tune ``config.detection.model`` on the converted dataset.

    Trains with Ultralytics, copies the resulting best checkpoint into
    ``config.paths.weights_dir``, evaluates it on the validation split, and
    records the run (config, git SHA, dependency versions, metrics) to
    MLflow and to ``outputs/metrics.json``.

    Args:
        config: Validated pipeline configuration.

    Returns:
        The path to the copied best checkpoint and its validation mAP.

    Raises:
        FileNotFoundError: If ``dataset.yaml`` has not been generated yet
            (run ``prepare_dataset`` first).
    """
    from ultralytics import YOLO
    from ultralytics import settings as ultralytics_settings

    # Ultralytics' own MLflow auto-logging starts a second run that corrupts
    # mlflow's active-run tracking out from under mlflow_run() below.
    ultralytics_settings.update({"mlflow": False})

    logger = get_logger(__name__, config.log_level)
    set_seed(config.detection.seed)

    dataset_yaml = config.paths.converted_dir / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"dataset.yaml not found at {dataset_yaml}; run prepare_dataset first."
        )

    model = YOLO(config.detection.model)
    runs_dir = config.paths.outputs_dir / "runs" / "detect"

    with mlflow_run("detection-train", config):
        model.train(
            data=str(dataset_yaml),
            epochs=config.detection.epochs,
            imgsz=config.detection.imgsz,
            batch=config.detection.batch,
            device=config.detection.device,
            fraction=config.detection.fraction,
            seed=config.detection.seed,
            project=str(runs_dir),
            name="train",
            exist_ok=True,
        )

        if model.trainer is None:
            raise RuntimeError("model.train() completed but left no trainer; cannot locate best.pt")
        best_src = Path(model.trainer.best)
        config.paths.weights_dir.mkdir(parents=True, exist_ok=True)
        best_dst = config.paths.weights_dir / "best.pt"
        shutil.copy2(best_src, best_dst)

        detection_metrics = evaluate_detector(config, best_dst)

        import mlflow

        mlflow.log_metrics(
            {"map50": detection_metrics.map50, "map50_95": detection_metrics.map50_95}
        )

    update_metrics(
        config.paths.outputs_dir / "metrics.json",
        "detection",
        {"map50": detection_metrics.map50, "map50_95": detection_metrics.map50_95},
    )
    logger.info(
        "Training complete: mAP@0.5=%.4f mAP@0.5:0.95=%.4f -> %s",
        detection_metrics.map50,
        detection_metrics.map50_95,
        best_dst,
    )
    return TrainResult(
        best_weights_path=best_dst,
        map50=detection_metrics.map50,
        map50_95=detection_metrics.map50_95,
    )
