"""Export the fine-tuned detector to ONNX and verify PyTorch/ONNX parity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mot_pipeline.config import Config


@dataclass(frozen=True)
class ParityResult:
    """Outcome of comparing PyTorch and ONNX detections on a sample frame."""

    onnx_path: Path
    parity_ok: bool
    max_abs_diff_px: float


def export_to_onnx(config: Config, weights_path: Path) -> Path:
    """Export a detector checkpoint to ONNX.

    YOLO26's NMS-free, DFL-free head makes the exported graph self-contained
    (no external NMS post-processing step is required), which is why the
    export is clean and the resulting ONNX model benchmarks well.

    Args:
        config: Validated pipeline configuration.
        weights_path: Path to the ``.pt`` checkpoint to export.

    Returns:
        Path to the exported ``.onnx`` file, moved into ``config.paths.weights_dir``.

    Raises:
        FileNotFoundError: If ``weights_path`` does not exist.
    """
    from ultralytics import YOLO

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = YOLO(str(weights_path))
    exported_path = Path(
        model.export(
            format=config.export.format,
            opset=config.export.opset,
            dynamic=config.export.dynamic,
            imgsz=config.detection.imgsz,
        )
    )

    config.paths.weights_dir.mkdir(parents=True, exist_ok=True)
    final_path = config.paths.weights_dir / "best.onnx"
    if exported_path.resolve() != final_path.resolve():
        exported_path.replace(final_path)
    return final_path


def check_parity(
    pt_weights_path: Path,
    onnx_weights_path: Path,
    sample_image_path: Path,
    imgsz: int,
    conf: float = 0.25,
    pixel_tolerance: float = 5.0,
) -> ParityResult:
    """Confirm PyTorch and ONNX produce equivalent detections on a sample frame.

    Boxes from each model are sorted by their left edge before comparison
    so detections line up even if the two runtimes return them in a
    different order; this is a fast sanity check, not a full matching
    algorithm, which is sufficient to catch export issues such as a bad
    opset or a transposed output.

    Args:
        pt_weights_path: Path to the original ``.pt`` checkpoint.
        onnx_weights_path: Path to the exported ``.onnx`` model.
        sample_image_path: A representative frame to run both models on.
        imgsz: Inference image size (should match the export ``imgsz``).
        conf: Confidence threshold used for both models.
        pixel_tolerance: Maximum allowed per-coordinate pixel difference for
            parity to be considered passing.

    Returns:
        Whether detection counts match and matched boxes agree within
        ``pixel_tolerance``, plus the maximum observed absolute difference.

    Raises:
        FileNotFoundError: If any input path does not exist.
    """
    from ultralytics import YOLO

    for path in (pt_weights_path, onnx_weights_path, sample_image_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    pt_boxes = _sorted_predicted_boxes(YOLO(str(pt_weights_path)), sample_image_path, imgsz, conf)
    onnx_boxes = _sorted_predicted_boxes(
        YOLO(str(onnx_weights_path)), sample_image_path, imgsz, conf
    )

    if len(pt_boxes) != len(onnx_boxes):
        return ParityResult(onnx_weights_path, parity_ok=False, max_abs_diff_px=float("inf"))
    if len(pt_boxes) == 0:
        return ParityResult(onnx_weights_path, parity_ok=True, max_abs_diff_px=0.0)

    max_abs_diff = float(np.max(np.abs(pt_boxes - onnx_boxes)))
    return ParityResult(
        onnx_weights_path, parity_ok=max_abs_diff <= pixel_tolerance, max_abs_diff_px=max_abs_diff
    )


def _sorted_predicted_boxes(model: object, image_path: Path, imgsz: int, conf: float) -> np.ndarray:
    result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]  # type: ignore[attr-defined]
    boxes = result.boxes.xyxy.cpu().numpy()
    return boxes[np.argsort(boxes[:, 0])]
