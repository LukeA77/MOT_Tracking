"""Orchestrate raw MOT17 -> converted YOLO dataset conversion.

This is the I/O-and-side-effects edge of the data stage: it reads raw
sequence directories, calls the pure transforms in :mod:`mot_pipeline.data.convert`,
copies frame images, writes label files, and emits ``dataset.yaml``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from mot_pipeline.config import Config
from mot_pipeline.data.convert import build_yolo_labels_by_frame, filter_gt_rows
from mot_pipeline.data.dataset_yaml import write_dataset_yaml
from mot_pipeline.data.mot17 import SeqInfo, discover_train_sequences, read_gt, read_seqinfo
from mot_pipeline.data.split import resolve_split
from mot_pipeline.utils.logging import get_logger

_FRAME_FILENAME_DIGITS = 6


@dataclass(frozen=True)
class SequenceSummary:
    """Per-sequence outcome of dataset conversion."""

    sequence: str
    subset: str
    n_images: int
    n_boxes: int


@dataclass
class PrepareSummary:
    """Aggregate outcome of :func:`prepare_dataset`."""

    sequences: list[SequenceSummary] = field(default_factory=list)
    dataset_yaml_path: Path | None = None

    @property
    def total_images(self) -> int:
        """Total number of frame images converted across all sequences."""
        return sum(s.n_images for s in self.sequences)

    @property
    def total_boxes(self) -> int:
        """Total number of YOLO boxes written across all sequences."""
        return sum(s.n_boxes for s in self.sequences)


def prepare_dataset(config: Config) -> PrepareSummary:
    """Convert raw MOT17 into a YOLO-format dataset under ``config.paths.converted_dir``.

    Args:
        config: Validated pipeline configuration.

    Returns:
        Summary of images/boxes written per sequence, plus the dataset.yaml path.

    Raises:
        FileNotFoundError: If a configured split sequence is missing on disk,
            or an expected frame image is missing.
    """
    logger = get_logger(__name__, config.log_level)
    available = discover_train_sequences(config.paths.raw_dir)
    resolved = resolve_split(config.split, available)

    summary = PrepareSummary()
    for subset, sequences in (("train", resolved.train), ("val", resolved.val)):
        for seq_name, seq_dir in sequences.items():
            n_images, n_boxes = _convert_sequence(seq_name, seq_dir, subset, config)
            summary.sequences.append(SequenceSummary(seq_name, subset, n_images, n_boxes))
            logger.info(
                "Converted %s (%s): %d images, %d boxes", seq_name, subset, n_images, n_boxes
            )

    summary.dataset_yaml_path = write_dataset_yaml(
        config.paths.converted_dir, class_names=["person"]
    )
    logger.info("Wrote dataset.yaml to %s", summary.dataset_yaml_path)
    return summary


def _convert_sequence(seq_name: str, seq_dir: Path, subset: str, config: Config) -> tuple[int, int]:
    seqinfo = read_seqinfo(seq_dir / "seqinfo.ini")
    gt_rows = read_gt(seq_dir / "gt" / "gt.txt")
    kept_rows = filter_gt_rows(gt_rows, config.convert)
    labels_by_frame = build_yolo_labels_by_frame(
        kept_rows, seqinfo.im_width, seqinfo.im_height, seqinfo.seq_length
    )

    images_out = config.paths.converted_dir / "images" / subset
    labels_out = config.paths.converted_dir / "labels" / subset
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_boxes = 0
    for frame, lines in labels_by_frame.items():
        stem = f"{seq_name}_{frame:0{_FRAME_FILENAME_DIGITS}d}"
        src_image = _frame_image_path(seq_dir, seqinfo, frame)
        if not src_image.is_file():
            raise FileNotFoundError(f"Expected frame image not found: {src_image}")
        shutil.copy2(src_image, images_out / f"{stem}{seqinfo.im_ext}")
        (labels_out / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        n_boxes += len(lines)

    return len(labels_by_frame), n_boxes


def _frame_image_path(seq_dir: Path, seqinfo: SeqInfo, frame: int) -> Path:
    filename = f"{frame:0{_FRAME_FILENAME_DIGITS}d}{seqinfo.im_ext}"
    return seq_dir / seqinfo.im_dir / filename
