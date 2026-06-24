"""Pure MOT17 ground-truth -> YOLO label conversion.

Every function here is a pure transform over in-memory data (no file I/O),
so conversion correctness can be unit-tested with synthetic inputs without
touching disk. Orchestration (reading raw files, writing converted output)
lives in :mod:`mot_pipeline.data.prepare`.
"""

from __future__ import annotations

from dataclasses import dataclass

from mot_pipeline.config import ConvertConfig
from mot_pipeline.data.mot17 import GtRow

_YOLO_PERSON_CLASS = 0


@dataclass(frozen=True)
class YoloBox:
    """A single normalised YOLO-format detection box."""

    cls: int
    x_center: float
    y_center: float
    width: float
    height: float


def filter_gt_rows(rows: list[GtRow], convert_config: ConvertConfig) -> list[GtRow]:
    """Keep only ground-truth rows that should become training labels.

    A row is kept when (if enabled) its consider flag is 1, its class is in
    ``convert_config.keep_classes``, and its visibility is at least
    ``convert_config.min_visibility``.

    Args:
        rows: Raw ground-truth rows for one sequence.
        convert_config: Filtering thresholds from the validated config.

    Returns:
        The subset of ``rows`` passing all filters, in the original order.
    """
    keep_classes = set(convert_config.keep_classes)
    kept = []
    for row in rows:
        if convert_config.use_consider_flag and row.consider_flag != 1:
            continue
        if row.cls not in keep_classes:
            continue
        if row.visibility < convert_config.min_visibility:
            continue
        kept.append(row)
    return kept


def gt_row_to_yolo_box(row: GtRow, im_width: int, im_height: int) -> YoloBox:
    """Convert one MOT17 ground-truth row to a normalised YOLO box.

    The single output class is always ``0`` ("person"); MOT17 pedestrian
    class ids are not preserved since this dataset trains a single-class
    detector. Coordinates are clipped to ``[0, 1]`` to guard against boxes
    that extend past the image border in the source annotations.

    Args:
        row: A ground-truth row already passed through :func:`filter_gt_rows`.
        im_width: Image width in pixels, from ``seqinfo.ini``.
        im_height: Image height in pixels, from ``seqinfo.ini``.

    Returns:
        The equivalent normalised :class:`YoloBox`.

    Raises:
        ValueError: If ``im_width`` or ``im_height`` is not positive.
    """
    if im_width <= 0 or im_height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {im_width}x{im_height}")

    x_center = (row.bb_left + row.bb_width / 2) / im_width
    y_center = (row.bb_top + row.bb_height / 2) / im_height
    width = row.bb_width / im_width
    height = row.bb_height / im_height

    return YoloBox(
        cls=_YOLO_PERSON_CLASS,
        x_center=_clip01(x_center),
        y_center=_clip01(y_center),
        width=_clip01(width),
        height=_clip01(height),
    )


def format_yolo_line(box: YoloBox) -> str:
    """Format a :class:`YoloBox` as one YOLO label-file line.

    Args:
        box: The box to format.

    Returns:
        A line of the form ``"cls x_center y_center width height"``,
        space-separated, with 6 decimal places.
    """
    return f"{box.cls} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"


def group_rows_by_frame(rows: list[GtRow]) -> dict[int, list[GtRow]]:
    """Group ground-truth rows by frame number.

    Args:
        rows: Ground-truth rows for one sequence, any order.

    Returns:
        A mapping from frame number to the rows belonging to that frame.
    """
    grouped: dict[int, list[GtRow]] = {}
    for row in rows:
        grouped.setdefault(row.frame, []).append(row)
    return grouped


def build_yolo_labels_by_frame(
    kept_rows: list[GtRow],
    im_width: int,
    im_height: int,
    seq_length: int,
) -> dict[int, list[str]]:
    """Build YOLO label lines for every frame of a sequence.

    Frames with no kept detections map to an empty list, so callers write an
    empty (valid) label file for them rather than omitting the frame.

    Args:
        kept_rows: Ground-truth rows already filtered by :func:`filter_gt_rows`.
        im_width: Image width in pixels.
        im_height: Image height in pixels.
        seq_length: Total number of frames in the sequence (from ``seqinfo.ini``).

    Returns:
        A mapping from frame number (``1..seq_length``) to its YOLO label lines.
    """
    grouped = group_rows_by_frame(kept_rows)
    return {
        frame: [format_yolo_line(gt_row_to_yolo_box(row, im_width, im_height)) for row in rows]
        for frame, rows in ((f, grouped.get(f, [])) for f in range(1, seq_length + 1))
    }


def _clip01(value: float) -> float:
    """Clip a normalised coordinate to the closed interval ``[0, 1]``."""
    return max(0.0, min(1.0, value))
