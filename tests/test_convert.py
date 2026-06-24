"""Tests for pure MOT17 -> YOLO conversion (the highest-value tests in the suite)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mot_pipeline.config import ConvertConfig
from mot_pipeline.data.convert import (
    YoloBox,
    build_yolo_labels_by_frame,
    filter_gt_rows,
    format_yolo_line,
    gt_row_to_yolo_box,
)
from mot_pipeline.data.mot17 import GtRow, read_gt, read_seqinfo


def _row(**overrides: float | int) -> GtRow:
    defaults: dict[str, float | int] = dict(
        frame=1,
        track_id=1,
        bb_left=100,
        bb_top=50,
        bb_width=40,
        bb_height=80,
        consider_flag=1,
        cls=1,
        visibility=1.0,
    )
    defaults.update(overrides)
    return GtRow(**defaults)  # type: ignore[arg-type]


def test_gt_row_to_yolo_box_computes_normalised_center_and_size() -> None:
    box = gt_row_to_yolo_box(_row(), im_width=640, im_height=480)
    assert box.cls == 0
    assert box.x_center == pytest.approx(0.1875)
    assert box.y_center == pytest.approx(0.1875)
    assert box.width == pytest.approx(0.0625)
    assert box.height == pytest.approx(80 / 480)


def test_gt_row_to_yolo_box_clips_positive_overflow_to_one() -> None:
    box = gt_row_to_yolo_box(
        _row(bb_left=620, bb_top=460, bb_width=80, bb_height=80), im_width=640, im_height=480
    )
    assert box.x_center == pytest.approx(1.0)
    assert box.y_center == pytest.approx(1.0)
    assert box.width == pytest.approx(0.125)
    assert box.height == pytest.approx(80 / 480)


def test_gt_row_to_yolo_box_clips_negative_overflow_to_zero() -> None:
    box = gt_row_to_yolo_box(_row(bb_left=-30, bb_width=20), im_width=640, im_height=480)
    assert box.x_center == 0.0


def test_gt_row_to_yolo_box_rejects_non_positive_image_dimensions() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        gt_row_to_yolo_box(_row(), im_width=0, im_height=480)


def test_format_yolo_line_uses_six_decimal_places() -> None:
    box = YoloBox(cls=0, x_center=0.1875, y_center=0.1875, width=0.0625, height=1 / 6)
    assert format_yolo_line(box) == "0 0.187500 0.187500 0.062500 0.166667"


def test_filter_gt_rows_applies_consider_flag_class_and_visibility() -> None:
    rows = [
        _row(track_id=1),  # kept
        _row(track_id=2, cls=2),  # wrong class -> filtered
        _row(track_id=3, consider_flag=0),  # consider flag -> filtered
        _row(track_id=4, visibility=0.05),  # below visibility threshold -> filtered
    ]
    config = ConvertConfig(keep_classes=[1], use_consider_flag=True, min_visibility=0.1)

    kept = filter_gt_rows(rows, config)

    assert [row.track_id for row in kept] == [1]


def test_filter_gt_rows_can_disable_consider_flag_filtering() -> None:
    rows = [_row(track_id=1, consider_flag=0)]
    config = ConvertConfig(use_consider_flag=False)

    kept = filter_gt_rows(rows, config)

    assert [row.track_id for row in kept] == [1]


def test_build_yolo_labels_by_frame_covers_every_frame_including_empty_ones() -> None:
    kept_rows = [_row(frame=1, track_id=1), _row(frame=3, track_id=2)]

    labels = build_yolo_labels_by_frame(kept_rows, im_width=640, im_height=480, seq_length=3)

    assert set(labels.keys()) == {1, 2, 3}
    assert len(labels[1]) == 1
    assert labels[2] == []
    assert len(labels[3]) == 1


def test_build_yolo_labels_by_frame_with_no_detections_at_all() -> None:
    labels = build_yolo_labels_by_frame([], im_width=640, im_height=480, seq_length=5)
    assert labels == {1: [], 2: [], 3: [], 4: [], 5: []}


def test_end_to_end_conversion_against_synthetic_sequence_fixture(mot17_sequence_dir: Path) -> None:
    seqinfo = read_seqinfo(mot17_sequence_dir / "seqinfo.ini")
    gt_rows = read_gt(mot17_sequence_dir / "gt" / "gt.txt")
    config = ConvertConfig(keep_classes=[1], use_consider_flag=True, min_visibility=0.1)

    kept = filter_gt_rows(gt_rows, config)
    labels = build_yolo_labels_by_frame(
        kept, im_width=seqinfo.im_width, im_height=seqinfo.im_height, seq_length=seqinfo.seq_length
    )

    # Frame 1: rows for ids 1 and 2 survive (id 3 wrong class, id 4 consider_flag=0,
    # id 5 below visibility). Frame 2: no ground truth at all. Frame 3: one row (id 6).
    assert len(labels[1]) == 2
    assert labels[2] == []
    assert len(labels[3]) == 1

    cls, x_center, y_center, width, height = labels[1][0].split()
    assert cls == "0"
    assert float(x_center) == pytest.approx(0.1875)
    assert float(y_center) == pytest.approx(0.1875)
    assert float(width) == pytest.approx(0.0625)
    assert float(height) == pytest.approx(80 / 480, abs=1e-6)
