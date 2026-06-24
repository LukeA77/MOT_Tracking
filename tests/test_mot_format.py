"""Tests for MOTChallenge result file read/write round-tripping."""

from __future__ import annotations

from pathlib import Path

import pytest

from mot_pipeline.utils.mot_format import MotRecord, read_mot_results, write_mot_results


def test_write_then_read_is_lossless(tmp_path: Path) -> None:
    records = [
        MotRecord(
            frame=1,
            track_id=1,
            bb_left=10.5,
            bb_top=20.25,
            bb_width=30.0,
            bb_height=60.0,
            conf=0.913,
        ),
        MotRecord(
            frame=1, track_id=2, bb_left=100.0, bb_top=50.0, bb_width=25.5, bb_height=80.0, conf=0.5
        ),
        MotRecord(
            frame=2, track_id=1, bb_left=11.0, bb_top=21.0, bb_width=30.0, bb_height=60.0, conf=0.9
        ),
    ]
    out_path = tmp_path / "results.txt"

    write_mot_results(out_path, records)
    round_tripped = read_mot_results(out_path)

    assert round_tripped == records


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    out_path = tmp_path / "nested" / "dir" / "results.txt"
    write_mot_results(
        out_path, [MotRecord(frame=1, track_id=1, bb_left=0, bb_top=0, bb_width=1, bb_height=1)]
    )
    assert out_path.exists()


def test_write_empty_records_produces_empty_file(tmp_path: Path) -> None:
    out_path = tmp_path / "empty.txt"
    write_mot_results(out_path, [])
    assert read_mot_results(out_path) == []


def test_write_rejects_non_positive_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame/track ids"):
        write_mot_results(
            tmp_path / "bad.txt",
            [MotRecord(frame=0, track_id=1, bb_left=0, bb_top=0, bb_width=1, bb_height=1)],
        )


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_mot_results(tmp_path / "missing.txt")


def test_read_rejects_malformed_line(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.txt"
    bad_path.write_text("1,2,3,4,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 10"):
        read_mot_results(bad_path)


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "with_blanks.txt"
    path.write_text("1,1,0.0,0.0,1.0,1.0,1.0,-1,-1,-1\n\n\n", encoding="utf-8")
    assert len(read_mot_results(path)) == 1
