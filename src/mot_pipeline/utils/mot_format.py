"""Read/write MOTChallenge-format tracking result files.

Each line of a MOTChallenge results file is:
    frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_NUM_FIELDS = 10
_UNUSED = "-1,-1,-1"


@dataclass(frozen=True)
class MotRecord:
    """A single MOTChallenge result row: one tracked box in one frame."""

    frame: int
    track_id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    conf: float = 1.0


def write_mot_results(path: Path, records: list[MotRecord]) -> None:
    """Write tracking results to a MOTChallenge-format ``.txt`` file.

    Args:
        path: Destination file path. Parent directories are created if needed.
        records: Tracked boxes, written in the given order (conventionally
            sorted by frame then track id).

    Raises:
        ValueError: If any record has a non-positive frame or track id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        if record.frame < 1 or record.track_id < 1:
            raise ValueError(
                f"MOTChallenge frame/track ids must be >= 1, got frame={record.frame}, "
                f"track_id={record.track_id}"
            )
        lines.append(
            f"{record.frame},{record.track_id},"
            f"{record.bb_left:.3f},{record.bb_top:.3f},"
            f"{record.bb_width:.3f},{record.bb_height:.3f},"
            f"{record.conf:.6f},{_UNUSED}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_mot_results(path: Path) -> list[MotRecord]:
    """Read a MOTChallenge-format ``.txt`` file into :class:`MotRecord` rows.

    Args:
        path: Path to a MOTChallenge results or ground-truth file.

    Returns:
        Parsed records in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line does not have the expected number of fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"MOTChallenge result file not found: {path}")

    records: list[MotRecord] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) != _NUM_FIELDS:
            raise ValueError(
                f"{path}:{line_no}: expected {_NUM_FIELDS} comma-separated fields, "
                f"got {len(fields)}: {line!r}"
            )
        records.append(
            MotRecord(
                frame=int(fields[0]),
                track_id=int(fields[1]),
                bb_left=float(fields[2]),
                bb_top=float(fields[3]),
                bb_width=float(fields[4]),
                bb_height=float(fields[5]),
                conf=float(fields[6]),
            )
        )
    return records
