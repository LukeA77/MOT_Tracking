"""Parsing of raw MOT17 sequence metadata (``seqinfo.ini``) and ground truth (``gt.txt``).

MOT17 ships each labelled sequence three times under ``train/`` (suffixes
``-DPM``, ``-FRCNN``, ``-SDP``) with identical images and ground truth,
differing only in bundled public detections we never use. This module
discovers and deduplicates those to one directory per base sequence name,
and parses each sequence's metadata and ground-truth rows.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

from mot_pipeline.utils.paths import base_sequence_name, require_dir

# Preference order when more than one public-detector variant of a sequence
# is present on disk; images/GT are identical across variants so the choice
# only affects nothing observable, but must be deterministic.
_VARIANT_PREFERENCE = ("-FRCNN", "-SDP", "-DPM")


@dataclass(frozen=True)
class SeqInfo:
    """Parsed contents of a MOT17 sequence's ``seqinfo.ini``."""

    name: str
    im_dir: str
    frame_rate: int
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str


@dataclass(frozen=True)
class GtRow:
    """A single row of a MOT17 ``gt.txt`` ground-truth file."""

    frame: int
    track_id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    consider_flag: int
    cls: int
    visibility: float


def read_seqinfo(seqinfo_path: Path) -> SeqInfo:
    """Parse a MOT17 ``seqinfo.ini`` file.

    Args:
        seqinfo_path: Path to the sequence's ``seqinfo.ini``.

    Returns:
        The parsed :class:`SeqInfo`.

    Raises:
        FileNotFoundError: If ``seqinfo_path`` does not exist.
        KeyError: If the ``[Sequence]`` section is missing a required key.
    """
    if not seqinfo_path.is_file():
        raise FileNotFoundError(f"seqinfo.ini not found at {seqinfo_path}")
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    section = parser["Sequence"]
    return SeqInfo(
        name=section["name"],
        im_dir=section["imDir"],
        frame_rate=int(section["frameRate"]),
        seq_length=int(section["seqLength"]),
        im_width=int(section["imWidth"]),
        im_height=int(section["imHeight"]),
        im_ext=section["imExt"],
    )


def read_gt(gt_path: Path) -> list[GtRow]:
    """Parse a MOT17 ``gt/gt.txt`` ground-truth file.

    Columns (1-indexed, no header): frame, id, bb_left, bb_top, bb_width,
    bb_height, consider_flag, class, visibility.

    Args:
        gt_path: Path to the sequence's ``gt.txt``.

    Returns:
        Parsed rows in file order.

    Raises:
        FileNotFoundError: If ``gt_path`` does not exist.
        ValueError: If a line does not have exactly 9 fields.
    """
    if not gt_path.is_file():
        raise FileNotFoundError(f"gt.txt not found at {gt_path}")

    rows: list[GtRow] = []
    for line_no, raw_line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) != 9:
            raise ValueError(f"{gt_path}:{line_no}: expected 9 fields, got {len(fields)}: {line!r}")
        rows.append(
            GtRow(
                frame=int(fields[0]),
                track_id=int(fields[1]),
                bb_left=float(fields[2]),
                bb_top=float(fields[3]),
                bb_width=float(fields[4]),
                bb_height=float(fields[5]),
                consider_flag=int(fields[6]),
                cls=int(fields[7]),
                visibility=float(fields[8]),
            )
        )
    return rows


def discover_train_sequences(raw_dir: Path) -> dict[str, Path]:
    """Discover and deduplicate MOT17 train-split sequence directories.

    Only ``raw_dir/train`` is scanned: the MOT17 test split has no public
    ground truth and must never enter a discovered sequence map.

    Args:
        raw_dir: Root of the extracted MOT17 dataset (containing ``train/``
            and ``test/``).

    Returns:
        A mapping from base sequence name (e.g. ``"MOT17-09"``) to the
        chosen on-disk directory for that sequence.

    Raises:
        FileNotFoundError: If ``raw_dir/train`` does not exist.
    """
    train_dir = require_dir(raw_dir / "train", "MOT17 train directory")

    variants_by_base: dict[str, dict[str, Path]] = {}
    for entry in sorted(train_dir.iterdir()):
        if not entry.is_dir():
            continue
        base = base_sequence_name(entry.name)
        suffix = entry.name[len(base) :]
        variants_by_base.setdefault(base, {})[suffix] = entry

    chosen: dict[str, Path] = {}
    for base, variants in variants_by_base.items():
        for suffix in _VARIANT_PREFERENCE:
            if suffix in variants:
                chosen[base] = variants[suffix]
                break
        else:
            chosen[base] = next(iter(variants.values()))
    return chosen
