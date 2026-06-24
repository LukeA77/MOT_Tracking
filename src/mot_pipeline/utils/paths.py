"""Path resolution helpers shared by the data, tracking, and export stages."""

from __future__ import annotations

from pathlib import Path

_TRACKING_SUFFIXES = ("-DPM", "-FRCNN", "-SDP")


def base_sequence_name(sequence_dir_name: str) -> str:
    """Strip the MOT17 public-detector suffix from a sequence directory name.

    MOT17 ships each sequence three times (``-DPM``, ``-FRCNN``, ``-SDP``)
    with identical images/ground-truth, differing only in the bundled public
    detections we never use. This collapses all three to one base name.

    Args:
        sequence_dir_name: Raw directory name, e.g. ``"MOT17-09-FRCNN"``.

    Returns:
        The base sequence name, e.g. ``"MOT17-09"``.
    """
    for suffix in _TRACKING_SUFFIXES:
        if sequence_dir_name.endswith(suffix):
            return sequence_dir_name[: -len(suffix)]
    return sequence_dir_name


def require_dir(path: Path, description: str) -> Path:
    """Validate that a directory exists, raising a clear error if not.

    Args:
        path: Directory path to check.
        description: Human-readable description used in the error message,
            e.g. ``"MOT17 raw data directory"``.

    Returns:
        The same ``path``, unchanged, for chaining.

    Raises:
        FileNotFoundError: If ``path`` does not exist or is not a directory.
    """
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found at {path}")
    return path


def sequence_image_dir(sequence_dir: Path) -> Path:
    """Return the ``img1`` subdirectory of a raw MOT17 sequence directory.

    Args:
        sequence_dir: Path to a sequence directory, e.g.
            ``data/MOT17/train/MOT17-09-FRCNN``.

    Returns:
        Path to that sequence's image frames directory.
    """
    return sequence_dir / "img1"
