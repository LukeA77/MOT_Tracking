"""Render annotated tracking videos: boxes with a stable colour per track id."""

from __future__ import annotations

import colorsys
from pathlib import Path

from mot_pipeline.utils.mot_format import MotRecord

# Golden-ratio hue spacing gives visually distinct, deterministic colours
# without needing to track which hues have already been assigned.
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def color_for_track_id(track_id: int) -> tuple[int, int, int]:
    """Deterministically map a track id to a stable BGR colour.

    Args:
        track_id: MOTChallenge track id (>= 1).

    Returns:
        A ``(B, G, R)`` colour tuple, each channel in ``[0, 255]``, suitable
        for OpenCV drawing functions.
    """
    hue = (track_id * _GOLDEN_RATIO_CONJUGATE) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(blue * 255), int(green * 255), int(red * 255))


def render_tracking_video(
    image_dir: Path,
    records: list[MotRecord],
    seq_length: int,
    frame_rate: int,
    out_path: Path,
    im_ext: str = ".jpg",
) -> Path:
    """Draw boxes + persistent track IDs over each frame and write an annotated video.

    Args:
        image_dir: Directory of source frame images (a sequence's ``img1``).
        records: Tracking results to overlay, any order.
        seq_length: Number of frames in the sequence.
        frame_rate: Output video frame rate, from the sequence's ``seqinfo.ini``.
        out_path: Destination ``.mp4`` path.
        im_ext: Source image file extension.

    Returns:
        The path to the written video.

    Raises:
        FileNotFoundError: If the first frame image cannot be read (used to
            size the video writer).
    """
    import cv2

    by_frame: dict[int, list[MotRecord]] = {}
    for record in records:
        by_frame.setdefault(record.frame, []).append(record)

    first_frame_path = image_dir / f"{1:06d}{im_ext}"
    first_frame = cv2.imread(str(first_frame_path))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read first frame at {first_frame_path}")
    height, width = first_frame.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height)
    )
    try:
        for frame in range(1, seq_length + 1):
            image = cv2.imread(str(image_dir / f"{frame:06d}{im_ext}"))
            if image is None:
                continue
            for record in by_frame.get(frame, []):
                color = color_for_track_id(record.track_id)
                x1, y1 = int(record.bb_left), int(record.bb_top)
                x2 = int(record.bb_left + record.bb_width)
                y2 = int(record.bb_top + record.bb_height)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    f"ID {record.track_id}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            writer.write(image)
    finally:
        writer.release()
    return out_path
