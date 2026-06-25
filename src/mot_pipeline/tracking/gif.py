"""Convert a short clip of a tracked video into a GIF for the README.

Uses OpenCV to decode frames and ``imageio`` to encode the GIF, avoiding a
hard dependency on an ``ffmpeg`` binary being available on PATH.
"""

from __future__ import annotations

from pathlib import Path


def video_to_gif(video_path: Path, out_path: Path, seconds: float = 5.0, fps: int = 10) -> Path:
    """Extract the first ``seconds`` of ``video_path`` and save as a looping GIF.

    Args:
        video_path: Source video, e.g. ``outputs/tracked_output_botsort.mp4``.
        out_path: Destination ``.gif`` path.
        seconds: Clip length to extract, measured from the start of the video.
        fps: Output GIF frame rate (downsampled from the source video for a
            reasonable file size).

    Returns:
        The path to the written GIF.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist or cannot be opened.
    """
    import cv2
    import imageio.v2 as imageio

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_stride = max(1, round(source_fps / fps))
    max_frames = int(seconds * source_fps)

    frames = []
    frame_idx = 0
    try:
        while frame_idx < max_frames:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            if frame_idx % frame_stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
    finally:
        capture.release()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, loop=0)  # type: ignore[arg-type]
    return out_path
