"""Detect + track over the eval sequence with each configured tracker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mot_pipeline.config import Config
from mot_pipeline.data.mot17 import discover_train_sequences, read_seqinfo
from mot_pipeline.tracking.visualize import render_tracking_video
from mot_pipeline.utils.logging import get_logger
from mot_pipeline.utils.mot_format import MotRecord, write_mot_results
from mot_pipeline.utils.paths import sequence_image_dir


@dataclass(frozen=True)
class TrackerRunResult:
    """Outcome of running one tracker over the eval sequence."""

    tracker: str
    results_path: Path
    video_path: Path
    n_tracks: int
    n_detections: int


def track_eval_sequence(config: Config, weights_path: Path) -> list[TrackerRunResult]:
    """Run detect+track over ``config.split.eval_sequence`` with every configured tracker.

    For each tracker, writes a MOTChallenge results file
    (``outputs/<seq>_<tracker>.txt``) and an annotated video
    (``outputs/tracked_output_<tracker>.mp4``) with persistent per-ID colours.

    Args:
        config: Validated pipeline configuration.
        weights_path: Path to the detector checkpoint to track with.

    Returns:
        One result per configured tracker, in config order.

    Raises:
        FileNotFoundError: If ``weights_path`` or the eval sequence is missing.
    """
    from ultralytics import YOLO

    logger = get_logger(__name__, config.log_level)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    available = discover_train_sequences(config.paths.raw_dir)
    eval_sequence = config.split.eval_sequence
    if eval_sequence not in available:
        raise FileNotFoundError(
            f"eval_sequence {eval_sequence!r} not found under {config.paths.raw_dir}. "
            f"Available sequences: {sorted(available)}"
        )
    seq_dir = available[eval_sequence]
    seqinfo = read_seqinfo(seq_dir / "seqinfo.ini")
    img_dir = sequence_image_dir(seq_dir)

    model = YOLO(str(weights_path))

    results: list[TrackerRunResult] = []
    for tracker_filename in config.tracking.trackers:
        tracker_name = Path(tracker_filename).stem
        tracker_cfg_path = config.paths.configs_dir / tracker_filename

        records = _track_sequence(model, img_dir, tracker_cfg_path, config)

        results_path = config.paths.outputs_dir / f"{eval_sequence}_{tracker_name}.txt"
        write_mot_results(results_path, records)

        video_path = config.paths.outputs_dir / f"tracked_output_{tracker_name}.mp4"
        render_tracking_video(
            img_dir, records, seqinfo.seq_length, seqinfo.frame_rate, video_path, seqinfo.im_ext
        )

        n_tracks = len({record.track_id for record in records})
        results.append(
            TrackerRunResult(tracker_name, results_path, video_path, n_tracks, len(records))
        )
        logger.info(
            "%s: %d detections across %d tracks -> %s",
            tracker_name,
            len(records),
            n_tracks,
            results_path,
        )

    return results


def _track_sequence(
    model: object, img_dir: Path, tracker_cfg_path: Path, config: Config
) -> list[MotRecord]:
    """Run one Ultralytics tracker over a sequence's frames and collect MOT records."""
    records: list[MotRecord] = []
    results_stream = model.track(  # type: ignore[attr-defined]
        source=str(img_dir),
        tracker=str(tracker_cfg_path),
        conf=config.tracking.conf,
        iou=config.tracking.iou,
        device=config.detection.device,
        stream=True,
        persist=True,
        verbose=False,
    )
    for frame_idx, result in enumerate(results_stream, start=1):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), track_id, conf in zip(xyxy, track_ids, confidences, strict=True):
            records.append(
                MotRecord(
                    frame=frame_idx,
                    track_id=int(track_id),
                    bb_left=float(x1),
                    bb_top=float(y1),
                    bb_width=float(x2 - x1),
                    bb_height=float(y2 - y1),
                    conf=float(conf),
                )
            )
    return records
