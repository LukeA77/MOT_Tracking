"""Tracking metrics (HOTA / MOTA / IDF1) for each tracker against ground truth.

MOTA and IDF1 are computed directly from our own MOTChallenge-format files
via ``py-motmetrics``, which needs no special on-disk layout. HOTA is
computed via the official ``trackeval`` toolkit if it is installed;
trackeval requires a specific folder layout, so :func:`_compute_hota` writes
a small adapter layout under a temp directory. Per the spec, trackeval is
treated as best-effort: if it is missing or its evaluation fails for any
reason, HOTA is reported as unavailable while MOTA/IDF1 are still computed.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from mot_pipeline.config import Config
from mot_pipeline.data.convert import filter_gt_rows
from mot_pipeline.data.mot17 import GtRow, discover_train_sequences, read_gt, read_seqinfo
from mot_pipeline.utils.logging import get_logger
from mot_pipeline.utils.mot_format import MotRecord, read_mot_results

_T = TypeVar("_T")


@dataclass(frozen=True)
class TrackingMetrics:
    """HOTA / MOTA / IDF1 for one tracker against ground truth on the eval sequence."""

    tracker: str
    mota: float
    idf1: float
    num_switches: int
    hota: float | None


def compute_tracking_metrics(config: Config, tracker: str, results_path: Path) -> TrackingMetrics:
    """Compute MOTA/IDF1 (always) and HOTA (best-effort) for one tracker.

    Args:
        config: Validated pipeline configuration.
        tracker: Tracker name, used for the returned record and logging.
        results_path: Path to the tracker's MOTChallenge results file.

    Returns:
        The computed metrics, with ``hota=None`` if trackeval is unavailable
        or its evaluation could not complete.

    Raises:
        FileNotFoundError: If the eval sequence's ground truth or
            ``results_path`` is missing.
    """
    logger = get_logger(__name__, config.log_level)

    available = discover_train_sequences(config.paths.raw_dir)
    eval_sequence = config.split.eval_sequence
    if eval_sequence not in available:
        raise FileNotFoundError(
            f"eval_sequence {eval_sequence!r} not found under {config.paths.raw_dir}"
        )
    seq_dir = available[eval_sequence]
    seqinfo = read_seqinfo(seq_dir / "seqinfo.ini")
    gt_rows = filter_gt_rows(read_gt(seq_dir / "gt" / "gt.txt"), config.convert)
    pred_records = read_mot_results(results_path)

    mota, idf1, num_switches = _compute_clear_and_identity(
        gt_rows, pred_records, seqinfo.seq_length
    )

    hota: float | None = None
    try:
        hota = _compute_hota(eval_sequence, seq_dir, results_path)
    except Exception as exc:  # noqa: BLE001 - trackeval is optional/best-effort, see module docstring
        logger.warning("HOTA unavailable for %s (%s); reporting MOTA/IDF1 only.", tracker, exc)

    return TrackingMetrics(
        tracker=tracker, mota=mota, idf1=idf1, num_switches=num_switches, hota=hota
    )


def _compute_clear_and_identity(
    gt_rows: list[GtRow], pred_records: list[MotRecord], seq_length: int
) -> tuple[float, float, int]:
    import motmetrics as mm

    accumulator = mm.MOTAccumulator(auto_id=False)
    gt_by_frame = _group_by_frame(gt_rows, key=lambda row: row.frame)
    pred_by_frame = _group_by_frame(pred_records, key=lambda record: record.frame)

    for frame in range(1, seq_length + 1):
        gt_frame = gt_by_frame.get(frame, [])
        pred_frame = pred_by_frame.get(frame, [])
        gt_boxes = [[row.bb_left, row.bb_top, row.bb_width, row.bb_height] for row in gt_frame]
        pred_boxes = [
            [record.bb_left, record.bb_top, record.bb_width, record.bb_height]
            for record in pred_frame
        ]
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        accumulator.update(
            [row.track_id for row in gt_frame],
            [record.track_id for record in pred_frame],
            distances,
            frameid=frame,
        )

    metrics_host = mm.metrics.create()
    summary = metrics_host.compute(
        accumulator, metrics=["mota", "idf1", "num_switches"], name="tracker"
    )
    return (
        float(summary["mota"].iloc[0]),
        float(summary["idf1"].iloc[0]),
        int(summary["num_switches"].iloc[0]),
    )


def _group_by_frame(rows: Iterable[_T], key: Callable[[_T], int]) -> dict[int, list[_T]]:
    grouped: dict[int, list[_T]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    return grouped


def _compute_hota(seq_name: str, seq_dir: Path, results_path: Path) -> float:
    """Best-effort HOTA computation via the official trackeval toolkit.

    Builds the GT/tracker/seqmap folder layout trackeval expects under a
    temporary directory, runs its evaluator for a single sequence/tracker,
    and returns the mean HOTA score. Any failure propagates to the caller,
    which treats HOTA as optional.
    """
    import trackeval

    with tempfile.TemporaryDirectory(prefix="trackeval_") as tmp:
        tmp_path = Path(tmp)
        tracker_name = "tracker"

        seq_gt_dir = tmp_path / "gt" / seq_name / "gt"
        seq_gt_dir.mkdir(parents=True)
        shutil.copy2(seq_dir / "gt" / "gt.txt", seq_gt_dir / "gt.txt")
        shutil.copy2(seq_dir / "seqinfo.ini", tmp_path / "gt" / seq_name / "seqinfo.ini")

        seqmap_path = tmp_path / "seqmap.txt"
        seqmap_path.write_text(f"name\n{seq_name}\n", encoding="utf-8")

        tracker_data_dir = tmp_path / "trackers" / tracker_name / "data"
        tracker_data_dir.mkdir(parents=True)
        shutil.copy2(results_path, tracker_data_dir / f"{seq_name}.txt")

        dataset = trackeval.datasets.MotChallenge2DBox(
            {
                "GT_FOLDER": str(tmp_path / "gt"),
                "TRACKERS_FOLDER": str(tmp_path / "trackers"),
                "TRACKERS_TO_EVAL": [tracker_name],
                "TRACKER_SUB_FOLDER": "data",
                "SEQMAP_FILE": str(seqmap_path),
                "SKIP_SPLIT_FOL": True,
                "CLASSES_TO_EVAL": ["pedestrian"],
            }
        )
        evaluator = trackeval.Evaluator(
            {
                "USE_PARALLEL": False,
                "PRINT_RESULTS": False,
                "PRINT_CONFIG": False,
                "OUTPUT_SUMMARY": False,
                "OUTPUT_DETAILED": False,
                "PLOT_CURVES": False,
            }
        )
        results, _ = evaluator.evaluate([dataset], [trackeval.metrics.HOTA()])

        hota_scores = results["MotChallenge2DBox"][tracker_name][seq_name]["pedestrian"]["HOTA"][
            "HOTA"
        ]
        return float(np.mean(hota_scores))
