"""Compute tracking metrics for every tracker and assemble the results report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mot_pipeline.config import Config
from mot_pipeline.data.mot17 import SeqInfo, discover_train_sequences, read_seqinfo
from mot_pipeline.evaluation.mot_metrics import TrackingMetrics, compute_tracking_metrics
from mot_pipeline.utils.logging import get_logger
from mot_pipeline.utils.metrics_io import load_metrics, update_metrics
from mot_pipeline.utils.mot_format import read_mot_results
from mot_pipeline.utils.paths import sequence_image_dir


def evaluate_trackers(config: Config) -> list[TrackingMetrics]:
    """Compute HOTA/MOTA/IDF1 for every configured tracker and update ``metrics.json``.

    Args:
        config: Validated pipeline configuration.

    Returns:
        One :class:`TrackingMetrics` per configured tracker.
    """
    logger = get_logger(__name__, config.log_level)
    results: list[TrackingMetrics] = []
    for tracker_filename in config.tracking.trackers:
        tracker_name = Path(tracker_filename).stem
        results_path = config.paths.outputs_dir / f"{config.split.eval_sequence}_{tracker_name}.txt"
        metrics = compute_tracking_metrics(config, tracker_name, results_path)
        results.append(metrics)
        hota_display = f"{metrics.hota:.3f}" if metrics.hota is not None else "unavailable"
        logger.info(
            "%s: MOTA=%.3f IDF1=%.3f HOTA=%s",
            tracker_name,
            metrics.mota,
            metrics.idf1,
            hota_display,
        )

    update_metrics(
        config.paths.outputs_dir / "metrics.json",
        "tracking",
        {
            m.tracker: {
                "mota": m.mota,
                "idf1": m.idf1,
                "num_switches": m.num_switches,
                "hota": m.hota,
            }
            for m in results
        },
    )
    return results


def render_results_summary(config: Config) -> Path:
    """Render the 3-panel results figure: sample detections, ID persistence, speed.

    Args:
        config: Validated pipeline configuration.

    Returns:
        Path to the written ``outputs/results_summary.png``.

    Raises:
        FileNotFoundError: If the eval sequence's first frame cannot be read.
    """
    import matplotlib.pyplot as plt

    metrics = load_metrics(config.paths.outputs_dir / "metrics.json")

    available = discover_train_sequences(config.paths.raw_dir)
    seq_dir = available[config.split.eval_sequence]
    seqinfo = read_seqinfo(seq_dir / "seqinfo.ini")
    img_dir = sequence_image_dir(seq_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_sample_frame(axes[0], img_dir, seqinfo, config)
    _plot_id_persistence(axes[1], config, seqinfo)
    _plot_speed_comparison(axes[2], metrics.get("speed", []))
    fig.tight_layout()

    out_path = config.paths.outputs_dir / "results_summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_sample_frame(ax: Any, img_dir: Path, seqinfo: SeqInfo, config: Config) -> None:
    import cv2

    image_path = img_dir / f"{1:06d}{seqinfo.im_ext}"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read sample frame at {image_path}")

    default_tracker = Path(config.tracking.trackers[0]).stem
    results_path = config.paths.outputs_dir / f"{config.split.eval_sequence}_{default_tracker}.txt"
    if results_path.is_file():
        for record in read_mot_results(results_path):
            if record.frame != 1:
                continue
            x1, y1 = int(record.bb_left), int(record.bb_top)
            x2, y2 = int(record.bb_left + record.bb_width), int(record.bb_top + record.bb_height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Sample frame: {config.split.eval_sequence} ({default_tracker})")
    ax.axis("off")


def _plot_id_persistence(ax: Any, config: Config, seqinfo: SeqInfo) -> None:
    for tracker_filename in config.tracking.trackers:
        tracker_name = Path(tracker_filename).stem
        results_path = config.paths.outputs_dir / f"{config.split.eval_sequence}_{tracker_name}.txt"
        if not results_path.is_file():
            continue
        counts = [0] * (seqinfo.seq_length + 1)
        for record in read_mot_results(results_path):
            counts[record.frame] += 1
        ax.plot(range(1, seqinfo.seq_length + 1), counts[1:], label=tracker_name)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Active tracks")
    ax.set_title("Track ID persistence")
    ax.legend()


def _plot_speed_comparison(ax: Any, speed_rows: list[dict[str, Any]]) -> None:
    if not speed_rows:
        ax.set_title("Speed comparison (no data)")
        ax.axis("off")
        return
    labels = [f"{row['runtime']}\n{row['device']}" for row in speed_rows]
    fps_values = [row["fps"] for row in speed_rows]
    ax.bar(labels, fps_values, color="steelblue")
    ax.set_ylabel("FPS")
    ax.set_title("Inference speed: PyTorch vs ONNX")
