"""PyTorch vs ONNX Runtime inference speed benchmarking harness."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from mot_pipeline.config import Config
from mot_pipeline.data.mot17 import discover_train_sequences, read_seqinfo
from mot_pipeline.utils.logging import get_logger
from mot_pipeline.utils.metrics_io import update_metrics
from mot_pipeline.utils.paths import sequence_image_dir
from mot_pipeline.utils.timing import timer


@dataclass(frozen=True)
class SpeedResult:
    """Mean inference speed for one runtime/device combination."""

    runtime: str
    device: str
    avg_ms_per_frame: float
    fps: float


def benchmark_callable(
    predict_fn: Callable[[object], object],
    frames: Sequence[object],
    warmup: int,
    runtime: str,
    device: str,
) -> SpeedResult:
    """Time repeated calls to ``predict_fn`` over ``frames``, discarding warmup iterations.

    Decoupled from any specific model so it can be exercised in tests with a
    dummy callable, and reused for both the PyTorch and ONNX Runtime paths.

    Args:
        predict_fn: A single-frame inference callable, e.g. ``lambda frame: model(frame)``.
        frames: Pre-loaded (or path) frames to run inference on, in order.
            Must contain more than ``warmup`` items.
        warmup: Number of leading iterations to run and discard before timing begins.
        runtime: Label for the runtime under test, e.g. ``"pytorch"`` or ``"onnxruntime"``.
        device: Label for the device under test, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        The measured mean latency and throughput.

    Raises:
        ValueError: If ``frames`` has ``warmup`` or fewer items.
    """
    if len(frames) <= warmup:
        raise ValueError(
            f"Need more than {warmup} frames to measure after warmup, got {len(frames)}"
        )

    for frame in frames[:warmup]:
        predict_fn(frame)

    timed_frames = frames[warmup:]
    with timer() as elapsed:
        for frame in timed_frames:
            predict_fn(frame)

    avg_seconds = elapsed.seconds / len(timed_frames)
    return SpeedResult(
        runtime=runtime,
        device=device,
        avg_ms_per_frame=avg_seconds * 1000,
        fps=(1.0 / avg_seconds) if avg_seconds > 0 else float("inf"),
    )


def run_speed_benchmark(
    config: Config, pt_weights_path: Path, onnx_weights_path: Path
) -> list[SpeedResult]:
    """Benchmark PyTorch vs ONNX Runtime inference speed across configured devices.

    Args:
        config: Validated pipeline configuration.
        pt_weights_path: Path to the PyTorch (``.pt``) checkpoint.
        onnx_weights_path: Path to the exported ONNX model.

    Returns:
        One :class:`SpeedResult` per ``(runtime, device)`` combination.

    Raises:
        FileNotFoundError: If a weights file or the eval sequence is missing.
        ValueError: If ``benchmark.n_frames`` exceeds the eval sequence length.
    """
    from ultralytics import YOLO

    logger = get_logger(__name__, config.log_level)
    for path in (pt_weights_path, onnx_weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"Weights file not found: {path}")

    frame_paths = [str(p) for p in _sample_frame_paths(config)]
    pt_model = YOLO(str(pt_weights_path))
    onnx_model = YOLO(str(onnx_weights_path))

    results: list[SpeedResult] = []
    for device in config.benchmark.devices:
        for runtime, model in (("pytorch", pt_model), ("onnxruntime", onnx_model)):
            predict_fn = partial(_predict_one, model, imgsz=config.detection.imgsz, device=device)
            result = benchmark_callable(
                predict_fn,
                frame_paths,
                config.benchmark.warmup,
                runtime=runtime,
                device=device,
            )
            results.append(result)
            logger.info(
                "%s/%s: %.2f ms/frame (%.1f FPS)",
                runtime,
                device,
                result.avg_ms_per_frame,
                result.fps,
            )

    update_metrics(
        config.paths.outputs_dir / "metrics.json", "speed", [asdict(result) for result in results]
    )
    return results


def _predict_one(model: object, frame: object, imgsz: int, device: str) -> object:
    return model.predict(frame, imgsz=imgsz, device=device, verbose=False)  # type: ignore[attr-defined]


def _sample_frame_paths(config: Config) -> list[Path]:
    available = discover_train_sequences(config.paths.raw_dir)
    eval_sequence = config.split.eval_sequence
    if eval_sequence not in available:
        raise FileNotFoundError(
            f"eval_sequence {eval_sequence!r} not found under {config.paths.raw_dir}"
        )
    seq_dir = available[eval_sequence]
    seqinfo = read_seqinfo(seq_dir / "seqinfo.ini")
    if seqinfo.seq_length < config.benchmark.n_frames:
        raise ValueError(
            f"benchmark.n_frames={config.benchmark.n_frames} exceeds eval sequence length "
            f"{seqinfo.seq_length} for {eval_sequence}"
        )
    img_dir = sequence_image_dir(seq_dir)
    return [
        img_dir / f"{frame:06d}{seqinfo.im_ext}"
        for frame in range(1, config.benchmark.n_frames + 1)
    ]
