"""Pydantic-validated configuration schema and loader.

Every tunable in the pipeline (paths, hyperparameters, thresholds, sequence
lists, device choice) is declared here and validated at startup. Library and
CLI code must read these values from a :class:`Config` instance rather than
hardcoding literals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# The only MOT17 sequences shipped with public ground truth. Test sequences
# (01, 03, 06, 07, 08, 12, 14) have no GT and must never appear in a split.
MOT17_GT_SEQUENCES: frozenset[str] = frozenset(
    {"MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"}
)


class PathsConfig(BaseModel):
    """Filesystem locations used throughout the pipeline."""

    raw_dir: Path = Path("data/MOT17")
    converted_dir: Path = Path("data/converted")
    weights_dir: Path = Path("weights")
    outputs_dir: Path = Path("outputs")
    configs_dir: Path = Path("configs")


class SplitConfig(BaseModel):
    """Sequence-level train/val split. See spec Section 8.1 for rationale."""

    train_sequences: list[str] = Field(
        default_factory=lambda: ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-11", "MOT17-13"]
    )
    val_sequences: list[str] = Field(default_factory=lambda: ["MOT17-09", "MOT17-10"])
    eval_sequence: str = "MOT17-09"

    @field_validator("train_sequences", "val_sequences")
    @classmethod
    def _sequences_have_gt(cls, sequences: list[str]) -> list[str]:
        unknown = [seq for seq in sequences if seq not in MOT17_GT_SEQUENCES]
        if unknown:
            raise ValueError(
                f"Sequences {unknown} have no public ground truth and cannot be used "
                f"in a split. Valid sequences are: {sorted(MOT17_GT_SEQUENCES)}"
            )
        return sequences

    @model_validator(mode="after")
    def _train_val_disjoint_and_eval_in_val(self) -> SplitConfig:
        overlap = set(self.train_sequences) & set(self.val_sequences)
        if overlap:
            raise ValueError(f"train_sequences and val_sequences overlap: {sorted(overlap)}")
        if self.eval_sequence not in self.val_sequences:
            raise ValueError(
                f"eval_sequence {self.eval_sequence!r} must be one of val_sequences "
                f"{self.val_sequences}"
            )
        return self


class ConvertConfig(BaseModel):
    """MOT17 ground-truth -> YOLO label conversion filters."""

    keep_classes: list[int] = Field(default_factory=lambda: [1])
    use_consider_flag: bool = True
    min_visibility: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("keep_classes")
    @classmethod
    def _non_empty_classes(cls, keep_classes: list[int]) -> list[int]:
        if not keep_classes:
            raise ValueError("keep_classes must not be empty")
        return keep_classes


class DetectionConfig(BaseModel):
    """Detector fine-tuning hyperparameters."""

    model: str = "yolo26n.pt"
    epochs: int = Field(default=30, gt=0)
    imgsz: int = Field(default=640, gt=0)
    batch: int = Field(default=16, gt=0)
    device: str = "0"
    fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)


class TrackingConfig(BaseModel):
    """Tracker selection and association thresholds."""

    trackers: list[str] = Field(default_factory=lambda: ["botsort.yaml", "bytetrack.yaml"])
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("trackers")
    @classmethod
    def _non_empty_trackers(cls, trackers: list[str]) -> list[str]:
        if not trackers:
            raise ValueError("trackers must not be empty")
        return trackers


class ExportConfig(BaseModel):
    """ONNX export options."""

    format: str = "onnx"
    opset: int = Field(default=13, gt=0)
    dynamic: bool = True
    quantize_int8: bool = False


class BenchmarkConfig(BaseModel):
    """Inference speed benchmarking harness options."""

    n_frames: int = Field(default=100, gt=0)
    warmup: int = Field(default=10, ge=0)
    devices: list[str] = Field(default_factory=lambda: ["cpu"])

    @field_validator("devices")
    @classmethod
    def _non_empty_devices(cls, devices: list[str]) -> list[str]:
        if not devices:
            raise ValueError("devices must not be empty")
        return devices

    @model_validator(mode="after")
    def _warmup_not_exceeding_n_frames(self) -> BenchmarkConfig:
        if self.warmup >= self.n_frames:
            raise ValueError(
                f"warmup ({self.warmup}) must be smaller than n_frames ({self.n_frames})"
            )
        return self


class Config(BaseModel):
    """Root configuration object for the whole pipeline."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    convert: ConvertConfig = Field(default_factory=ConvertConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    log_level: str = "INFO"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_dotted_key(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``target[a][b][c] = value`` given a dotted key ``"a.b.c"``."""
    *parents, leaf = dotted_key.split(".")
    node = target
    for parent in parents:
        node = node.setdefault(parent, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set override {dotted_key!r}: {parent!r} is not a section")
    node[leaf] = value


def parse_cli_overrides(overrides: list[str]) -> dict[str, Any]:
    """Parse ``key.subkey=value`` CLI override strings into a nested dict.

    Args:
        overrides: Strings like ``"detection.epochs=1"`` or
            ``"detection.device=cpu"``. Values are parsed with
            :func:`yaml.safe_load` so ``"1"`` becomes an int, ``"true"``
            becomes a bool, etc.

    Returns:
        A nested dict suitable for deep-merging onto a loaded config dict.

    Raises:
        ValueError: If an override string is not of the form ``key=value``.
    """
    result: dict[str, Any] = {}
    for raw_override in overrides:
        if "=" not in raw_override:
            raise ValueError(f"Invalid override {raw_override!r}, expected key.subkey=value")
        dotted_key, raw_value = raw_override.split("=", 1)
        _set_dotted_key(result, dotted_key.strip(), yaml.safe_load(raw_value))
    return result


def load_config(
    path: Path,
    override_path: Path | None = None,
    cli_overrides: list[str] | None = None,
) -> Config:
    """Load, merge, and validate the pipeline configuration.

    Args:
        path: Path to the base YAML config (typically ``configs/default.yaml``).
        override_path: Optional YAML file deep-merged on top of ``path``
            (typically ``configs/smoke.yaml``).
        cli_overrides: Optional ``"key.subkey=value"`` strings, applied last
            and taking precedence over both YAML files.

    Returns:
        A validated :class:`Config` instance.

    Raises:
        FileNotFoundError: If ``path`` or ``override_path`` does not exist.
        pydantic.ValidationError: If the merged config fails validation.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    merged: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if override_path is not None:
        if not override_path.is_file():
            raise FileNotFoundError(f"Override config file not found: {override_path}")
        override_data = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
        merged = _deep_merge(merged, override_data)

    if cli_overrides:
        merged = _deep_merge(merged, parse_cli_overrides(cli_overrides))

    return Config.model_validate(merged)
