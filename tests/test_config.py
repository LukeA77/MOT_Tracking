"""Tests for config loading, merging, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mot_pipeline.config import Config, load_config, parse_cli_overrides


def test_default_config_loads(configs_dir: Path) -> None:
    config = load_config(configs_dir / "default.yaml")
    assert isinstance(config, Config)
    assert config.detection.epochs == 30
    assert config.split.eval_sequence == "MOT17-09"


def test_smoke_override_merges_on_top_of_default(configs_dir: Path) -> None:
    config = load_config(configs_dir / "default.yaml", override_path=configs_dir / "smoke.yaml")
    assert config.detection.epochs == 1
    assert config.detection.device == "cpu"
    assert config.detection.fraction == 0.02
    assert config.benchmark.n_frames == 10
    # Unrelated sections fall through unchanged from default.yaml.
    assert config.export.opset == 13


def test_cli_overrides_take_precedence_over_files(configs_dir: Path) -> None:
    config = load_config(
        configs_dir / "default.yaml",
        override_path=configs_dir / "smoke.yaml",
        cli_overrides=["detection.epochs=5", "detection.device=cpu"],
    )
    assert config.detection.epochs == 5


def test_parse_cli_overrides_infers_types() -> None:
    overrides = parse_cli_overrides(
        ["detection.epochs=5", "detection.device=cpu", "split.eval_sequence=MOT17-09"]
    )
    assert overrides == {
        "detection": {"epochs": 5, "device": "cpu"},
        "split": {"eval_sequence": "MOT17-09"},
    }


def test_missing_config_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_sequence_without_gt_is_rejected(configs_dir: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(
            configs_dir / "default.yaml",
            cli_overrides=["split.train_sequences=[MOT17-01]"],
        )


def test_overlapping_train_val_is_rejected(configs_dir: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(
            configs_dir / "default.yaml",
            cli_overrides=["split.val_sequences=[MOT17-09, MOT17-02]"],
        )


def test_eval_sequence_must_be_in_val_sequences(configs_dir: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(
            configs_dir / "default.yaml",
            cli_overrides=["split.eval_sequence=MOT17-13"],
        )


def test_out_of_range_visibility_is_rejected(configs_dir: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(configs_dir / "default.yaml", cli_overrides=["convert.min_visibility=1.5"])


def test_negative_epochs_is_rejected(configs_dir: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(configs_dir / "default.yaml", cli_overrides=["detection.epochs=-1"])
