"""Tests for resolving a split against discovered sequence directories.

Note: that train/val are disjoint and that every configured sequence has
public ground truth is enforced by ``SplitConfig`` itself and covered in
``tests/test_config.py`` -- by the time a ``SplitConfig`` reaches
``resolve_split``, those invariants already hold. This module tests the
remaining job: cross-referencing configured names against what is actually
present on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mot_pipeline.config import SplitConfig
from mot_pipeline.data.split import resolve_split


def test_resolve_split_maps_names_to_discovered_directories() -> None:
    available = {
        "MOT17-02": Path("/data/MOT17/train/MOT17-02-FRCNN"),
        "MOT17-09": Path("/data/MOT17/train/MOT17-09-FRCNN"),
        "MOT17-10": Path("/data/MOT17/train/MOT17-10-FRCNN"),
    }
    split_config = SplitConfig(
        train_sequences=["MOT17-02"],
        val_sequences=["MOT17-09", "MOT17-10"],
        eval_sequence="MOT17-09",
    )

    resolved = resolve_split(split_config, available)

    assert resolved.train == {"MOT17-02": available["MOT17-02"]}
    assert resolved.val == {"MOT17-09": available["MOT17-09"], "MOT17-10": available["MOT17-10"]}
    assert resolved.eval_sequence == "MOT17-09"


def test_resolve_split_raises_clearly_when_sequence_missing_on_disk() -> None:
    available = {"MOT17-09": Path("/data/MOT17/train/MOT17-09-FRCNN")}
    split_config = SplitConfig(
        train_sequences=["MOT17-02"], val_sequences=["MOT17-09"], eval_sequence="MOT17-09"
    )

    with pytest.raises(FileNotFoundError, match="MOT17-02"):
        resolve_split(split_config, available)
