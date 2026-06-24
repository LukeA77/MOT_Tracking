"""Resolve a validated :class:`SplitConfig` against sequences discovered on disk.

:class:`~mot_pipeline.config.SplitConfig` already guarantees (at config-load
time) that train/val sequences are disjoint and that every named sequence
has public ground truth -- see its validators and ``tests/test_config.py``.
This module's job is the next step: confirming each configured sequence
actually exists among the sequences discovered by
:func:`mot_pipeline.data.mot17.discover_train_sequences`, and building the
resolved ``{name: path}`` mappings the data-preparation stage iterates over.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mot_pipeline.config import SplitConfig


@dataclass(frozen=True)
class ResolvedSplit:
    """Train/val sequence names resolved to their on-disk directories."""

    train: dict[str, Path]
    val: dict[str, Path]
    eval_sequence: str


def resolve_split(split_config: SplitConfig, available_sequences: dict[str, Path]) -> ResolvedSplit:
    """Resolve a split's sequence names to discovered sequence directories.

    Args:
        split_config: Validated split configuration.
        available_sequences: Mapping from base sequence name to directory,
            as returned by :func:`mot_pipeline.data.mot17.discover_train_sequences`.

    Returns:
        The resolved train/val directory mappings.

    Raises:
        FileNotFoundError: If a configured sequence is not among
            ``available_sequences``.
    """
    return ResolvedSplit(
        train=_resolve_subset(
            split_config.train_sequences, available_sequences, "split.train_sequences"
        ),
        val=_resolve_subset(split_config.val_sequences, available_sequences, "split.val_sequences"),
        eval_sequence=split_config.eval_sequence,
    )


def _resolve_subset(
    names: list[str], available_sequences: dict[str, Path], field_name: str
) -> dict[str, Path]:
    missing = [name for name in names if name not in available_sequences]
    if missing:
        raise FileNotFoundError(
            f"{field_name} references sequences not found under the raw data directory: "
            f"{missing}. Available sequences: {sorted(available_sequences)}"
        )
    return {name: available_sequences[name] for name in names}
