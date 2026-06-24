"""Shared pytest fixtures: repo paths and a synthetic MOT17 sequence."""

from __future__ import annotations

from pathlib import Path

import pytest

_SEQINFO_TEMPLATE = """[Sequence]
name=MOT17-02-FRCNN
imDir=img1
frameRate=30
seqLength=3
imWidth=640
imHeight=480
imExt=.jpg
"""

# Columns: frame,id,bb_left,bb_top,bb_width,bb_height,consider_flag,class,visibility
#
# frame 1: row 1 is a clean kept box; row 2 overflows the right/bottom border
#          (tests clipping to [0,1]); row 3 is a non-pedestrian class (filtered);
#          row 4 has consider_flag=0 (filtered); row 5 is below the visibility
#          threshold used in tests (filtered).
# frame 2: no rows at all -> must produce an empty (but present) label file.
# frame 3: one ordinary kept box.
_GT_TEMPLATE = """1,1,100,50,40,80,1,1,1.0
1,2,620,460,80,80,1,1,1.0
1,3,10,10,20,20,1,2,1.0
1,4,10,10,20,20,0,1,1.0
1,5,10,10,20,20,1,1,0.05
3,6,200,150,30,60,1,1,0.8
"""


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def configs_dir(repo_root: Path) -> Path:
    """Absolute path to the ``configs/`` directory."""
    return repo_root / "configs"


@pytest.fixture
def mot17_sequence_dir(tmp_path: Path) -> Path:
    """Write a synthetic MOT17-FRCNN sequence (``seqinfo.ini`` + ``gt/gt.txt``) to disk.

    Returns:
        Path to the sequence directory (``<tmp>/MOT17-02-FRCNN``).
    """
    seq_dir = tmp_path / "MOT17-02-FRCNN"
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "seqinfo.ini").write_text(_SEQINFO_TEMPLATE, encoding="utf-8")
    (seq_dir / "gt" / "gt.txt").write_text(_GT_TEMPLATE, encoding="utf-8")
    return seq_dir
