#!/usr/bin/env python
"""Thin CLI: convert a few seconds of a tracked video into a README GIF."""

from __future__ import annotations

from pathlib import Path

from mot_pipeline.tracking.gif import video_to_gif
from mot_pipeline.utils.cli import build_arg_parser, resolve_config


def main() -> None:
    """Parse args, load config, and write a short GIF clip from a tracked video."""
    parser = build_arg_parser("Convert a few seconds of a tracked video into a README GIF.")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Source .mp4. Defaults to the first tracker's video.",
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Clip length in seconds.")
    parser.add_argument("--out", type=Path, default=None, help="Output .gif path.")
    args = parser.parse_args()
    config = resolve_config(args)

    default_tracker = Path(config.tracking.trackers[0]).stem
    video_path = args.video or config.paths.outputs_dir / f"tracked_output_{default_tracker}.mp4"
    out_path = args.out or config.paths.outputs_dir / "tracked_output.gif"

    gif_path = video_to_gif(video_path, out_path, seconds=args.seconds)
    print(f"GIF written: {gif_path}")


if __name__ == "__main__":
    main()
