#!/usr/bin/env python
"""Thin CLI: load config, convert raw MOT17 into the YOLO dataset format."""

from __future__ import annotations

from mot_pipeline.data.prepare import prepare_dataset
from mot_pipeline.utils.cli import build_arg_parser, resolve_config


def main() -> None:
    """Parse args, load config, and run dataset preparation."""
    parser = build_arg_parser("Convert raw MOT17 into a YOLO-format detection dataset.")
    args = parser.parse_args()
    config = resolve_config(args)

    summary = prepare_dataset(config)

    print(f"Converted {len(summary.sequences)} sequences:")
    for seq in summary.sequences:
        print(
            f"  {seq.subset:5s} {seq.sequence:12s} images={seq.n_images:6d} boxes={seq.n_boxes:6d}"
        )
    print(f"Total: {summary.total_images} images, {summary.total_boxes} boxes")
    print(f"dataset.yaml: {summary.dataset_yaml_path}")


if __name__ == "__main__":
    main()
