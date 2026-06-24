#!/usr/bin/env python
"""Thin CLI: load config, fine-tune the detector, report mAP."""

from __future__ import annotations

from mot_pipeline.detection.train import train_detector
from mot_pipeline.utils.cli import build_arg_parser, resolve_config


def main() -> None:
    """Parse args, load config, and run detector fine-tuning."""
    parser = build_arg_parser("Fine-tune the detector on the converted MOT17 dataset.")
    args = parser.parse_args()
    config = resolve_config(args)

    result = train_detector(config)

    print(f"Best checkpoint: {result.best_weights_path}")
    print(f"mAP@0.5: {result.map50:.4f}")
    print(f"mAP@0.5:0.95: {result.map50_95:.4f}")


if __name__ == "__main__":
    main()
