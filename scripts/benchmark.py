#!/usr/bin/env python
"""Thin CLI: benchmark PyTorch vs ONNX inference speed across configured devices."""

from __future__ import annotations

from pathlib import Path

from mot_pipeline.benchmark.speed import run_speed_benchmark
from mot_pipeline.utils.cli import build_arg_parser, resolve_config


def main() -> None:
    """Parse args, load config, and run the speed benchmark."""
    parser = build_arg_parser("Benchmark PyTorch vs ONNX inference speed.")
    parser.add_argument("--weights", type=Path, default=None, help="PyTorch checkpoint.")
    parser.add_argument("--onnx-weights", type=Path, default=None, help="Exported ONNX model.")
    args = parser.parse_args()
    config = resolve_config(args)
    weights_path = args.weights if args.weights else config.paths.weights_dir / "best.pt"
    onnx_weights_path = (
        args.onnx_weights if args.onnx_weights else config.paths.weights_dir / "best.onnx"
    )

    results = run_speed_benchmark(config, weights_path, onnx_weights_path)

    print(f"{'runtime':12s} {'device':8s} {'ms/frame':>10s} {'fps':>8s}")
    for result in results:
        row = f"{result.runtime:12s} {result.device:8s} {result.avg_ms_per_frame:10.2f}"
        print(f"{row} {result.fps:8.1f}")


if __name__ == "__main__":
    main()
