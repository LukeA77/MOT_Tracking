#!/usr/bin/env python
"""Thin CLI: export the detector to ONNX and verify PyTorch/ONNX parity."""

from __future__ import annotations

from pathlib import Path

from mot_pipeline.data.mot17 import discover_train_sequences, read_seqinfo
from mot_pipeline.export.onnx import check_parity, export_to_onnx
from mot_pipeline.utils.cli import build_arg_parser, resolve_config
from mot_pipeline.utils.paths import sequence_image_dir


def main() -> None:
    """Parse args, load config, export to ONNX, and check parity against PyTorch."""
    parser = build_arg_parser("Export the detector to ONNX and verify parity with PyTorch.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="PyTorch checkpoint to export. Defaults to <weights_dir>/best.pt.",
    )
    args = parser.parse_args()
    config = resolve_config(args)
    weights_path = args.weights if args.weights else config.paths.weights_dir / "best.pt"

    onnx_path = export_to_onnx(config, weights_path)
    print(f"Exported ONNX model: {onnx_path}")

    available = discover_train_sequences(config.paths.raw_dir)
    seq_dir = available[config.split.eval_sequence]
    seqinfo = read_seqinfo(seq_dir / "seqinfo.ini")
    sample_image = sequence_image_dir(seq_dir) / f"{1:06d}{seqinfo.im_ext}"

    parity = check_parity(weights_path, onnx_path, sample_image, config.detection.imgsz)
    status = "OK" if parity.parity_ok else "MISMATCH"
    print(f"Parity check: {status} (max abs diff: {parity.max_abs_diff_px:.2f}px)")


if __name__ == "__main__":
    main()
