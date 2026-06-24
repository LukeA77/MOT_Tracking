#!/usr/bin/env python
"""Thin CLI: compute HOTA/MOTA/IDF1 for every tracker and render the results figure."""

from __future__ import annotations

from mot_pipeline.evaluation.report import evaluate_trackers, render_results_summary
from mot_pipeline.utils.cli import build_arg_parser, resolve_config


def main() -> None:
    """Parse args, load config, compute tracking metrics, and render the report."""
    parser = build_arg_parser("Compute tracking metrics and assemble the results report.")
    args = parser.parse_args()
    config = resolve_config(args)

    results = evaluate_trackers(config)
    for result in results:
        hota_display = f"{result.hota:.3f}" if result.hota is not None else "unavailable"
        row = f"{result.tracker:10s} MOTA={result.mota:.3f} IDF1={result.idf1:.3f}"
        print(f"{row} HOTA={hota_display}")

    figure_path = render_results_summary(config)
    print(f"Results figure: {figure_path}")


if __name__ == "__main__":
    main()
