from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from chunkrag.pipeline import load_experiment_config, run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to outputs/<timestamp>/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing non-empty output directory before starting.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"outputs/{timestamp}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            parser.error(
                f"Output directory is not empty: {output_dir}. "
                "Choose a fresh directory or pass --overwrite."
            )
        shutil.rmtree(output_dir)
    all_summaries = run_experiment_suite(config, output_dir)
    print(f"Saved results to {output_dir}")
    print(f"Generated {len(all_summaries)} summary rows.")


if __name__ == "__main__":
    main()
