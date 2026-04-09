from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from chunkrag.pipeline import load_experiment_config, run_dataset_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to outputs/<timestamp>/.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"outputs/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []
    for dataset_spec in config["datasets"]:
        all_summaries.extend(run_dataset_experiments(config, dataset_spec, output_dir))

    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(output_dir / "all_results.json", "w", encoding="utf-8") as handle:
        json.dump(all_summaries, handle, indent=2)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
