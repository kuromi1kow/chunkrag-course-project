#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from chunkrag.eaai_phase2.analysis import run_phase2_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen EAAI Phase 2 held-out analysis without model inference."
    )
    parser.add_argument(
        "--config",
        default="configs/eaai_phase2/techqa_adaptive_v1.json",
    )
    parser.add_argument("--include-mistral", action="store_true")
    args = parser.parse_args()
    outputs = run_phase2_analysis(args.config, include_mistral=args.include_mistral)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
