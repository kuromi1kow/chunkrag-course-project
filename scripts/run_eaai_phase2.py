#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chunkrag.eaai_phase2.experiment import Phase2Experiment


DEFAULT_CONFIG = "configs/eaai_phase2/techqa_adaptive_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen prospective EAAI Phase 2 experiment in resumable stages."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("partition")
    retrieval = subparsers.add_parser("retrieve")
    retrieval.add_argument("--split", required=True, choices=("development", "heldout_test"))
    generation = subparsers.add_parser("generate")
    generation.add_argument("--split", required=True, choices=("development", "heldout_test"))
    generation.add_argument("--generator", required=True, choices=("qwen", "mistral"))
    subparsers.add_parser("fit-gate")
    subparsers.add_parser("run-qwen")
    subparsers.add_parser("run-mistral")
    subparsers.add_parser("dry-run")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = Phase2Experiment(args.config)
    if args.command == "preflight":
        result = experiment.preflight(require_committed_implementation=True)
    elif args.command == "partition":
        partition = experiment.prepare_partition()
        result = {
            "partition_sha256": partition.partition_sha256,
            "development": len(partition.development),
            "heldout_test": len(partition.heldout_test),
            "reserve": len(partition.reserve),
        }
    elif args.command == "retrieve":
        result = {"output": str(experiment.run_retrieval(args.split, device=args.device))}
    elif args.command == "generate":
        result = {
            "output": str(
                experiment.run_generation(args.split, args.generator, device=args.device)
            )
        }
    elif args.command == "fit-gate":
        result = {"gate_manifest": str(experiment.fit_development_gate())}
    elif args.command == "run-qwen":
        result = experiment.run_qwen(device=args.device)
    elif args.command == "run-mistral":
        result = {"output": str(experiment.run_mistral(device=args.device))}
    elif args.command == "dry-run":
        result = {
            "preflight": experiment.preflight(require_committed_implementation=True),
            "plan": [
                "partition 608 eligible TechQA questions into 200/200/208",
                "paired development retrieval and Qwen generation (800 pairs, 1600 generations)",
                "freeze the development-trained gate",
                "paired held-out retrieval and Qwen generation (800 pairs, 1600 generations)",
                "run one question-level confirmatory F1 test",
                "optionally run held-out Mistral replication",
            ],
            "full_inference_started": False,
        }
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
