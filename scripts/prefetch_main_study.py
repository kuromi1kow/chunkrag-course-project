#!/usr/bin/env python3
"""Cache pinned external artifacts before canonical offline inference (Specification Sections 7, 25)."""

import argparse

from chunkrag.mainstudy.data import load_pinned_dataset
from chunkrag.mainstudy.protocol import load_protocol_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--datasets", action="store_true")
    args = parser.parse_args()
    if not args.models and not args.datasets:
        parser.error("Select --models and/or --datasets")
    config = load_protocol_config()
    if args.models:
        from huggingface_hub import snapshot_download

        for spec in config["models"].values():
            path = snapshot_download(spec["repository"], revision=spec["revision"])
            print(f"cached model {spec['repository']} at {path}")
    if args.datasets:
        for name, spec in config["datasets"].items():
            value = load_pinned_dataset(spec)
            print(f"cached dataset {name} rows={len(value)} fingerprint={value._fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
