#!/usr/bin/env python3
"""One-command protocol/repository verification (Specification Sections 24--25)."""

import argparse
import json

from chunkrag.mainstudy.protocol import repo_root
from chunkrag.mainstudy.validation import validate_repository


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-environment", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_repository(repo_root(), check_environment=args.check_environment), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
