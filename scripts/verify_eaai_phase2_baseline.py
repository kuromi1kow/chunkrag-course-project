#!/usr/bin/env python3
from __future__ import annotations

import json

from chunkrag.eaai_phase2.integrity import verify_baseline, verify_protocol_commit


def main() -> None:
    baseline = verify_baseline()
    protocol_sha256 = verify_protocol_commit()
    print(
        json.dumps(
            {
                "status": "passed",
                "verified_files": baseline.verified_files,
                "tree_sha256": baseline.tree_sha256,
                "protocol_sha256": protocol_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
