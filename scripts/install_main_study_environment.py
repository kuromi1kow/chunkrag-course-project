#!/usr/bin/env python3
"""Install the exact tracked transitive environment before canonical execution."""

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    lock = Path("requirements-main-study.transitive.json")
    payload = json.loads(lock.read_text(encoding="utf-8"))
    if sys.version.split()[0] != payload["python"]:
        raise SystemExit(f"Python {payload['python']} is required")
    requirements = [f"{row['name']}=={row['version']}" for row in payload["packages"]]
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", *requirements], check=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
