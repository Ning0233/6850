#!/usr/bin/env python3
"""HESPERIA REleASE entrypoint wrapper.

This wrapper delegates to RELease.py and defaults to the hesperia-release mode
when no --model argument is provided.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "hesperia-release"])

    target = Path(__file__).with_name("RELease.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
