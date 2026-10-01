#!/usr/bin/env python3
"""Verify integrity, anonymity status, and matrix coverage of an artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from letta_research_chat.anonymized_artifacts import verify_anonymized_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    result = verify_anonymized_artifact(args.artifact)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
