#!/usr/bin/env python3
"""Build the multi-model CIMemories paper-level comparative report."""

from __future__ import annotations

import argparse
from pathlib import Path

from letta_research_chat.comparative_report import build_comparative_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research_outputs"))
    parser.add_argument("--dataset", default="cimemories_raw")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()
    output = build_comparative_report(
        output_root=args.root,
        dataset_filter=None if args.dataset.lower() == "all" else args.dataset,
        output=args.output,
        inputs=args.inputs,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print((output / "REPORT.html").resolve())


if __name__ == "__main__":
    main()
