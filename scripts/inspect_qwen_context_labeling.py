#!/usr/bin/env python3
"""Inspect imported Qwen context labels in a readable terminal report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import sys
import textwrap


DEFAULT_PIPELINE = Path(
    "research_outputs/"
    "privacy-pipeline-cimemories-dataset_cimemories-raw_mem3_20260912_054751"
)

COLORS = {
    "necessary": "\033[1;32m",
    "ambiguous": "\033[1;33m",
    "inappropriate": "\033[1;31m",
    "heading": "\033[1;36m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}
SYMBOLS = {"necessary": "✓", "ambiguous": "?", "inappropriate": "✗"}
SHORT = {"necessary": "N", "ambiguous": "A", "inappropriate": "I"}


def _paint(text: str, style: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{COLORS[style]}{text}{COLORS['reset']}"


def _bar(share: float, private: float, width: int = 20) -> str:
    share_width = round(max(0.0, min(1.0, share)) * width)
    private_width = round(max(0.0, min(1.0, private)) * width)
    if share_width + private_width > width:
        private_width = width - share_width
    abstain_width = width - share_width - private_width
    return "█" * share_width + "▓" * private_width + "░" * abstain_width


def _label_files(root: Path) -> list[tuple[int, int, Path]]:
    results = []
    for path in sorted(
        root.glob(
            "personas/persona_*/pipeline/context_labelings/"
            "context_*/context_labeling_cimemories.json"
        )
    ):
        try:
            persona_idx = int(path.parents[3].name.removeprefix("persona_"))
            context_idx = int(path.parent.name.removeprefix("context_"))
        except ValueError:
            continue
        results.append((persona_idx, context_idx, path))
    return results


def _scenario_line(persona_idx: int, context_idx: int, path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        f"persona={persona_idx:02d} context={context_idx:02d}  "
        f"{data.get('persona_name')} → {data.get('recipient')}  |  {data.get('task')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one Qwen-generated CIMemories context labeling. The script "
            "is read-only and makes no API calls."
        )
    )
    parser.add_argument(
        "pipeline_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_PIPELINE,
        help=f"dataset pipeline directory (default: {DEFAULT_PIPELINE})",
    )
    parser.add_argument("--persona", type=int, help="persona index")
    parser.add_argument("--context", type=int, help="context index")
    parser.add_argument("--seed", type=int, help="reproducible random selection seed")
    parser.add_argument(
        "--label",
        choices=("all", "necessary", "ambiguous", "inappropriate"),
        default="all",
        help="show only one final-label category (default: all)",
    )
    parser.add_argument(
        "--search",
        help="show only attributes containing this case-insensitive text",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="show the three privacy-persona distributions under every attribute",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list matching scenarios instead of opening one",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="disable ANSI colors",
    )
    args = parser.parse_args()

    root = args.pipeline_dir.resolve()
    available = _label_files(root)
    if args.persona is not None:
        available = [item for item in available if item[0] == args.persona]
    if args.context is not None:
        available = [item for item in available if item[1] == args.context]
    if not available:
        raise SystemExit("No context-labeling artifacts match the requested selection.")

    if args.list:
        for persona_idx, context_idx, path in available:
            print(_scenario_line(persona_idx, context_idx, path))
        return

    rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()
    persona_idx, context_idx, path = rng.choice(available)
    data = json.loads(path.read_text(encoding="utf-8"))
    color = not args.plain and sys.stdout.isatty()
    width = max(72, min(shutil.get_terminal_size((100, 24)).columns, 120))

    heading = f" QWEN CONTEXT LABEL AUDIT · P{persona_idx:02d} C{context_idx:02d} "
    print(_paint(heading.center(width, "═"), "heading", color))
    print(f"Persona   : {data.get('persona_name')} (index {persona_idx})")
    print(f"Scenario  : {data.get('recipient')} — {data.get('task')}")
    print(f"Judge     : {data.get('llm_model')}")
    print(f"Provenance: {data.get('label_source')} · {data.get('source_labels_file')}")
    print(f"Artifact  : {path}")
    print(f"Discarded : {data.get('context_discarded', False)}")

    priors = data.get("westin_priors") or {}
    if priors:
        print(
            "Westin mix: "
            f"fundamentalist={float(priors.get('0', 0)):.0%}, "
            f"pragmatic={float(priors.get('1', 0)):.0%}, "
            f"unconcerned={float(priors.get('2', 0)):.0%}"
        )

    attributes = [
        item
        for item in data.get("attribute_results") or []
        if isinstance(item, dict) and isinstance(item.get("attribute"), str)
    ]
    counts = Counter(item.get("final_label_name", "unknown") for item in attributes)
    print(
        "Labels    : "
        + " · ".join(
            _paint(f"{SHORT[label]}={counts[label]}", label, color)
            for label in ("necessary", "ambiguous", "inappropriate")
        )
        + f" · total={len(attributes)}"
    )

    search = args.search.casefold() if args.search else None
    shown = 0
    for label in ("necessary", "ambiguous", "inappropriate"):
        if args.label not in ("all", label):
            continue
        items = [item for item in attributes if item.get("final_label_name") == label]
        if search:
            items = [item for item in items if search in item["attribute"].casefold()]
        if not items:
            continue
        print()
        title = f" {SYMBOLS[label]} {label.upper()} ({len(items)}) "
        print(_paint(title.ljust(width, "─"), label, color))
        for item in items:
            mixture = item.get("mixture_distribution") or {}
            share = float(mixture.get("share") or 0.0)
            private = float(mixture.get("private") or 0.0)
            abstain = float(mixture.get("abstain") or 0.0)
            entropy = item.get("mixture_entropy")
            prefix = (
                f"{SYMBOLS[label]} S {share:5.1%}  P {private:5.1%}  "
                f"X {abstain:5.1%}  {_bar(share, private)}"
            )
            if isinstance(entropy, (int, float)):
                prefix += f"  H={entropy:.2f}"
            print(_paint(prefix, label, color))
            indent = " " * 4
            wrapped = textwrap.wrap(
                item["attribute"],
                width=max(30, width - len(indent)),
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            for line in wrapped:
                print(f"{indent}{line}")

            if args.details:
                distributions = item.get("persona_distributions") or {}
                for persona_name in (
                    "privacy_fundamentalist",
                    "pragmatic",
                    "unconcerned",
                ):
                    distribution = distributions.get(persona_name) or {}
                    print(
                        _paint(
                            f"      {persona_name:24} "
                            f"share={float(distribution.get('share') or 0):5.1%}  "
                            f"private={float(distribution.get('private') or 0):5.1%}  "
                            f"abstain={float(distribution.get('abstain') or 0):5.1%}",
                            "dim",
                            color,
                        )
                    )
            shown += 1

    if shown == 0:
        print("\nNo attributes matched the selected label/search filters.")
    else:
        print(
            _paint(
                f"\nShown {shown}/{len(attributes)} attributes. "
                "S=share, P=private, X=abstain, H=mixture entropy.",
                "dim",
                color,
            )
        )


if __name__ == "__main__":
    main()
