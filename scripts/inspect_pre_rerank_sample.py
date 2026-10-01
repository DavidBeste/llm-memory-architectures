#!/usr/bin/env python3
"""Print one saved pre/post-rerank response pair for manual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


DEFAULT_PIPELINE = Path(
    "research_outputs/"
    "privacy-pipeline-cimemories-dataset_cimemories-raw_mem3_20260912_054751"
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _one(paths: list[Path], description: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"Could not find {description}")
    if len(paths) > 1:
        choices = "\n  ".join(str(path) for path in paths)
        raise ValueError(f"Found multiple {description} artifacts:\n  {choices}")
    return paths[0]


def _record_map(records: list[dict]) -> dict[tuple[int, int], dict]:
    result = {}
    for record in records:
        context_idx = record.get("context_idx")
        repeat_idx = record.get("repeat_idx")
        if isinstance(context_idx, int) and isinstance(repeat_idx, int):
            result[(context_idx, repeat_idx)] = record
    return result


def _print_exposures(
    condition: str,
    record: dict,
    labels: dict[str, str],
    *,
    show_judge_raw: bool,
) -> None:
    print(f"\n=== {condition} RESPONSE ===")
    print(record.get("assistant_response") or "(empty response)")
    print(f"\n=== {condition} JUDGED EXPOSURES ===")
    exposed = record.get("exposed_attributes") or {}
    if not exposed:
        print("(none)")
    for attribute, evidence in exposed.items():
        label = labels.get(attribute, "UNMATCHED KEY")
        canonical = attribute in set(record.get("attributes") or [])
        print(f"\nAttribute: {attribute}")
        print(f"Label: {label}")
        print(f"Canonical key: {canonical}")
        print(f"Evidence: {evidence}")

    if show_judge_raw:
        print(f"\n=== {condition} RAW JUDGE OUTPUT ===")
        raw = record.get("judge_response") or {}
        found_text = False
        for item in raw.get("output") or []:
            for content in item.get("content") or []:
                if content.get("type") == "output_text":
                    print(content.get("text") or "")
                    found_text = True
        if not found_text:
            print(json.dumps(raw, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one paired pre/post-rerank response without making API calls. "
            "If --context is omitted, a random available pair is selected."
        )
    )
    parser.add_argument(
        "pipeline_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_PIPELINE,
        help=f"dataset pipeline directory (default: {DEFAULT_PIPELINE})",
    )
    parser.add_argument("--persona", type=int, default=0, help="persona index (default: 0)")
    parser.add_argument("--context", type=int, help="context index; random when omitted")
    parser.add_argument("--repeat", type=int, help="repeat index; random when omitted")
    parser.add_argument(
        "--seed",
        type=int,
        help="optional random seed for reproducible random selection",
    )
    parser.add_argument(
        "--show-judge-raw",
        action="store_true",
        help="also print the complete saved judge text output",
    )
    args = parser.parse_args()

    root = args.pipeline_dir.resolve()
    persona = root / "personas" / f"persona_{args.persona:03d}"
    query = persona / "query"
    pre = root / "pre_rerank_response_evaluation" / "personas" / f"persona_{args.persona:03d}"

    source_records = _read_jsonl(query / "responses.jsonl")
    pre_generation_records = _read_jsonl(pre / "responses.jsonl")
    pre_records = _read_jsonl(pre / "exposed_attributes.jsonl")
    post_path = _one(
        sorted(query.glob("exposed_attributes_gpt-5.2_*.jsonl")),
        "post-rerank GPT-5.2 exposure",
    )
    post_records = _read_jsonl(post_path)

    pre_by_key = _record_map(pre_records)
    pre_generation_by_key = _record_map(pre_generation_records)
    post_by_key = _record_map(post_records)
    available = sorted(set(pre_by_key) & set(post_by_key))
    if args.context is not None:
        available = [key for key in available if key[0] == args.context]
    if args.repeat is not None:
        available = [key for key in available if key[1] == args.repeat]
    if not available:
        raise ValueError(
            "No paired sample matches the requested persona/context/repeat."
        )

    rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()
    context_idx, repeat_idx = rng.choice(available)
    pre_record = pre_by_key[(context_idx, repeat_idx)]
    post_record = post_by_key[(context_idx, repeat_idx)]
    pre_generation_record = pre_generation_by_key.get((context_idx, repeat_idx))
    if pre_generation_record is None:
        raise ValueError(
            f"No pre-rerank generation record found for context {context_idx}, "
            f"repeat {repeat_idx}"
        )

    source_record = next(
        (
            record
            for record in source_records
            if record.get("context_idx") == context_idx
            and record.get("repeat_idx") == 0
        ),
        None,
    )
    if source_record is None:
        raise ValueError(f"No source record found for context {context_idx}")

    label_path = (
        persona
        / "pipeline"
        / "context_labelings"
        / f"context_{context_idx:03d}"
        / "context_labeling_cimemories.json"
    )
    labeling = json.loads(label_path.read_text(encoding="utf-8"))
    labels = {
        item["attribute"]: item.get("final_label_name", "unknown")
        for item in labeling.get("attribute_results") or []
        if isinstance(item, dict) and isinstance(item.get("attribute"), str)
    }

    selected = set((source_record.get("rerank") or {}).get("selected_memories") or [])
    candidates = pre_generation_record.get("pre_rerank_facts") or []

    print(f"Pipeline: {root}")
    print(f"Persona: {args.persona} ({pre_record.get('persona_name')})")
    print(f"Context: {context_idx}")
    print(f"Repeat: {repeat_idx}")
    print(f"Recipient: {pre_record.get('recipient')}")
    print(f"Task: {pre_record.get('task')}")
    print(f"Candidates: {len(candidates)}; selected: {len(selected)}")

    print("\n=== CANDIDATE FACTS ===")
    for index, fact in enumerate(candidates):
        status = "SELECTED" if fact in selected else "rejected"
        label = labels.get(fact, "not in context labeling")
        print(f"{index:02d} [{status:8}] [{label}] {fact}")

    _print_exposures(
        "PRE-RERANK",
        pre_record,
        labels,
        show_judge_raw=args.show_judge_raw,
    )
    _print_exposures(
        "POST-RERANK",
        post_record,
        labels,
        show_judge_raw=args.show_judge_raw,
    )


if __name__ == "__main__":
    main()
