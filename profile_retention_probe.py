#!/usr/bin/env python
"""Measure how much of a CIMemories persona survives Memobase ingestion.

Generation, judging and Letta are all skipped: this only exercises the
ingestion -> extraction -> retrieval path, so a cell costs a handful of
extraction calls rather than a pipeline run.

Usage:
    python profile_retention_probe.py <dataset> <persona_idx> <memory_mode> <label>

Results are appended to research_outputs/profile_retention_probe.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from letta_research_chat.cli import (  # noqa: E402
    build_profile_memory_statements,
    open_profile_memory,
    parse_memory_mode,
    profile_memory_ingestion_mode,
)
from letta_research_chat.config import LettaConfig  # noqa: E402
from letta_research_chat.persona import load_persona_entry, persona_label  # noqa: E402

RESULTS_PATH = Path("research_outputs") / "profile_retention_probe.json"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def attribute_values(entry: dict) -> list[tuple[str, str, str]]:
    """Return (key, domain, normalized value) for each source attribute."""
    out = []
    for key, record in (entry.get("information_attributes") or {}).items():
        if not isinstance(record, dict):
            continue
        value = record.get("value")
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        value = normalize(value)
        # Values shorter than 3 chars match by accident; score those by statement instead.
        if len(value) < 3:
            value = normalize(record.get("memory_statement") or "")
        domain = str(record.get("information_domain") or "general")
        out.append((str(key), domain, value))
    return out


def profile_text(payload: dict) -> str:
    parts = []
    for topic, subs in (payload or {}).items():
        if not isinstance(subs, dict):
            continue
        for sub, item in subs.items():
            content = item.get("content") if isinstance(item, dict) else item
            parts.append(f"{topic} {sub} {content}")
    return normalize(" ".join(parts))


def wait_for_drain(client, user_id: str, timeout_s: int = 1800) -> int:
    """Block until the extracted profile stops growing.

    insert(sync=False) plus a single trailing flush does not cover blobs the
    server already picked up, so the profile keeps growing after the CLI would
    have read it. Waiting here measures extraction capability rather than the
    race.

    An empty buffer is NOT sufficient: Memobase's background processor keeps
    running after the buffer drains (it logs "Completed processing. Iterations:
    N"), and polling only the buffer under-counts by 2-15x. Poll the profile
    itself and require the entry count to hold steady across three checks.
    """

    def entry_count() -> int:
        payload = client.profile(user_id) or {}
        return sum(len(v) for v in payload.values() if isinstance(v, dict))

    deadline = time.time() + timeout_s
    waited = 0
    last = -1
    stable = 0
    while time.time() < deadline:
        current = entry_count()
        stable = stable + 1 if current == last else 0
        last = current
        if stable >= 3 and current > 0:
            return waited
        time.sleep(20)
        waited += 20
    return waited


def main() -> None:
    if len(sys.argv) != 5:
        print(__doc__)
        raise SystemExit(2)
    dataset, persona_idx, mode_arg, label = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    memory_mode = parse_memory_mode(mode_arg)

    cfg = LettaConfig()
    entry = load_persona_entry(dataset, persona_idx)
    name = persona_label(entry)
    attributes = attribute_values(entry)
    statements = build_profile_memory_statements(entry, name, memory_mode)

    print(f"[probe] {label}: persona={name} mode={memory_mode} "
          f"({profile_memory_ingestion_mode(memory_mode)})")
    print(f"[probe] source attributes={len(attributes)} ingest blobs={len(statements)} "
          f"chars={sum(len(s) for s in statements)}")

    client = open_profile_memory(cfg)
    started = time.time()
    user_id = client.create_user({"source": "profile_retention_probe", "label": label})
    init = client.insert_memory_statements(user_id, statements, sync_flush=True)
    at_read_time = client.profile(user_id) or {}
    waited = wait_for_drain(client, user_id)
    settled = client.profile(user_id) or {}
    elapsed = round(time.time() - started, 1)

    text = profile_text(settled)
    hits = [(k, d) for k, d, v in attributes if v and v in text]
    by_domain: dict[str, list[int]] = {}
    for key, domain, value in attributes:
        found = 1 if (value and value in text) else 0
        by_domain.setdefault(domain, []).append(found)

    def count_entries(payload: dict) -> int:
        return sum(len(v) for v in (payload or {}).values() if isinstance(v, dict))

    result = {
        "label": label,
        "memory_mode": memory_mode,
        "ingestion_mode": profile_memory_ingestion_mode(memory_mode),
        "persona": name,
        "user_id": user_id,
        "source_attributes": len(attributes),
        "ingest_blobs": init.inserted_count,
        "entries_at_read_time": count_entries(at_read_time),
        "entries_settled": count_entries(settled),
        "extra_wait_seconds": waited,
        "elapsed_seconds": elapsed,
        "value_recall": round(100.0 * len(hits) / max(1, len(attributes)), 1),
        "domain_recall": {
            d: round(100.0 * sum(v) / len(v), 1) for d, v in sorted(by_domain.items())
        },
        "topics": {t: len(s) for t, s in sorted(settled.items()) if isinstance(s, dict)},
    }

    print(f"[probe] entries at read time : {result['entries_at_read_time']}")
    print(f"[probe] entries settled      : {result['entries_settled']} "
          f"(after {waited}s extra wait)")
    print(f"[probe] value recall         : {result['value_recall']}% "
          f"({len(hits)}/{len(attributes)})")
    print(f"[probe] topics               : {result['topics']}")
    print("[probe] per-domain recall:")
    for d, pct in result["domain_recall"].items():
        print(f"           {d:16s} {pct:5.1f}%")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.is_file() else []
    existing.append(result)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2))
    print(f"[probe] appended -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
