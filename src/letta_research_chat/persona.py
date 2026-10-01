from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


def _walk_collect_memory_statements(obj: Any, out: list[str]) -> None:
    """
    Recursively collect any strings found at:
      - dict["memory_statement"]
      - dict["memory_statements"] (list of strings)
    """
    if obj is None:
        return

    if isinstance(obj, dict):
        ms = obj.get("memory_statement")
        if isinstance(ms, str):
            s = ms.strip()
            if s:
                out.append(s)

        msl = obj.get("memory_statements")
        if isinstance(msl, list):
            for x in msl:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())

        for v in obj.values():
            _walk_collect_memory_statements(v, out)

    elif isinstance(obj, list):
        for it in obj:
            _walk_collect_memory_statements(it, out)


def load_persona_entry(filename: str, index: int) -> dict[str, Any]:
    data = load_persona_dataset(filename)

    if index < 0 or index >= len(data):
        raise IndexError(f"Index {index} out of range (0..{len(data)-1}).")

    entry = data[index]
    if not isinstance(entry, dict):
        raise ValueError(f"Persona entry at index {index} is not an object/dict.")
    return entry


def load_persona_dataset(filename: str) -> list[dict[str, Any]]:
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Persona JSON root must be a list/array of personas.")

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Persona entry at index {i} is not an object/dict.")
    return data


def load_persona_context(filename: str, index: int, context_index: int) -> dict[str, str]:
    entry = load_persona_entry(filename, index)
    contexts = entry.get("contexts")
    if not isinstance(contexts, list):
        raise ValueError(f"Persona entry at index {index} does not contain a contexts list.")
    if context_index < 0 or context_index >= len(contexts):
        raise IndexError(f"Context index {context_index} out of range (0..{len(contexts)-1}).")

    context = contexts[context_index]
    if not isinstance(context, dict):
        raise ValueError(f"Context entry at index {context_index} is not an object/dict.")

    recipient = context.get("recipient")
    task = context.get("task")
    if not isinstance(recipient, str) or not recipient.strip():
        raise ValueError(f"Context entry at index {context_index} has no valid recipient.")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"Context entry at index {context_index} has no valid task.")

    return {"recipient": recipient.strip(), "task": task.strip()}


def persona_label(entry: dict[str, Any]) -> str:
    bio = entry.get("bio")
    if isinstance(bio, dict):
        name = bio.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    ia = entry.get("information_attributes")
    if isinstance(ia, dict):
        # Match on the attribute key ("name", "full_name", ...). The previous
        # check tested `"name" in (v.get("event") or "name")`, which is
        # unconditionally true whenever `event` is absent and so returned the
        # first attribute's value - an address or an amount - as the persona name.
        for key, v in ia.items():
            if not isinstance(v, dict):
                continue
            val = v.get("value")
            if not (isinstance(val, str) and val.strip()):
                continue
            if "name" in str(key).lower():
                return val.strip()
    return "<unknown persona>"


def extract_persona_memory_statements(entry: dict[str, Any]) -> list[str]:
    out: list[str] = []
    _walk_collect_memory_statements(entry, out)

    # de-dupe while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


@dataclass
class PersonaInitResult:
    persona_index: int
    label: str
    found: int
    inserted: int
    failed: int
