from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .memory import MemoryClient
from .memory_snapshot_export import (
    _full_graph_html,
    _full_graph_svg,
    _graphml,
    _memory_dump_html,
    _read_full_graph,
)
from .persona import PersonaInitResult
from .profile_memory import ProfileMemoryInitResult


def _json_safe(value: Any, *, include_embeddings: bool) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, include_embeddings=include_embeddings)
            for key, item in value.items()
            if include_embeddings or "embedding" not in str(key).lower()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, include_embeddings=include_embeddings) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "passages", "results", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def capture_initialized_memory_snapshot(
    *,
    output_dir: Path,
    memory_mode: int,
    memory_mode_name: str,
    persona_idx: int,
    persona_name: str,
    source_statements: list[str],
    init_result: PersonaInitResult,
    include_embeddings: bool = False,
    mem: MemoryClient | None = None,
    agent_id: str | None = None,
    graph_group_id: str | None = None,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    profile_init_result: ProfileMemoryInitResult | None = None,
) -> dict[str, Any] | None:
    """Persist the backend-observed memory immediately after initialization.

    Attacker-only modes have no initialized user-memory structure and therefore
    return ``None``. The caller deliberately treats capture failures as
    initialization failures so an experiment cannot silently proceed without
    its requested audit artifact.
    """
    if memory_mode in (7, 8):
        return None

    root = output_dir / "initialized_memory"
    root.mkdir(parents=True, exist_ok=False)
    os.chmod(root, 0o700)
    source = [str(item) for item in source_statements]
    backend: dict[str, Any]
    files: dict[str, str] = {}
    observed_count: int
    completeness_basis: str

    if 4 <= memory_mode <= 6:
        required = {
            "graph_group_id": graph_group_id,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Graph snapshot is missing configuration: {', '.join(missing)}")
        graph = _read_full_graph(
            group_id=str(graph_group_id),
            neo4j_uri=str(neo4j_uri),
            neo4j_user=str(neo4j_user),
            neo4j_password=str(neo4j_password),
            include_embeddings=include_embeddings,
        )
        episode_count = sum("Episodic" in node.get("labels", []) for node in graph["nodes"])
        observed_count = episode_count
        completeness_basis = "observed Graphiti Episodic nodes equal successfully inserted episodes"
        backend = graph
        _write(root / "full_graph.json", graph)
        (root / "full_graph.graphml").write_text(_graphml(graph), encoding="utf-8")
        (root / "full_graph.svg").write_text(_full_graph_svg(graph, persona_name), encoding="utf-8")
        (root / "full_graph.html").write_text(_full_graph_html(graph, persona_name), encoding="utf-8")
        for name in ("full_graph.graphml", "full_graph.svg", "full_graph.html"):
            os.chmod(root / name, 0o600)
        files.update(
            backend_json="full_graph.json",
            interactive_html="full_graph.html",
            static_svg="full_graph.svg",
            graphml="full_graph.graphml",
        )
    elif memory_mode >= 9:
        if profile_init_result is None:
            raise ValueError("Profile snapshot requires the completed Memobase initialization result")
        profile = _json_safe(
            profile_init_result.profile, include_embeddings=include_embeddings
        )
        observed_count = int(profile_init_result.entries_settled or 0)
        completeness_basis = "Memobase settle poll converged and saved profile entry count matches settled count"
        backend = {
            "profile": profile,
            "inserted_statements": source,
            "initialization": {
                key: value
                for key, value in profile_init_result.to_json().items()
                if key != "profile"
            },
        }
        _write(root / "full_profile.json", backend)
        (root / "full_profile.html").write_text(
            _memory_dump_html(
                persona_name=persona_name,
                persona_idx=persona_idx,
                mode_name=memory_mode_name,
                backend=backend,
                exact_data_href="full_profile.json",
            ),
            encoding="utf-8",
        )
        os.chmod(root / "full_profile.html", 0o600)
        files.update(backend_json="full_profile.json", interactive_html="full_profile.html")
    else:
        if mem is None or not agent_id:
            raise ValueError("List snapshot requires a MemoryClient and experiment agent ID")
        raw = mem.list_archival_passages(
            agent_id, limit=max(1, init_result.inserted), ascending=True
        )
        entries = _json_safe(_items(raw), include_embeddings=include_embeddings)
        observed_count = len(entries)
        completeness_basis = "observed Letta archival passage count equals successfully inserted statements"
        atomic_entries = []
        for index, entry in enumerate(entries):
            if isinstance(entry, dict):
                text = entry.get("text") or entry.get("content") or entry.get("passage") or ""
                atomic_entries.append(
                    {"index": index, "memory_statement": str(text), "backend_record": entry}
                )
            else:
                atomic_entries.append(
                    {"index": index, "memory_statement": str(entry), "backend_record": entry}
                )
        backend = {
            "atomic_entries": atomic_entries,
            "raw_backend_response": _json_safe(raw, include_embeddings=include_embeddings),
            "agent_id": agent_id,
        }
        _write(root / "full_list_memory.json", backend)
        (root / "full_list_memory.html").write_text(
            _memory_dump_html(
                persona_name=persona_name,
                persona_idx=persona_idx,
                mode_name=memory_mode_name,
                backend=backend,
                exact_data_href="full_list_memory.json",
            ),
            encoding="utf-8",
        )
        os.chmod(root / "full_list_memory.html", 0o600)
        files.update(backend_json="full_list_memory.json", interactive_html="full_list_memory.html")

    if profile_init_result is not None:
        complete = (
            init_result.failed == 0
            and profile_init_result.settle_converged
            and observed_count == int(profile_init_result.entries_settled or 0)
            and (init_result.inserted == 0 or observed_count > 0)
        )
    else:
        complete = init_result.failed == 0 and observed_count == init_result.inserted
    manifest = {
        "schema_version": 1,
        "capture_stage": "after_memory_initialization_before_scenario_retrieval",
        "memory_mode": memory_mode,
        "memory_mode_name": memory_mode_name,
        "persona_idx": persona_idx,
        "persona_name": persona_name,
        "source_statement_count": len(source),
        "initialization_found": init_result.found,
        "initialization_inserted": init_result.inserted,
        "initialization_failed": init_result.failed,
        "backend_observed_primary_count": observed_count,
        "completeness_basis": completeness_basis,
        "complete": complete,
        "include_embeddings": include_embeddings,
        "embedding_policy": "included_when_exposed_by_backend" if include_embeddings else "recursively_omitted",
        "files": files,
    }
    _write(root / "manifest.json", manifest)
    if not complete:
        raise RuntimeError(
            "Initialized-memory snapshot is incomplete: "
            f"inserted={init_result.inserted}, observed={observed_count}, "
            f"failed={init_result.failed}, basis={completeness_basis}"
        )
    return {**manifest, "directory": str(root)}
