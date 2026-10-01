from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from .config import DEFAULT_ARCHIVAL_MEMORY_SEARCH_LIMIT
from .http import HttpClient
from .efficiency import extract_token_usage, sum_token_usage


def _iter_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        for key in ("data", "passages", "results", "conversations", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []
    if isinstance(payload, list):
        return payload
    return []


@dataclass
class MemoryClient:
    base_url: str
    http: HttpClient
    _usage_events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def _record_usage(self, operation: str, response: Any) -> Any:
        usage = extract_token_usage(response)
        self._usage_events.append(
            {"operation": operation, "usage": usage, "usage_available": usage is not None}
        )
        return response

    def usage_snapshot(self) -> dict[str, Any]:
        usage = sum_token_usage(event.get("usage") for event in self._usage_events)
        return {
            **(usage or {}),
            "calls": len(self._usage_events),
            "usage_calls": sum(1 for event in self._usage_events if event.get("usage") is not None),
        }

    def usage_events(self) -> list[dict[str, Any]]:
        return list(self._usage_events)

    def list_archival_passages(self, agent_id: str, limit: int = 50, ascending: bool = False) -> dict[str, Any]:
        url = (
            f"{self.base_url}/agents/{quote(agent_id)}/archival-memory"
            f"?limit={limit}&ascending={'true' if ascending else 'false'}"
        )
        return self._record_usage("archival_list", self.http.get(url))

    def search_archival_memory(
        self,
        agent_id: str,
        query: str,
        limit: int = DEFAULT_ARCHIVAL_MEMORY_SEARCH_LIMIT,
    ) -> dict[str, Any]:
        # Letta's archival search endpoint calls this parameter ``top_k``.
        # Sending ``limit`` is silently ignored by current servers, causing
        # them to return the server default (commonly five results).
        url = (
            f"{self.base_url}/agents/{quote(agent_id)}/archival-memory/search"
            f"?query={quote(query)}&top_k={limit}"
        )
        return self._record_usage("archival_search", self.http.get(url))

    def insert_archival_memory(self, agent_id: str, text: str) -> dict[str, Any]:
        payload = {"text": text}
        return self._record_usage(
            "archival_insert",
            self.http.post(f"{self.base_url}/agents/{quote(agent_id)}/archival-memory", payload),
        )

    @staticmethod
    def format_archival(payload: Any) -> str:
        items = _iter_items(payload)
        if not items:
            return "[archival] (no entries)\n"

        out = ["[archival]"]
        for i, p in enumerate(items, 1):
            if not isinstance(p, dict):
                continue
            text = p.get("text") or p.get("content") or p.get("passage") or ""
            created = p.get("created_at") or p.get("date") or ""
            pid = p.get("id") or p.get("memory_id") or ""
            text = text.strip() if isinstance(text, str) else str(text)
            created = created.strip() if isinstance(created, str) else str(created)

            head = f"{i}."
            if created:
                head += f" [{created}]"
            if pid:
                head += f" ({pid})"
            out.append(head)
            out.append(text)
            out.append("")
        return "\n".join(out) + "\n"
