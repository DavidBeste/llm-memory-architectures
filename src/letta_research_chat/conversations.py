from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from .http import HttpClient


def _parse_iso_dt(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


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
class ConversationClient:
    base_url: str
    http: HttpClient

    # Your server expects agent_id as query param:
    # POST /v1/conversations?agent_id=...
    def create_conversation(self, agent_id: str) -> dict[str, Any]:
        url = f"{self.base_url}/conversations?agent_id={quote(agent_id)}"
        return self.http.post(url, payload={})

    def list_conversations(self, agent_id: str, limit: int = 50) -> dict[str, Any]:
        url = f"{self.base_url}/conversations?agent_id={quote(agent_id)}&limit={limit}"
        return self.http.get(url)

    def conversation_exists(self, agent_id: str, conversation_id: str, limit: int = 100) -> bool:
        convos = self.list_conversations(agent_id, limit=limit)
        items = _iter_items(convos)
        for c in items:
            if not isinstance(c, dict):
                continue
            cid = c.get("id") or c.get("conversation_id")
            if cid == conversation_id:
                return True
        return False

    def send_conversation_message(self, conversation_id: str, text: str) -> dict[str, Any]:
        return self.http.post(
            f"{self.base_url}/conversations/{quote(conversation_id)}/messages",
            {"input": text, "streaming": False},
        )

    def get_conversation_messages(self, conversation_id: str, limit: int = 100) -> dict[str, Any]:
        return self.http.get(f"{self.base_url}/conversations/{quote(conversation_id)}/messages/?limit={limit}")

    def get_latest_conversation_id(self, agent_id: str, limit: int = 100) -> str | None:
        convos = self.list_conversations(agent_id, limit=limit)
        items = _iter_items(convos)
        items = [c for c in items if isinstance(c, dict)]
        if not items:
            return None

        def score(c: dict[str, Any]) -> datetime:
            dt = _parse_iso_dt(c.get("updated_at") or c.get("last_updated_at") or c.get("created_at"))
            return dt or datetime.min.replace(tzinfo=timezone.utc)

        items.sort(key=score, reverse=True)
        latest = items[0]
        return latest.get("id") or latest.get("conversation_id")
