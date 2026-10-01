from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import quote

from .config import DEFAULT_ARCHIVAL_MEMORY_SEARCH_LIMIT
from .http import HttpClient


def build_agent_system_prompt(
    archival_search_limit: int = DEFAULT_ARCHIVAL_MEMORY_SEARCH_LIMIT,
) -> str:
    return (
        "You are an assistant.\n"
        "\n"
        "Memory rules:\n"
        "1) If the user explicitly asks you to remember something, store it using archival_memory_insert.\n"
        "2) If the user asks about something they previously told you, or if their question depends on prior personal facts/preferences,\n"
        "   you MUST first call archival_memory_search to retrieve relevant info before answering.\n"
        "3) If you are unsure whether a relevant memory exists, do a quick archival_memory_search anyway.\n"
        "\n"
        "TOOL USAGE CONTRACT (STRICT):\n"
        "- When calling archival_memory_search, you MUST NOT use tags (do not provide a tags field at all).\n"
        "- Use ONLY a plain-text query string.\n"
        "- Do NOT pass a limit argument to archival_memory_search; this Letta tool uses top_k instead.\n"
        f"- You MUST include \"top_k\": {archival_search_limit} in every archival_memory_search call.\n"
        "- If a search returns no results, retry ONCE with a simpler query (still without tags, still with top_k).\n"
        "\n"
        "Correct example:\n"
        f"archival_memory_search({{\"query\": \"user investment amount\", \"top_k\": {archival_search_limit}}})\n"
        "Incorrect examples:\n"
        "archival_memory_search({\"query\": \"...\", \"tags\": [\"persona\"]})\n"
        "archival_memory_search({\"tags\": [\"...\"]})\n"
        f"archival_memory_search({{\"query\": \"...\", \"limit\": {archival_search_limit}}})\n"
        "archival_memory_search({\"query\": \"...\"})\n"
        "\n"
        "How to retrieve:\n"
        "- Use archival_memory_search with a short, specific query derived from the user's request.\n"
        "- If results are found, incorporate them into your answer.\n"
        "- If no results are found, say you couldn't find anything in archival memory and proceed normally.\n"
    )


def build_no_tool_agent_system_prompt() -> str:
    return (
        "You are an assistant.\n"
        "\n"
        "Use only the information provided in the current prompt. Do not assume additional user memories exist.\n"
        "When user knowledge memories are provided, include only facts necessary to complete the requested message.\n"
        "Do not include sensitive or unrelated facts just because they appear in the prompt.\n"
    )


@dataclass
class AgentClient:
    base_url: str
    http: HttpClient

    def list_agents(self) -> list[dict[str, Any]]:
        return self.http.get(f"{self.base_url}/agents/")

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self.http.get(f"{self.base_url}/agents/{quote(agent_id)}")

    def find_agent_id_by_name(self, name: str) -> str | None:
        agents = self.list_agents()
        matches = [a for a in agents if a.get("name") == name]
        if not matches:
            return None

        def parse_dt(a: dict[str, Any]) -> datetime:
            s = a.get("created_at")
            if not s:
                return datetime.min
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return datetime.min

        matches.sort(key=parse_dt, reverse=True)
        return matches[0].get("id")

    def create_agent(
        self,
        name: str,
        model: str = "gpt-5.2",
        model_endpoint_type: str = "openai",
        model_endpoint: str = "https://api.openai.com/v1",
        reasoning_effort: str | None = None,
        context_window: int = 128000,
        embedding_endpoint_type: str = "openai",
        embedding_endpoint: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        archival_search_limit: int = DEFAULT_ARCHIVAL_MEMORY_SEARCH_LIMIT,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        llm_config = {
            "model": model,
            "model_endpoint_type": model_endpoint_type,
            "model_endpoint": model_endpoint,
            "context_window": context_window,
        }
        if reasoning_effort is not None:
            llm_config["reasoning_effort"] = reasoning_effort

        payload = {
            "name": name,
            "system": system_prompt or build_agent_system_prompt(archival_search_limit),
            "memory": {"memory": {"human": {"value": "", "limit": 2000, "label": "human"}}},
            "tools": tools if tools is not None else ["archival_memory_insert", "archival_memory_search"],
            "llm_config": llm_config,
            "embedding_config": {
                "embedding_endpoint_type": embedding_endpoint_type,
                "embedding_endpoint": embedding_endpoint,
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
            },
        }
        agent = self.http.post(f"{self.base_url}/agents/", payload)
        return agent["id"]

    def get_or_create_agent_id(self, name: str, **create_kwargs: Any) -> tuple[str, bool]:
        agent_id = self.find_agent_id_by_name(name)
        if agent_id:
            return agent_id, False
        return self.create_agent(name, **create_kwargs), True

    def get_agent_model(self, agent_id: str) -> str | None:
        agent = self.get_agent(agent_id)
        llm_config = agent.get("llm_config")
        if not isinstance(llm_config, dict):
            return None
        model = llm_config.get("model")
        return model if isinstance(model, str) else None

    def update_agent_model(
        self,
        agent_id: str,
        model: str,
        model_endpoint_type: str | None = None,
        model_endpoint: str | None = None,
        reasoning_effort: str | None = None,
        context_window: int | None = None,
    ) -> dict[str, Any]:
        agent = self.get_agent(agent_id)
        llm_config = agent.get("llm_config")
        if not isinstance(llm_config, dict):
            raise RuntimeError(f"Agent {agent_id} has no llm_config to update.")
        llm_config = dict(llm_config)
        llm_config["model"] = model
        if model_endpoint_type is not None:
            llm_config["model_endpoint_type"] = model_endpoint_type
        if model_endpoint is not None:
            llm_config["model_endpoint"] = model_endpoint
        if reasoning_effort is not None:
            llm_config["reasoning_effort"] = reasoning_effort
        if context_window is not None:
            llm_config["context_window"] = context_window
        return self.http.patch(
            f"{self.base_url}/agents/{quote(agent_id)}",
            {"id": agent_id, "llm_config": llm_config},
        )

    def update_agent_embedding_config(
        self,
        agent_id: str,
        *,
        embedding_endpoint_type: str,
        embedding_endpoint: str,
        embedding_model: str,
        embedding_dim: int,
    ) -> dict[str, Any]:
        agent = self.get_agent(agent_id)
        embedding_config = agent.get("embedding_config")
        if not isinstance(embedding_config, dict):
            raise RuntimeError(f"Agent {agent_id} has no embedding_config to update.")
        embedding_config = dict(embedding_config)
        embedding_config["embedding_endpoint_type"] = embedding_endpoint_type
        embedding_config["embedding_endpoint"] = embedding_endpoint
        embedding_config["embedding_model"] = embedding_model
        embedding_config["embedding_dim"] = embedding_dim
        return self.http.patch(
            f"{self.base_url}/agents/{quote(agent_id)}",
            {"id": agent_id, "embedding_config": embedding_config},
        )

    def update_agent_system_prompt(
        self,
        agent_id: str,
        system_prompt: str | None = None,
        archival_search_limit: int = DEFAULT_ARCHIVAL_MEMORY_SEARCH_LIMIT,
    ) -> dict[str, Any]:
        return self.http.patch(
            f"{self.base_url}/agents/{quote(agent_id)}",
            {
                "id": agent_id,
                "system": system_prompt
                or build_agent_system_prompt(archival_search_limit),
            },
        )

    def send_agent_message(self, agent_id: str, text: str) -> dict[str, Any]:
        return self.http.post(f"{self.base_url}/agents/{quote(agent_id)}/messages", {"input": text})

    def reset_messages(self, agent_id: str) -> dict[str, Any]:
        return self.http.patch(
            f"{self.base_url}/agents/{quote(agent_id)}/reset-messages",
            {"add_default_initial_messages": False},
        )

    def get_core_memory(self, agent_id: str) -> dict[str, Any]:
        return self.http.get(f"{self.base_url}/agents/{quote(agent_id)}/core-memory")
