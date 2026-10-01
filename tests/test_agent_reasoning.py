from __future__ import annotations

import unittest
from typing import Any

from letta_research_chat.agent import AgentClient
from letta_research_chat.cli import agent_llm_config_kwargs
from letta_research_chat.config import LettaConfig


class FakeHttp:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.patches: list[tuple[str, dict[str, Any]]] = []
        self.agent: dict[str, Any] = {
            "id": "agent-1",
            "llm_config": {
                "model": "gpt-5",
                "model_endpoint_type": "openai",
                "model_endpoint": "https://api.openai.com/v1",
                "context_window": 128000,
                "reasoning_effort": "minimal",
            },
        }

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((path, payload))
        return {"id": "agent-1"}

    def get(self, path: str) -> dict[str, Any]:
        return self.agent

    def patch(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.patches.append((path, payload))
        return payload


class AgentReasoningEffortTests(unittest.TestCase):
    def test_create_agent_includes_explicit_reasoning_effort(self) -> None:
        http = FakeHttp()
        agent = AgentClient("http://localhost:8283/v1", http)  # type: ignore[arg-type]

        agent.create_agent("test", model="gpt-5.6-sol", reasoning_effort="medium")

        self.assertEqual(http.posts[0][1]["llm_config"]["reasoning_effort"], "medium")

    def test_create_agent_omits_reasoning_effort_when_unset(self) -> None:
        http = FakeHttp()
        agent = AgentClient("http://localhost:8283/v1", http)  # type: ignore[arg-type]

        agent.create_agent("test")

        self.assertNotIn("reasoning_effort", http.posts[0][1]["llm_config"])

    def test_update_agent_replaces_incompatible_reasoning_effort(self) -> None:
        http = FakeHttp()
        agent = AgentClient("http://localhost:8283/v1", http)  # type: ignore[arg-type]

        agent.update_agent_model("agent-1", "gpt-5.6-sol", reasoning_effort="medium")

        self.assertEqual(http.patches[0][1]["llm_config"]["reasoning_effort"], "medium")

    def test_config_kwargs_propagate_reasoning_effort(self) -> None:
        cfg = LettaConfig(agent_reasoning_effort="medium")

        self.assertEqual(agent_llm_config_kwargs(cfg)["reasoning_effort"], "medium")


if __name__ == "__main__":
    unittest.main()
