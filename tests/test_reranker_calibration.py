import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from letta_research_chat.cli import calibrate_reranker
from letta_research_chat.persona import PersonaInitResult


class FakeAgent:
    def __init__(self):
        self.calls = []

    def create_agent(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return "agent-calibration"


class FakeMemory:
    def __init__(self):
        self.searches = []

    def search_archival_memory(self, agent_id, query, limit):
        self.searches.append((agent_id, query, limit))
        return {"query": query}


class CalibrationHttp:
    def __init__(self):
        self.posts = []

    def post(self, url, payload, headers, timeout):
        self.posts.append((url, payload, headers, timeout))
        return {
            "id": f"call-{len(self.posts)}",
            "model": payload["model"],
            "choices": [
                {
                    "message": {"content": '{"selected_indices":[0]}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "total_tokens": 23,
            },
        }


class RerankerCalibrationTests(unittest.TestCase):
    def test_calibration_initializes_and_retrieves_once_per_context(self) -> None:
        entry = {
            "bio": {"name": "Test Person"},
            "contexts": [
                {"recipient": "Bank", "task": "Apply"},
                {"recipient": "Advisor", "task": "Plan"},
                {"recipient": "Unused", "task": "Not in pilot"},
            ],
        }
        cfg = SimpleNamespace(
            rerank_prompt_mode="indices_only",
            rerank_candidate_source="search",
            rerank_candidate_limit=50,
            rerank_output_limit=20,
            rerank_model="test-reranker",
            agent_name="test-agent",
            archival_search_limit=10,
        )
        agent = FakeAgent()
        memory = FakeMemory()
        http = CalibrationHttp()

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            env = {
                "LETTA_RERANK_BASE_URL": "https://example.test/v1",
                "LETTA_RERANK_API_STYLE": "chat_completions",
                "LETTA_RERANK_API_KEY": "test-key",
                "LETTA_RERANK_CHECKPOINT_DIR": str(Path(tmp) / "cache"),
            }
            with (
                patch.dict(os.environ, env, clear=True),
                patch(
                    "letta_research_chat.cli.resolve_dataset_filename",
                    return_value="dataset.json",
                ),
                patch(
                    "letta_research_chat.cli.load_persona_entry",
                    return_value=entry,
                ),
                patch(
                    "letta_research_chat.cli.extract_persona_memory_statements",
                    return_value=["alpha", "beta"],
                ),
                patch(
                    "letta_research_chat.cli.extract_archival_memory_texts",
                    return_value=["alpha", "beta"],
                ),
                patch(
                    "letta_research_chat.cli.init_archival_from_entry",
                    return_value=PersonaInitResult(0, "Test Person", 2, 2, 0),
                ),
                patch(
                    "letta_research_chat.cli.agent_llm_config_kwargs",
                    return_value={},
                ),
            ):
                output = calibrate_reranker(
                    cfg=cfg,
                    http=http,
                    agent=agent,
                    mem=memory,
                    dataset_name="dataset.json",
                    persona_idx=0,
                    agent_model="generation-model",
                    pilot_contexts=2,
                    reasoning_efforts=("low", "high"),
                    max_output_tokens=2048,
                    output_root=output_root,
                )

            report = json.loads(
                (output / "reranker_calibration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(agent.calls), 1)
            self.assertEqual(len(memory.searches), 2)
            self.assertEqual(len(http.posts), 4)
            self.assertEqual(
                [post[1]["reasoning_effort"] for post in http.posts],
                ["low", "high", "low", "high"],
            )
            self.assertEqual(report["context_indices"], [0, 1])
            self.assertEqual([row["valid"] for row in report["summaries"]], [2, 2])
            self.assertEqual(len(list((output / "calls").glob("*.json"))), 4)
            self.assertTrue((output / "reranker_calibration.md").is_file())


if __name__ == "__main__":
    unittest.main()
