from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from letta_research_chat.graph_memory import (
    _ProviderUsageRecorder,
    _build_graphiti_provider_kwargs,
    _is_official_openai_base_url,
    _reasoning_compatible_graphiti_client,
)


class _FakeLLMConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeOpenAIClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeOpenAIGenericClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeEmptyResponseError(Exception):
    pass


class _FakeRateLimitError(Exception):
    pass


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class GraphitiProviderTests(unittest.TestCase):
    def test_provider_usage_recorder_keeps_per_call_and_aggregate_usage(self) -> None:
        recorder = _ProviderUsageRecorder()
        recorder.record(
            "embedder",
            "embeddings",
            {"usage": {"prompt_tokens": 12, "total_tokens": 12}},
            {"model": "text-embedding-3-small"},
        )
        recorder.record(
            "embedder",
            "embeddings",
            {"usage": {"prompt_tokens": 8, "total_tokens": 8}},
            {"model": "text-embedding-3-small"},
        )
        snapshot = recorder.snapshot()
        self.assertEqual(snapshot["embedder"]["input_tokens"], 20)
        self.assertEqual(snapshot["embedder"]["total_tokens"], 20)
        self.assertEqual(snapshot["embedder"]["calls"], 2)
        self.assertEqual(len(recorder.event_snapshot()), 2)

    def _build(
        self,
        base_url: str,
        *,
        reasoning_effort: str | None = "medium",
    ) -> dict[str, object]:
        modules = {
            "graphiti_core": _module("graphiti_core"),
            "graphiti_core.llm_client": _module("graphiti_core.llm_client"),
            "graphiti_core.llm_client.config": _module(
                "graphiti_core.llm_client.config",
                LLMConfig=_FakeLLMConfig,
                DEFAULT_MAX_TOKENS=4096,
            ),
            "graphiti_core.llm_client.errors": _module(
                "graphiti_core.llm_client.errors",
                EmptyResponseError=_FakeEmptyResponseError,
                RateLimitError=_FakeRateLimitError,
            ),
            "graphiti_core.llm_client.openai_client": _module(
                "graphiti_core.llm_client.openai_client", OpenAIClient=_FakeOpenAIClient
            ),
            "graphiti_core.llm_client.openai_generic_client": _module(
                "graphiti_core.llm_client.openai_generic_client",
                OpenAIGenericClient=_FakeOpenAIGenericClient,
            ),
        }
        with patch.dict(sys.modules, modules):
            return _build_graphiti_provider_kwargs(
                llm_base_url=base_url,
                llm_api_key="test-key",
                llm_model="gpt-5.6-sol",
                llm_reasoning_effort=reasoning_effort,
                llm_structured_output_mode="json_object",
                llm_max_tokens=4096,
                embedding_base_url=None,
                embedding_api_key=None,
                embedding_model=None,
                embedding_dim=1536,
                reranker_base_url=None,
                reranker_api_key=None,
                reranker_model=None,
            )

    def test_official_openai_url_uses_native_responses_client(self) -> None:
        result = self._build("https://api.openai.com/v1/")

        client = result["llm_client"]
        self.assertIsInstance(client, _FakeOpenAIClient)
        self.assertEqual(client.kwargs["reasoning"], "medium")
        self.assertEqual(client.kwargs["max_tokens"], 4096)

    def test_local_url_keeps_generic_compatible_client(self) -> None:
        result = self._build("http://127.0.0.1:8000/v1")

        client = result["llm_client"]
        self.assertIsInstance(client, _FakeOpenAIGenericClient)
        self.assertEqual(client.kwargs["structured_output_mode"], "json_object")
        self.assertEqual(client.letta_reasoning_effort, "medium")

    def test_local_url_without_reasoning_keeps_exact_legacy_client(self) -> None:
        result = self._build("http://127.0.0.1:8000/v1", reasoning_effort=None)

        client = result["llm_client"]
        self.assertIs(type(client), _FakeOpenAIGenericClient)
        self.assertFalse(hasattr(client, "letta_reasoning_effort"))

    def test_gpt_5_6_defaults_to_supported_reasoning_effort(self) -> None:
        result = self._build("https://api.openai.com/v1", reasoning_effort=None)

        client = result["llm_client"]
        self.assertEqual(client.kwargs["reasoning"], "medium")

    def test_openai_url_detection_does_not_accept_lookalike_host(self) -> None:
        self.assertTrue(_is_official_openai_base_url("https://api.openai.com/v1"))
        self.assertFalse(_is_official_openai_base_url("https://api.openai.com.example/v1"))

    def test_reasoning_adapter_forwards_effort_and_saves_empty_response(self) -> None:
        calls = []

        class FakeCompletions:
            async def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    id="response-test",
                    choices=[
                        SimpleNamespace(
                            finish_reason="length",
                            message=SimpleNamespace(
                                content="",
                                reasoning_content="reasoning only",
                            ),
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=32,
                        total_tokens=42,
                        completion_tokens_details=SimpleNamespace(reasoning_tokens=32),
                    ),
                    model_dump=lambda: {
                        "id": "response-test",
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {
                                    "content": "",
                                    "reasoning_content": "reasoning only",
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 32,
                            "total_tokens": 42,
                            "completion_tokens_details": {"reasoning_tokens": 32},
                        },
                    },
                )

        class FakeBase:
            def __init__(self, **kwargs):
                self.client = SimpleNamespace(
                    chat=SimpleNamespace(completions=FakeCompletions())
                )
                self.model = "test-model"
                self.temperature = 0

            def _clean_input(self, value):
                return value

            def _build_response_format(self, response_model):
                return {"type": "json_schema"}

            def _strip_code_fences(self, value):
                return value.strip()

        modules = {
            "graphiti_core": _module("graphiti_core"),
            "graphiti_core.llm_client": _module("graphiti_core.llm_client"),
            "graphiti_core.llm_client.config": _module(
                "graphiti_core.llm_client.config", DEFAULT_MAX_TOKENS=4096
            ),
            "graphiti_core.llm_client.errors": _module(
                "graphiti_core.llm_client.errors",
                EmptyResponseError=_FakeEmptyResponseError,
                RateLimitError=_FakeRateLimitError,
            ),
        }
        messages = [SimpleNamespace(role="user", content="extract")]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), patch.dict(
            os.environ, {"GRAPHITI_LLM_ERROR_DIR": tmp}
        ):
            client = _reasoning_compatible_graphiti_client(
                FakeBase,
                reasoning_effort="high",
            )
            with self.assertRaisesRegex(_FakeEmptyResponseError, "finish_reason=length"):
                asyncio.run(client._generate_response(messages, max_tokens=2048))
            saved = list(Path(tmp).glob("graphiti-response_*.json"))
            self.assertEqual(len(saved), 1)
            payload = json.loads(saved[0].read_text(encoding="utf-8"))

        self.assertEqual(calls[0]["reasoning_effort"], "high")
        self.assertEqual(calls[0]["max_tokens"], 2048)
        self.assertEqual(payload["metadata"]["finish_reason"], "length")
        self.assertTrue(payload["metadata"]["reasoning_content_present"])
        self.assertNotIn("api_key", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
