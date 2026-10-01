import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from letta_research_chat.judge import (
    JudgeClient,
    JudgeResponseJSONError,
    _extract_output_text,
)


class RecordingHttp:
    def __init__(self, response):
        self.response = response
        self.posts = []

    def post(self, url, payload, headers, timeout):
        self.posts.append((url, payload, headers, timeout))
        return self.response

    def post_sse_json(self, url, payload, headers, timeout):
        self.posts.append((url, payload, headers, timeout))
        return self.response


class JudgeResponseDiagnosticsTests(unittest.TestCase):
    def test_memory_stage_judge_uses_dedicated_route_and_settings(self) -> None:
        env = {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_STYLE": "responses",
            "OPENAI_API_KEY": "openai-test-key",
            "LETTA_JUDGE_MODEL": "gpt-5.2",
            "TOGETHER_API_KEY": "together-test-key",
            "LETTA_MEMORY_STAGE_JUDGE_BASE_URL": "https://api.together.xyz/v1",
            "LETTA_MEMORY_STAGE_JUDGE_API_STYLE": "chat_completions",
            "LETTA_MEMORY_STAGE_JUDGE_MODEL": "Qwen/Qwen3.8-Flash",
            "LETTA_MEMORY_STAGE_JUDGE_REASONING_EFFORT": "high",
            "LETTA_MEMORY_STAGE_JUDGE_MAX_TOKENS": "4096",
            "LETTA_MEMORY_STAGE_JUDGE_TIMEOUT": "300",
            "LETTA_MEMORY_STAGE_JUDGE_STREAM": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            memory_judge = JudgeClient.from_memory_stage_env(RecordingHttp({}))
            response_judge = JudgeClient.from_env(RecordingHttp({}))

        self.assertEqual(memory_judge.base_url, "https://api.together.xyz/v1")
        self.assertEqual(memory_judge.api_style, "chat_completions")
        self.assertEqual(memory_judge.model, "Qwen/Qwen3.8-Flash")
        self.assertEqual(memory_judge.api_key, "together-test-key")
        self.assertEqual(memory_judge.credential_source, "TOGETHER_API_KEY")
        self.assertEqual(memory_judge.reasoning_effort, "high")
        self.assertEqual(memory_judge.max_output_tokens, 4096)
        self.assertEqual(memory_judge.timeout, 300)
        self.assertTrue(memory_judge.stream_chat_completions)
        self.assertFalse(response_judge.stream_chat_completions)
        self.assertEqual(response_judge.base_url, "https://api.openai.com/v1")
        self.assertEqual(response_judge.model, "gpt-5.2")

    def test_memory_stage_judge_falls_back_to_generic_settings(self) -> None:
        env = {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_STYLE": "responses",
            "OPENAI_API_KEY": "shared-test-key",
            "LETTA_JUDGE_MODEL": "gpt-5.2",
            "LETTA_JUDGE_REASONING_EFFORT": "none",
            "LETTA_JUDGE_MAX_TOKENS": "3072",
            "LETTA_JUDGE_TIMEOUT": "240",
        }
        with patch.dict(os.environ, env, clear=True):
            judge = JudgeClient.from_memory_stage_env(RecordingHttp({}))

        self.assertEqual(judge.base_url, "https://api.openai.com/v1")
        self.assertEqual(judge.api_style, "responses")
        self.assertEqual(judge.model, "gpt-5.2")
        self.assertEqual(judge.reasoning_effort, "none")
        self.assertEqual(judge.max_output_tokens, 3072)
        self.assertEqual(judge.timeout, 240)
        self.assertEqual(judge.configuration_source, "generic_judge_environment_fallback")
        self.assertFalse(judge.stream_chat_completions)

    def test_reranker_can_use_together_while_judge_stays_on_openai(self) -> None:
        env = {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_STYLE": "responses",
            "OPENAI_API_KEY": "openai-test-key",
            "LETTA_JUDGE_MODEL": "gpt-5.2",
            "LETTA_RERANK_BASE_URL": "https://api.together.xyz/v1",
            "LETTA_RERANK_API_STYLE": "chat_completions",
            "LETTA_RERANK_API_KEY": "together-test-key",
            "LETTA_RERANK_MODEL": "Prism-ML/Ternary-Bonsai-27B",
            "LETTA_RERANK_MAX_TOKENS": "777",
            "LETTA_RERANK_TIMEOUT": "91",
        }
        with patch.dict(os.environ, env, clear=True):
            judge = JudgeClient.from_env(RecordingHttp({}))
            reranker = JudgeClient.from_rerank_env(RecordingHttp({}))

        self.assertEqual(judge.base_url, "https://api.openai.com/v1")
        self.assertEqual(judge.api_style, "responses")
        self.assertEqual(judge.model, "gpt-5.2")
        self.assertEqual(judge.api_key, "openai-test-key")
        self.assertEqual(reranker.base_url, "https://api.together.xyz/v1")
        self.assertEqual(reranker.api_style, "chat_completions")
        self.assertEqual(reranker.model, "Prism-ML/Ternary-Bonsai-27B")
        self.assertEqual(reranker.api_key, "together-test-key")
        self.assertEqual(reranker.max_output_tokens, 777)
        self.assertEqual(reranker.timeout, 91)

    def test_reranker_keeps_legacy_shared_routing_without_override(self) -> None:
        env = {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_STYLE": "responses",
            "OPENAI_API_KEY": "shared-test-key",
        }
        with patch.dict(os.environ, env, clear=True):
            reranker = JudgeClient.from_rerank_env(
                RecordingHttp({}), model="gpt-5.6-sol"
            )

        self.assertEqual(reranker.base_url, "https://api.openai.com/v1")
        self.assertEqual(reranker.api_style, "responses")
        self.assertEqual(reranker.model, "gpt-5.6-sol")
        self.assertEqual(reranker.api_key, "shared-test-key")

    def test_malformed_json_saves_complete_provider_response(self) -> None:
        response = {
            "id": "resp_extra_json",
            "output_text": '{"ok": true}\n{"ok": false}',
            "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
        }
        client = JudgeClient(
            http=RecordingHttp(response),
            api_key="test-key",
            model="gpt-5-nano",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LETTA_JUDGE_ERROR_DIR": tmp}):
                with self.assertRaises(JudgeResponseJSONError) as raised:
                    client.complete_json_object("test prompt")
            saved = list(Path(tmp).glob("judge-response_*.json"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text()), response)
        self.assertIs(raised.exception.response, response)

    def test_responses_payload_includes_explicit_reasoning_effort(self) -> None:
        http = RecordingHttp({"output_text": '{"ok": true}'})
        client = JudgeClient(
            http=http,
            api_key="test-key",
            model="gpt-5-nano",
            reasoning_effort="minimal",
        )

        parsed, _, _ = client.complete_json_object("test prompt")

        self.assertEqual(parsed, {"ok": True})
        self.assertEqual(http.posts[0][1]["reasoning"], {"effort": "minimal"})

    def test_chat_completions_payload_includes_explicit_reasoning_effort(self) -> None:
        http = RecordingHttp(
            {
                "choices": [
                    {
                        "message": {"content": '{"ok": true}'},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        client = JudgeClient(
            http=http,
            api_key="test-key",
            model="reasoning-model",
            api_style="chat_completions",
            reasoning_effort="high",
        )

        parsed, _, _ = client.complete_json_object("test prompt")

        self.assertEqual(parsed, {"ok": True})
        self.assertEqual(http.posts[0][1]["reasoning_effort"], "high")

    def test_chat_completions_does_not_send_none_reasoning_effort(self) -> None:
        http = RecordingHttp(
            {"choices": [{"message": {"content": '{"ok": true}'}}]}
        )
        client = JudgeClient(
            http=http,
            api_key="test-key",
            model="reasoning-model",
            api_style="chat_completions",
            reasoning_effort="none",
        )

        client.complete_json_object("test prompt")

        self.assertNotIn("reasoning_effort", http.posts[0][1])

    def test_memory_stage_streaming_combines_content_and_usage(self) -> None:
        chunks = [
            {
                "id": "stream-test",
                "model": "Qwen/Qwen3.8-Flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "hidden"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "stream-test",
                "model": "Qwen/Qwen3.8-Flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": '{"ok":'},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "stream-test",
                "model": "Qwen/Qwen3.8-Flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " true}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                },
            },
        ]
        http = RecordingHttp(chunks)
        client = JudgeClient(
            http=http,
            api_key="test-key",
            model="Qwen/Qwen3.8-Flash",
            api_style="chat_completions",
            stream_chat_completions=True,
        )

        parsed, response, _ = client.complete_json_object("test prompt")

        self.assertEqual(parsed, {"ok": True})
        self.assertTrue(http.posts[0][1]["stream"])
        self.assertEqual(
            http.posts[0][1]["stream_options"], {"include_usage": True}
        )
        self.assertEqual(response["usage"]["total_tokens"], 15)
        self.assertEqual(
            response["choices"][0]["message"]["reasoning_content"], "hidden"
        )

    def test_generic_chat_completion_remains_non_streaming(self) -> None:
        http = RecordingHttp(
            {"choices": [{"message": {"content": '{"ok": true}'}}]}
        )
        client = JudgeClient(
            http=http,
            api_key="test-key",
            api_style="chat_completions",
        )

        client.complete_json_object("test prompt")

        self.assertFalse(http.posts[0][1]["stream"])
        self.assertNotIn("stream_options", http.posts[0][1])

    def test_empty_output_reports_incomplete_response_metadata(self) -> None:
        response = {
            "id": "resp_test",
            "model": "gpt-5-nano",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "content": []}],
            "usage": {
                "input_tokens": 7552,
                "output_tokens": 2048,
                "total_tokens": 9600,
                "input_tokens_details": {"cached_tokens": 1024},
                "output_tokens_details": {"reasoning_tokens": 2048},
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LETTA_JUDGE_ERROR_DIR": tmp}):
                with self.assertRaisesRegex(ValueError, "incomplete_reason=max_output_tokens") as raised:
                    _extract_output_text(response)

            saved = list(Path(tmp).glob("judge-response_*.json"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text()), response)

        message = str(raised.exception)
        self.assertIn("id=resp_test", message)
        self.assertIn("model=gpt-5-nano", message)
        self.assertIn("status=incomplete", message)
        self.assertIn("input_tokens=7552", message)
        self.assertIn("reasoning_output_tokens=2048", message)
        self.assertIn("output_types=reasoning", message)
        self.assertIn(f"raw_response={saved[0].resolve()}", message)

    def test_diagnostic_does_not_expose_non_text_response_content(self) -> None:
        secret = "sensitive refusal details"
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": secret}],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LETTA_JUDGE_ERROR_DIR": tmp}):
                with self.assertRaises(ValueError) as raised:
                    _extract_output_text(response)

            saved = list(Path(tmp).glob("judge-response_*.json"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text()), response)

        message = str(raised.exception)
        self.assertIn("output_types=message[refusal]", message)
        self.assertNotIn(secret, message)

    def test_output_text_extraction_is_unchanged(self) -> None:
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"ok": true}'}],
                }
            ],
        }

        self.assertEqual(_extract_output_text(response), '{"ok": true}')


if __name__ == "__main__":
    unittest.main()
