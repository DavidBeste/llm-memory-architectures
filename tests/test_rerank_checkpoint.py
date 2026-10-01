import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from letta_research_chat.cli import _llm_rerank_memories


class RecordingHttp:
    def __init__(self, selected_indices):
        self.selected_indices = selected_indices
        self.posts = []

    def post(self, url, payload, headers, timeout):
        self.posts.append((url, payload, headers, timeout))
        return {
            "id": f"response-{len(self.posts)}",
            "model": "test-reranker",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"selected_indices": self.selected_indices}
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        }


class RerankCheckpointTests(unittest.TestCase):
    def _env(self, checkpoint_dir: str) -> dict[str, str]:
        return {
            "LETTA_RERANK_BASE_URL": "https://example.test/v1",
            "LETTA_RERANK_API_STYLE": "chat_completions",
            "LETTA_RERANK_API_KEY": "secret-not-written-to-checkpoint",
            "LETTA_RERANK_MODEL": "test-reranker",
            "LETTA_RERANK_MAX_TOKENS": "123",
            "LETTA_RERANK_CHECKPOINT_DIR": checkpoint_dir,
        }

    def test_successful_exact_call_is_reused_without_second_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_http = RecordingHttp([1, 0])
            with patch.dict(os.environ, self._env(tmp), clear=True):
                first = _llm_rerank_memories(
                    first_http,
                    recipient="Bank",
                    task="Apply",
                    query="Bank Apply",
                    candidates=["alpha", "beta"],
                    output_limit=1,
                    model="test-reranker",
                )

                second_http = RecordingHttp([0])
                second = _llm_rerank_memories(
                    second_http,
                    recipient="Bank",
                    task="Apply",
                    query="Bank Apply",
                    candidates=["alpha", "beta"],
                    output_limit=1,
                    model="test-reranker",
                )

            self.assertEqual(len(first_http.posts), 1)
            self.assertEqual(len(second_http.posts), 0)
            self.assertFalse(first.checkpoint_reused)
            self.assertTrue(second.checkpoint_reused)
            self.assertEqual(first.selected_memories, ["beta"])
            self.assertEqual(second.selected_memories, ["beta"])
            self.assertEqual(first.input_fingerprint, second.input_fingerprint)
            checkpoint = Path(second.checkpoint_path or "")
            self.assertTrue(checkpoint.is_file())
            self.assertNotIn(
                "secret-not-written-to-checkpoint",
                checkpoint.read_text(encoding="utf-8"),
            )

    def test_candidate_change_does_not_reuse_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, self._env(tmp), clear=True):
                first_http = RecordingHttp([0])
                first = _llm_rerank_memories(
                    first_http,
                    recipient="Bank",
                    task="Apply",
                    query="Bank Apply",
                    candidates=["alpha"],
                    output_limit=1,
                    model="test-reranker",
                )
                changed_http = RecordingHttp([0])
                changed = _llm_rerank_memories(
                    changed_http,
                    recipient="Bank",
                    task="Apply",
                    query="Bank Apply",
                    candidates=["different"],
                    output_limit=1,
                    model="test-reranker",
                )

            self.assertEqual(len(first_http.posts), 1)
            self.assertEqual(len(changed_http.posts), 1)
            self.assertNotEqual(first.input_fingerprint, changed.input_fingerprint)
            self.assertFalse(changed.checkpoint_reused)

    def test_indices_only_changes_only_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, self._env(tmp), clear=True):
                legacy = _llm_rerank_memories(
                    RecordingHttp([0]),
                    recipient="Bank",
                    task="Apply",
                    query="Bank Apply",
                    candidates=["alpha", "beta"],
                    output_limit=1,
                    model="test-reranker",
                    prompt_mode="legacy",
                )
                compact = _llm_rerank_memories(
                    RecordingHttp([0]),
                    recipient="Bank",
                    task="Apply",
                    query="Bank Apply",
                    candidates=["alpha", "beta"],
                    output_limit=1,
                    model="test-reranker",
                    prompt_mode="indices_only",
                )

            legacy_prefix, legacy_contract = (legacy.prompt or "").split(
                "\n\nRespond with a single JSON object:\n", 1
            )
            compact_prefix, compact_contract = (compact.prompt or "").split(
                "\n\nSelect at most 1 candidate indices.\n", 1
            )
            self.assertEqual(legacy_prefix, compact_prefix)
            self.assertIn('"rationale": "brief reason"', legacy_contract)
            self.assertNotIn("rationale", compact_contract)
            self.assertTrue(
                compact_contract.endswith('{"selected_indices":[0]}')
            )
            self.assertEqual(legacy.prompt_mode, "legacy")
            self.assertEqual(compact.prompt_mode, "indices_only")
            self.assertNotEqual(legacy.input_fingerprint, compact.input_fingerprint)

    def test_unknown_prompt_mode_fails_before_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            http = RecordingHttp([0])
            with patch.dict(os.environ, self._env(tmp), clear=True):
                with self.assertRaisesRegex(
                    ValueError, "LETTA_RERANK_PROMPT_MODE"
                ):
                    _llm_rerank_memories(
                        http,
                        recipient="Bank",
                        task="Apply",
                        query="Bank Apply",
                        candidates=["alpha"],
                        output_limit=1,
                        model="test-reranker",
                        prompt_mode="unknown",
                    )
            self.assertEqual(http.posts, [])


if __name__ == "__main__":
    unittest.main()
