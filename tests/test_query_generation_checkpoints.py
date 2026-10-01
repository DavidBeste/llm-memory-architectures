from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from letta_research_chat.cli import (
    QUERY_RESUME_STATE_FILENAME,
    _find_query_resume_state,
    _load_query_generation_checkpoint,
    _query_resume_configuration,
    _query_resume_fingerprint,
    _run_query_contexts_async,
    run_query_recipients_with_tasks_experiment,
)
from letta_research_chat.config import LettaConfig


class FakeConversations:
    def __init__(self, *, fail_prompts: set[str] | None = None) -> None:
        self.fail_prompts = fail_prompts or set()
        self.prompts: list[str] = []
        self.next_id = 0

    def create_conversation(self, agent_id: str) -> dict[str, str]:
        conversation_id = f"conversation-{self.next_id}"
        self.next_id += 1
        return {"id": conversation_id}

    def send_conversation_message(self, conversation_id: str, prompt: str) -> dict:
        self.prompts.append(prompt)
        if prompt in self.fail_prompts:
            raise TimeoutError(f"timed out: {prompt}")
        return {
            "messages": [
                {
                    "message_type": "assistant_message",
                    "content": f"response for {prompt}",
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    def get_conversation_messages(self, conversation_id: str, limit: int = 500) -> list:
        return [{"conversation_id": conversation_id}]


class FakeAgent:
    def __init__(self) -> None:
        self.create_calls = 0

    def create_agent(self, *args, **kwargs) -> str:
        self.create_calls += 1
        return "agent-test"


class FakeMemory:
    pass


def run_contexts(
    conversations: FakeConversations,
    checkpoint_dir: Path,
    *,
    configuration: dict | None = None,
) -> list[dict]:
    contexts = [
        (0, {"recipient": "Recipient A", "task": "Task A"}),
        (1, {"recipient": "Recipient B", "task": "Task B"}),
    ]
    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    with patch("letta_research_chat.cli.asyncio.to_thread", new=direct_to_thread):
        return asyncio.run(
            _run_query_contexts_async(
                convos=conversations,
                experiment_agent_id="agent-test",
                contexts=contexts,
                prompts_by_context={0: "prompt-a", 1: "prompt-b"},
                list_search_by_context={},
                rerank_by_context={},
                graph_by_context={},
                attacker_rag_by_context={},
                attacker_injection_by_context={},
                profile_by_context={},
                repeats_per_context=1,
                concurrency=1,
                generation_checkpoint_configuration=configuration or {"model": "model-a"},
                generation_checkpoint_dir=checkpoint_dir,
            )
        )


class QueryGenerationCheckpointTests(unittest.TestCase):
    def test_interrupted_batch_reuses_completed_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            first = FakeConversations(fail_prompts={"prompt-b"})
            with self.assertRaises(TimeoutError):
                run_contexts(first, checkpoint_dir)

            self.assertEqual(first.prompts, ["prompt-a", "prompt-b"])
            self.assertEqual(len(list(checkpoint_dir.glob("*.json"))), 1)

            resumed = FakeConversations()
            results = run_contexts(resumed, checkpoint_dir)

            self.assertEqual(resumed.prompts, ["prompt-b"])
            by_context = {result["context_idx"]: result for result in results}
            self.assertTrue(by_context[0]["generation_checkpoint_reused"])
            self.assertFalse(by_context[1]["generation_checkpoint_reused"])
            self.assertEqual(by_context[0]["assistant_response"], "response for prompt-a")

    def test_configuration_change_does_not_reuse_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            run_contexts(FakeConversations(), checkpoint_dir, configuration={"model": "model-a"})

            changed = FakeConversations()
            results = run_contexts(
                changed,
                checkpoint_dir,
                configuration={"model": "model-b"},
            )

            self.assertEqual(changed.prompts, ["prompt-a", "prompt-b"])
            self.assertTrue(all(not result["generation_checkpoint_reused"] for result in results))
            self.assertEqual(len(list(checkpoint_dir.glob("*.json"))), 4)

    def test_corrupt_or_stale_checkpoint_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(
                _load_query_generation_checkpoint(path, expected_fingerprint="expected")
            )

            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "input_fingerprint": "different",
                        "result": {
                            "conversation_id": "conversation",
                            "assistant_response": "response",
                            "history_payload": [],
                            "efficiency": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(
                _load_query_generation_checkpoint(path, expected_fingerprint="expected")
            )

    def test_finds_only_matching_incomplete_prepared_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matching = root / "matching" / QUERY_RESUME_STATE_FILENAME
            matching.parent.mkdir()
            matching.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "prepared",
                        "configuration_fingerprint": "expected",
                        "run": {"experiment_agent_id": "agent"},
                        "prepared": {"prompts_by_context": {"0": "prompt"}},
                    }
                ),
                encoding="utf-8",
            )
            complete = root / "complete" / QUERY_RESUME_STATE_FILENAME
            complete.parent.mkdir()
            complete.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "configuration_fingerprint": "expected",
                        "run": {"experiment_agent_id": "agent"},
                        "prepared": {"prompts_by_context": {"0": "prompt"}},
                    }
                ),
                encoding="utf-8",
            )

            found = _find_query_resume_state(
                root, expected_fingerprint="expected"
            )
            self.assertIsNotNone(found)
            self.assertEqual(found[0], matching)

    def test_full_query_resume_reuses_prepared_plan_and_completed_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "bio": {"name": "Test Person"},
                            "information_attributes": {
                                "name": {"memory_statement": "My name is Test Person."}
                            },
                            "contexts": [
                                {"recipient": "A", "task": "Task A"},
                                {"recipient": "B", "task": "Task B"},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            outputs = root / "outputs"
            checkpoints = root / "generation-checkpoints"
            conversations = FakeConversations(fail_prompts={"prompt-b"})
            agent = FakeAgent()
            prepare_calls: list[int] = []

            def prepare_prompt(**kwargs):
                context_idx = int(kwargs["context_idx"])
                prepare_calls.append(context_idx)
                return (f"prompt-{'a' if context_idx == 0 else 'b'}",) + (
                    None,
                ) * 6

            def snapshot(**kwargs):
                directory = kwargs["output_dir"] / "initialized_memory"
                directory.mkdir(parents=True, exist_ok=True)
                return {"directory": str(directory)}

            environment = {
                "LETTA_QUERY_GENERATION_CHECKPOINT_DIR": str(checkpoints),
                "LETTA_RESEARCH_QUERY_CONCURRENCY": "1",
            }
            async def direct_to_thread(function, *args, **kwargs):
                return function(*args, **kwargs)

            patches = (
                patch(
                    "letta_research_chat.cli.resolve_dataset_filename",
                    return_value=str(dataset),
                ),
                patch(
                    "letta_research_chat.cli.build_query_recipient_prompt",
                    side_effect=prepare_prompt,
                ),
                patch(
                    "letta_research_chat.cli.capture_initialized_memory_snapshot",
                    side_effect=snapshot,
                ),
            )
            with (
                patch.dict(os.environ, environment, clear=False),
                patches[0],
                patches[1],
                patches[2],
                patch(
                    "letta_research_chat.cli.asyncio.to_thread",
                    new=direct_to_thread,
                ),
            ):
                with self.assertRaises(TimeoutError):
                    run_query_recipients_with_tasks_experiment(
                        cfg=LettaConfig(),
                        agent=agent,
                        convos=conversations,
                        mem=FakeMemory(),
                        dataset_name=str(dataset),
                        user_idx=0,
                        memory_mode=8,
                        use_convos=True,
                        repeats_per_context=1,
                        output_root=outputs,
                        resume_compatible=True,
                    )

                self.assertEqual(agent.create_calls, 1)
                self.assertEqual(prepare_calls, [0, 1])
                self.assertEqual(len(list(outputs.rglob(QUERY_RESUME_STATE_FILENAME))), 1)

                conversations.fail_prompts.clear()
                output = run_query_recipients_with_tasks_experiment(
                    cfg=LettaConfig(),
                    agent=agent,
                    convos=conversations,
                    mem=FakeMemory(),
                    dataset_name=str(dataset),
                    user_idx=0,
                    memory_mode=8,
                    use_convos=True,
                    repeats_per_context=1,
                    output_root=outputs,
                    resume_compatible=True,
                )

            self.assertEqual(agent.create_calls, 1)
            self.assertEqual(prepare_calls, [0, 1])
            self.assertEqual(conversations.prompts.count("prompt-a"), 1)
            self.assertEqual(conversations.prompts.count("prompt-b"), 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(
                manifest["generation_checkpointing"][
                    "resumed_prepared_query_in_place"
                ]
            )
            self.assertEqual(
                manifest["generation_checkpointing"]["reused_records"], 1
            )
            self.assertEqual(manifest["generation_checkpointing"]["new_records"], 1)
            state = json.loads(
                (output / QUERY_RESUME_STATE_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "complete")

    def test_prepared_resume_skips_list_graph_and_profile_backends(self) -> None:
        for memory_mode, result_field in (
            (3, "rerank_by_context"),
            (6, "graph_by_context"),
            (19, "profile_by_context"),
        ):
            with self.subTest(memory_mode=memory_mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dataset = root / "dataset.json"
                dataset.write_text(
                    json.dumps(
                        [
                            {
                                "bio": {"name": "Test Person"},
                                "information_attributes": {
                                    "name": {
                                        "memory_statement": "My name is Test Person."
                                    }
                                },
                                "contexts": [
                                    {"recipient": "A", "task": "Task A"}
                                ],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                outputs = root / "outputs"
                query_output = outputs / f"prepared-mode-{memory_mode}"
                query_output.mkdir(parents=True)
                (query_output / "histories").mkdir()
                cfg = LettaConfig()
                contexts = [(0, {"recipient": "A", "task": "Task A"})]
                attributes = ["My name is Test Person."]
                configuration = _query_resume_configuration(
                    cfg,
                    filename=str(dataset),
                    user_idx=0,
                    persona_name="Test Person",
                    attributes=attributes,
                    valid_contexts=contexts,
                    memory_mode=memory_mode,
                    agent_model=cfg.agent_model,
                    repeats_per_context=1,
                    include_snapshot_embeddings=False,
                )
                prepared_results = {
                    "list_search_by_context": {},
                    "rerank_by_context": {},
                    "graph_by_context": {},
                    "attacker_rag_by_context": {},
                    "attacker_injection_by_context": {},
                    "profile_by_context": {},
                }
                prepared_results[result_field] = {
                    "0": {
                        "query": "saved query",
                        "selected_memories": ["saved memory"],
                        "candidate_count": 1,
                    }
                }
                state = {
                    "schema_version": 1,
                    "status": "generation_in_progress",
                    "configuration": configuration,
                    "configuration_fingerprint": _query_resume_fingerprint(
                        configuration
                    ),
                    "run": {
                        "timestamp": "20260925_000000",
                        "experiment_id": f"prepared-mode-{memory_mode}",
                        "experiment_agent_name": "prepared-agent",
                        "experiment_agent_id": "agent-prepared",
                        "agent_creation_ms": 1.0,
                        "graph_group_id": (
                            "saved-graph-group" if memory_mode == 6 else None
                        ),
                    },
                    "prepared": {
                        "archival_backup": [],
                        "init_result": {
                            "persona_index": -1,
                            "label": "Test Person",
                            "found": 1,
                            "inserted": 1,
                            "failed": 0,
                        },
                        "profile_user_id": (
                            "saved-profile-user" if memory_mode == 19 else None
                        ),
                        "profile_init_result": (
                            {
                                "backend": "memobase",
                                "user_id": "saved-profile-user",
                                "inserted_count": 1,
                                "flush_sync": True,
                            }
                            if memory_mode == 19
                            else None
                        ),
                        "profile_config_updated": memory_mode == 19,
                        "profile_server_config": None,
                        "initialization_usage_before": None,
                        "initialization_usage_after": None,
                        "retrieval_usage_before": None,
                        "retrieval_usage_after": None,
                        "backend_usage_events": [],
                        "graph_ingestion_warning_records": [],
                        "graph_ingestion_diagnostics": None,
                        "memory_initialization_ms": 2.0,
                        "initialization_snapshot": {
                            "directory": str(query_output / "initialized_memory")
                        },
                        "prompts_by_context": {"0": "saved prompt"},
                        **prepared_results,
                        "context_preparation_efficiency": {
                            "0": {
                                "duration_ms": 3.0,
                                "exact_model_tokens": None,
                                "exact_model_tokens_expected": False,
                            }
                        },
                        "preparation_elapsed_ms": 6.0,
                    },
                }
                (query_output / QUERY_RESUME_STATE_FILENAME).write_text(
                    json.dumps(state), encoding="utf-8"
                )

                async def direct_to_thread(function, *args, **kwargs):
                    return function(*args, **kwargs)

                environment = {
                    "LETTA_QUERY_GENERATION_CHECKPOINT_DIR": str(
                        root / "generation-checkpoints"
                    ),
                    "LETTA_RESEARCH_QUERY_CONCURRENCY": "1",
                }
                with (
                    patch.dict(os.environ, environment, clear=False),
                    patch(
                        "letta_research_chat.cli.resolve_dataset_filename",
                        return_value=str(dataset),
                    ),
                    patch(
                        "letta_research_chat.cli.open_graph_memory",
                        side_effect=AssertionError("graph backend reopened"),
                    ),
                    patch(
                        "letta_research_chat.cli.open_profile_memory",
                        side_effect=AssertionError("profile backend reopened"),
                    ),
                    patch(
                        "letta_research_chat.cli.build_query_recipient_prompt",
                        side_effect=AssertionError("prompt plan rebuilt"),
                    ),
                    patch(
                        "letta_research_chat.cli.asyncio.to_thread",
                        new=direct_to_thread,
                    ),
                ):
                    output = run_query_recipients_with_tasks_experiment(
                        cfg=cfg,
                        agent=FakeAgent(),
                        convos=FakeConversations(),
                        mem=FakeMemory(),
                        dataset_name=str(dataset),
                        user_idx=0,
                        memory_mode=memory_mode,
                        use_convos=True,
                        repeats_per_context=1,
                        output_root=outputs,
                        resume_compatible=True,
                    )

                self.assertEqual(output, query_output)
                manifest = json.loads(
                    (output / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertTrue(
                    manifest["generation_checkpointing"][
                        "resumed_prepared_query_in_place"
                    ]
                )


if __name__ == "__main__":
    unittest.main()
