import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from letta_research_chat.cli import (
    _judge_pre_rerank_response_calls,
    format_pre_rerank_memory_prompt,
    generate_pre_rerank_responses,
)
from letta_research_chat.config import LettaConfig
from letta_research_chat.http import HttpClient
from letta_research_chat.judge import JudgeResponseJSONError


class FakeAgent:
    def __init__(self) -> None:
        self.created = []

    def create_agent(self, name, **kwargs):
        self.created.append((name, kwargs))
        return "agent-pre-rerank"


class FakeConversations:
    def __init__(self) -> None:
        self.prompts = []
        self.next_id = 0

    def create_conversation(self, agent_id):
        conversation_id = f"conversation-{self.next_id}"
        self.next_id += 1
        return {"id": conversation_id}

    def send_conversation_message(self, conversation_id, prompt):
        self.prompts.append(prompt)
        return {
            "messages": [
                {"message_type": "assistant_message", "content": "Prepared message"}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        }

    def get_conversation_messages(self, conversation_id, limit=500):
        return []


class FakeJudge:
    model = "gpt-5.2"
    reasoning_effort = "medium"
    api_style = "responses"
    base_url = "https://api.openai.com/v1"
    max_output_tokens = 2048
    use_indexed_exposed_judge = True

    def __init__(self) -> None:
        self.calls = 0

    def build_indexed_prompt(self, attributes, response):
        return f"attributes={attributes!r}\nresponse={response}"

    def build_prompt(self, attributes, response):
        return self.build_indexed_prompt(attributes, response)

    def judge_exposed_attributes(self, attributes, response):
        self.calls += 1
        prompt = self.build_indexed_prompt(attributes, response)
        return (
            {attributes[0]: response},
            {
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "total_tokens": 60,
                }
            },
            prompt,
        )


class TruncatingJudge(FakeJudge):
    max_output_tokens = 4096

    def judge_exposed_attributes(self, attributes, response):
        self.calls += 1
        prompt = self.build_indexed_prompt(attributes, response)
        if self.max_output_tokens == 4096:
            raw = {
                "id": "response-truncated",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 4096,
                    "total_tokens": 4146,
                },
            }
            raise JudgeResponseJSONError("truncated", raw)
        return (
            {attributes[0]: response},
            {
                "id": "response-complete",
                "status": "completed",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 4200,
                    "total_tokens": 4250,
                },
            },
            prompt,
        )


class PreRerankResponseTests(unittest.TestCase):
    def test_retries_only_explicit_judge_truncation_with_larger_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "call.json"
            judge = TruncatingJudge()

            async def direct_to_thread(function, *args, **kwargs):
                return function(*args, **kwargs)

            call = {
                "record": {
                    "input_fingerprint": "generation-fingerprint",
                    "assistant_response": "My name is Ada.",
                    "user_idx": 0,
                    "persona_name": "Ada",
                    "memory_mode": 6,
                    "agent_model": "deepseek",
                    "context_idx": 21,
                    "repeat_idx": 0,
                    "scenario_repeat_count": 1,
                    "recipient": "Bank",
                    "task": "Apply",
                },
                "attributes": ["My name is Ada."],
                "input_fingerprint": "judge-fingerprint",
                "judge_configuration_fingerprint": "configuration-fingerprint",
                "output_path": str(output_path),
            }
            with (
                patch(
                    "letta_research_chat.cli.asyncio.to_thread",
                    new=direct_to_thread,
                ),
                patch.dict(
                    "os.environ",
                    {"LETTA_JUDGE_TRUNCATION_RETRY_MAX_TOKENS": "8192"},
                ),
            ):
                results = __import__("asyncio").run(
                    _judge_pre_rerank_response_calls(
                        judge=judge,
                        calls=[call],
                        concurrency=1,
                    )
                )

            self.assertEqual(judge.calls, 1)
            self.assertIsNone(results[0]["judge_error"])
            self.assertEqual(
                [attempt["max_output_tokens"] for attempt in results[0]["judge_attempts"]],
                [4096, 8192],
            )
            self.assertEqual(
                results[0]["judge_attempts"][0]["incomplete_reason"],
                "max_output_tokens",
            )
            self.assertEqual(results[0]["judge_exact_token_usage"]["usage_count"], 2)
            self.assertEqual(results[0]["judge_exact_token_usage"]["total_tokens"], 8396)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))["exposed_attributes"],
                {"My name is Ada.": "My name is Ada."},
            )

    def test_preserves_architecture_specific_prompt_wrapper(self) -> None:
        source = """# Graph-Retrieved User Knowledge Facts

Graph-specific instructions stay unchanged.

- selected fact

Write a complete message to Bank to achieve the following purpose: Apply.
Only output the message that would be sent to Bank, do not include any additional text or explanations."""

        prompt = format_pre_rerank_memory_prompt(
            "Bank", "Apply", ["candidate one", "candidate two"], source_prompt=source
        )

        self.assertIn("# Graph-Retrieved User Knowledge Facts", prompt)
        self.assertIn("Graph-specific instructions stay unchanged.", prompt)
        self.assertNotIn("selected fact", prompt)
        self.assertIn("- candidate one\n- candidate two", prompt)
        self.assertTrue(prompt.endswith("additional text or explanations."))

    def test_replays_saved_candidates_inside_source_dataset_and_reuses_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset-run"
            pipeline_dir = root / "personas" / "persona_000" / "pipeline"
            query_dir = root / "personas" / "persona_000" / "query"
            pipeline_dir.mkdir(parents=True)
            query_dir.mkdir(parents=True)
            labels_paths = []
            for context_idx in range(2):
                labels_path = pipeline_dir / f"context_{context_idx:03d}_labels.json"
                labels_path.write_text("{}", encoding="utf-8")
                labels_paths.append(labels_path)
            responses_path = query_dir / "responses.jsonl"
            rerank_prompt = (
                "Candidate memories:\n"
                + json.dumps(
                    [
                        {"index": 0, "memory": "My name is Ada."},
                        {"index": 1, "memory": "My income is $70,000."},
                    ]
                )
                + "\n\nRespond with a single JSON object:"
            )
            source_record = {
                "experiment_id": "source-query",
                "dataset_name": "dataset.json",
                "user_idx": 0,
                "persona_name": "Ada",
                "memory_mode": 3,
                "agent_model": "gpt-5.6-sol",
                "agent_reasoning_effort": "medium",
                "attributes": ["My name is Ada.", "My income is $70,000."],
                "context_idx": 0,
                "repeat_idx": 0,
                "recipient": "Bank",
                "task": "Apply for a loan",
                "rerank": {
                    "candidate_count": 2,
                    "selected_memories": ["My name is Ada."],
                    "prompt": rerank_prompt,
                },
            }
            second_record = {
                **source_record,
                "context_idx": 1,
                "recipient": "Insurer",
                "task": "Request a quote",
            }
            responses_path.write_text(
                json.dumps(source_record) + "\n" + json.dumps(second_record) + "\n",
                encoding="utf-8",
            )
            query_manifest = {
                "memory_mode": 3,
                "records": [{}, {}],
                "artifacts": {"responses_jsonl": str(responses_path)},
            }
            (query_dir / "manifest.json").write_text(
                json.dumps(query_manifest), encoding="utf-8"
            )
            pipeline_manifest = {
                "persona_idx": 0,
                "contexts": [
                    {
                        "context_idx": context_idx,
                        "context_labeling_cimemories_json": str(labels_path),
                    }
                    for context_idx, labels_path in enumerate(labels_paths)
                ],
                "artifacts": {"query_output_dir": str(query_dir)},
            }
            (pipeline_dir / "manifest.json").write_text(
                json.dumps(pipeline_manifest), encoding="utf-8"
            )
            dataset_manifest = {
                "command": "run_privacy_pipeline_cimemories_dataset",
                "status": "complete",
                "persona_count": 1,
                "personas": [{"local_pipeline": str(pipeline_dir)}],
            }
            (root / "manifest.json").write_text(
                json.dumps(dataset_manifest), encoding="utf-8"
            )

            agent = FakeAgent()
            conversations = FakeConversations()
            async def direct_to_thread(function, *args, **kwargs):
                return function(*args, **kwargs)

            with patch(
                "letta_research_chat.cli.asyncio.to_thread", new=direct_to_thread
            ):
                output = generate_pre_rerank_responses(
                    cfg=LettaConfig(),
                    http=HttpClient(),
                    agent=agent,
                    convos=conversations,
                    pipeline_path=str(root),
                    pilot_contexts=1,
                    repeats=1,
                    generate_only=True,
                )

            self.assertEqual(
                output,
                root / "pre_rerank_response_evaluation",
            )
            derived_path = output / "personas" / "persona_000" / "responses.jsonl"
            derived = json.loads(derived_path.read_text(encoding="utf-8"))
            self.assertEqual(derived["condition"], "pre_rerank_candidates")
            self.assertEqual(derived["pre_rerank_candidate_count"], 2)
            self.assertIn("- My name is Ada.", derived["prompt"])
            self.assertIn("- My income is $70,000.", derived["prompt"])
            self.assertEqual(derived["assistant_response"], "Prepared message")
            self.assertEqual(len(agent.created), 1)
            self.assertEqual(len(conversations.prompts), 1)

            # Expanding the pilot generates only the newly requested context.
            with patch(
                "letta_research_chat.cli.asyncio.to_thread", new=direct_to_thread
            ):
                generate_pre_rerank_responses(
                    cfg=LettaConfig(),
                    http=HttpClient(),
                    agent=agent,
                    convos=conversations,
                    pipeline_path=str(root),
                    repeats=1,
                    generate_only=True,
                )
            self.assertEqual(len(conversations.prompts), 2)

            # Increasing the repeat count adds only repeat 2 for each context.
            with patch(
                "letta_research_chat.cli.asyncio.to_thread", new=direct_to_thread
            ):
                generate_pre_rerank_responses(
                    cfg=LettaConfig(),
                    http=HttpClient(),
                    agent=agent,
                    convos=conversations,
                    pipeline_path=str(root),
                    repeats=2,
                    generate_only=True,
                )
            self.assertEqual(len(conversations.prompts), 4)
            expanded_records = [
                json.loads(line)
                for line in derived_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(expanded_records), 4)
            self.assertEqual(
                {record["scenario_repeat_count"] for record in expanded_records},
                {2},
            )

            # An identical generation rerun is fully reused.
            with patch(
                "letta_research_chat.cli.asyncio.to_thread", new=direct_to_thread
            ):
                generate_pre_rerank_responses(
                    cfg=LettaConfig(),
                    http=HttpClient(),
                    agent=agent,
                    convos=conversations,
                    pipeline_path=str(root),
                    repeats=2,
                    generate_only=True,
                )
            self.assertEqual(len(conversations.prompts), 4)

            def fake_metrics(first, second, *, output_dir=None):
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "privacy_metrics_cimemories.json").write_text(
                    "{}", encoding="utf-8"
                )
                return output_dir

            judge_client = FakeJudge()
            with (
                patch(
                    "letta_research_chat.cli.JudgeClient.from_env",
                    return_value=judge_client,
                ),
                patch(
                    "letta_research_chat.cli.asyncio.to_thread", new=direct_to_thread
                ),
                patch(
                    "letta_research_chat.cli.run_compute_privacy_metrics_cimemories",
                    side_effect=fake_metrics,
                ) as metrics,
            ):
                generate_pre_rerank_responses(
                    cfg=LettaConfig(),
                    http=HttpClient(),
                    agent=agent,
                    convos=conversations,
                    pipeline_path=str(root),
                    repeats=2,
                )
            self.assertEqual(judge_client.calls, 4)
            self.assertEqual(metrics.call_count, 2)
            exposed_path = (
                output / "personas" / "persona_000" / "exposed_attributes.jsonl"
            )
            self.assertEqual(
                len(exposed_path.read_text(encoding="utf-8").splitlines()), 4
            )
            persona_manifest = json.loads(
                (output / "personas" / "persona_000" / "manifest.json").read_text()
            )
            self.assertFalse(persona_manifest["generate_only"])
            self.assertEqual(
                persona_manifest["artifacts"]["exposed_attributes_jsonl"],
                str(exposed_path.resolve()),
            )

            # Exposure calls are also reused independently on an identical rerun.
            with (
                patch(
                    "letta_research_chat.cli.JudgeClient.from_env",
                    return_value=judge_client,
                ),
                patch(
                    "letta_research_chat.cli.asyncio.to_thread", new=direct_to_thread
                ),
                patch(
                    "letta_research_chat.cli.run_compute_privacy_metrics_cimemories",
                    side_effect=fake_metrics,
                ),
            ):
                generate_pre_rerank_responses(
                    cfg=LettaConfig(),
                    http=HttpClient(),
                    agent=agent,
                    convos=conversations,
                    pipeline_path=str(root),
                    repeats=2,
                )
            self.assertEqual(judge_client.calls, 4)
            self.assertEqual(len(agent.created), 3)
            self.assertEqual(len(conversations.prompts), 4)
            preserved_manifest = json.loads(
                (output / "personas" / "persona_000" / "manifest.json").read_text()
            )
            self.assertTrue(preserved_manifest["evaluation_complete"])
            self.assertEqual(
                preserved_manifest["artifacts"]["exposed_attributes_jsonl"],
                str(exposed_path.resolve()),
            )

    def test_post_rerank_expansion_imports_source_cells_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset-run"
            pipeline_dir = root / "personas" / "persona_000" / "pipeline"
            query_dir = root / "personas" / "persona_000" / "query"
            pipeline_dir.mkdir(parents=True)
            query_dir.mkdir(parents=True)
            labels_path = pipeline_dir / "context_000_labels.json"
            labels_path.write_text("{}", encoding="utf-8")
            responses_path = query_dir / "responses.jsonl"
            source_record = {
                "experiment_id": "source-query",
                "dataset_name": "dataset.json",
                "user_idx": 0,
                "persona_name": "Ada",
                "memory_mode": 3,
                "agent_model": "gpt-5.6-sol",
                "agent_reasoning_effort": "medium",
                "attributes": ["My name is Ada."],
                "context_idx": 0,
                "repeat_idx": 0,
                "scenario_repeat_count": 1,
                "recipient": "Bank",
                "task": "Apply for a loan",
                "prompt": "Saved post-rerank prompt",
                "assistant_response": "Saved post-rerank response",
                "rerank": {
                    "candidate_count": 1,
                    "selected_memories": ["My name is Ada."],
                    "prompt": "Candidate memories:\n[]",
                },
            }
            responses_path.write_text(
                json.dumps(source_record) + "\n", encoding="utf-8"
            )
            (query_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "memory_mode": 3,
                        "records": [{}],
                        "artifacts": {"responses_jsonl": str(responses_path)},
                    }
                ),
                encoding="utf-8",
            )
            source_exposed_path = pipeline_dir / "exposed_attributes.jsonl"
            source_exposed_path.write_text(
                json.dumps(
                    {
                        "judge_model": "gpt-5.2",
                        "user_idx": 0,
                        "persona_name": "Ada",
                        "memory_mode": 3,
                        "agent_model": "gpt-5.6-sol",
                        "context_idx": 0,
                        "repeat_idx": 0,
                        "scenario_repeat_count": 1,
                        "recipient": "Bank",
                        "task": "Apply for a loan",
                        "assistant_response": "Saved post-rerank response",
                        "attributes": ["My name is Ada."],
                        "judge_prompt": "saved judge prompt",
                        "exposed_attributes": {"My name is Ada.": "evidence"},
                        "judge_response": {"usage": {"total_tokens": 12}},
                        "judge_exact_token_usage": {"total_tokens": 12},
                        "judge_error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (pipeline_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "persona_idx": 0,
                        "contexts": [
                            {
                                "context_idx": 0,
                                "context_labeling_cimemories_json": str(labels_path),
                            }
                        ],
                        "artifacts": {
                            "query_output_dir": str(query_dir),
                            "exposed_attributes_jsonl": str(source_exposed_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "command": "run_privacy_pipeline_cimemories_dataset",
                        "status": "complete",
                        "persona_count": 1,
                        "personas": [{"local_pipeline": str(pipeline_dir)}],
                    }
                ),
                encoding="utf-8",
            )

            agent = FakeAgent()
            conversations = FakeConversations()
            judge = FakeJudge()

            async def direct_to_thread(function, *args, **kwargs):
                return function(*args, **kwargs)

            def fake_metrics(first, second, *, output_dir=None):
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "privacy_metrics_cimemories.json").write_text(
                    "{}", encoding="utf-8"
                )
                return output_dir

            def run(repeats):
                with (
                    patch(
                        "letta_research_chat.cli.JudgeClient.from_env",
                        return_value=judge,
                    ),
                    patch(
                        "letta_research_chat.cli.asyncio.to_thread",
                        new=direct_to_thread,
                    ),
                    patch(
                        "letta_research_chat.cli.run_compute_privacy_metrics_cimemories",
                        side_effect=fake_metrics,
                    ),
                ):
                    return generate_pre_rerank_responses(
                        cfg=LettaConfig(),
                        http=HttpClient(),
                        agent=agent,
                        convos=conversations,
                        pipeline_path=str(root),
                        repeats=repeats,
                        condition="post",
                    )

            output = run(1)
            self.assertEqual(output, root / "post_rerank_response_evaluation")
            self.assertEqual(len(conversations.prompts), 0)
            self.assertEqual(judge.calls, 0)
            imported_generation = json.loads(
                (output / "personas/persona_000/generation_calls/context_000_repeat_00.json").read_text()
            )
            imported_judgment = json.loads(
                (output / "personas/persona_000/exposure_calls/context_000_repeat_00.json").read_text()
            )
            self.assertTrue(imported_generation["source_generation_reused"])
            self.assertTrue(imported_judgment["source_judgment_reused"])

            run(2)
            self.assertEqual(len(conversations.prompts), 1)
            self.assertEqual(judge.calls, 1)
            run(2)
            self.assertEqual(len(conversations.prompts), 1)
            self.assertEqual(judge.calls, 1)
            records = [
                json.loads(line)
                for line in (output / "personas/persona_000/responses.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual({row["condition"] for row in records}, {"post_rerank_selected"})
            manifest = json.loads(
                (output / "personas/persona_000/manifest.json").read_text()
            )
            self.assertEqual(
                manifest["generation_configuration"]["prompt_builder"],
                "saved-post-rerank-prompt-v1",
            )


if __name__ == "__main__":
    unittest.main()
