import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from letta_research_chat.memory_stage_metrics import (  # noqa: E402
    _aggregate,
    _context_metrics,
    _judge_artifact_suffix,
    _judge_provenance,
    _parse_targeted_results,
)
from letta_research_chat.judge import JudgeResponseJSONError  # noqa: E402


class FakeJudge:
    model = "test-judge"
    api_style = "responses"

    def __init__(self):
        self.calls = 0

    def complete_json_object(self, prompt):
        self.calls += 1
        return (
            {
                "pre_rerank_indices": [0, 1, 2],
                "post_rerank_indices": [0, 2, 3],
            },
            {"usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}},
            prompt,
        )


class TargetedFakeJudge:
    model = "test-judge"
    api_style = "responses"
    reasoning_effort = "low"

    def __init__(self, *, omit_last_in_batches=False):
        self.calls = 0
        self.omit_last_in_batches = omit_last_in_batches

    def complete_json_object(self, prompt):
        self.calls += 1
        if "Target attributes:\n" in prompt:
            targets = json.loads(prompt.split("Target attributes:\n", 1)[1])
        else:
            targets = [json.loads(prompt.split("Target attribute:\n", 1)[1])]
        facts = {
            0: (True, True),
            1: (True, False),
            2: (True, True),
            3: (False, False),
        }
        results = []
        for target in targets:
            index = target["attribute_index"]
            pre, post = facts[index]
            results.append(
                {
                    "attribute_index": index,
                    "pre_rerank_present": pre,
                    "post_rerank_present": post,
                }
            )
        if len(results) == 1:
            results[0].pop("attribute_index")
            parsed = results[0]
        else:
            if self.omit_last_in_batches:
                results[-1] = dict(results[-2])
            parsed = {"results": results}
        raw = {"usage": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}}
        return parsed, raw, prompt


class MalformedBatchFakeJudge(TargetedFakeJudge):
    def complete_json_object(self, prompt):
        if "Target attributes:\n" in prompt:
            self.calls += 1
            response = {
                "usage": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
            }
            raise JudgeResponseJSONError("extra JSON value", response)
        return super().complete_json_object(prompt)


class MemoryStageMetricsTests(unittest.TestCase):
    def test_judge_artifacts_are_namespaced_without_saving_api_key(self):
        judge = FakeJudge()
        judge.base_url = "https://api.example/v1"
        judge.max_output_tokens = 4096
        judge.timeout = 300
        judge.api_key = "must-not-appear"
        judge.credential_source = "LETTA_MEMORY_STAGE_JUDGE_API_KEY"
        judge.configuration_source = "dedicated_memory_stage_environment"
        judge.stream_chat_completions = True

        provenance = _judge_provenance(judge)
        suffix = _judge_artifact_suffix(judge)

        self.assertIn("test-judge", suffix)
        self.assertTrue(provenance["credential_configured"])
        self.assertEqual(
            provenance["credential_source"],
            "LETTA_MEMORY_STAGE_JUDGE_API_KEY",
        )
        self.assertTrue(provenance["stream_chat_completions"])
        self.assertNotIn("must-not-appear", json.dumps(provenance))

    def test_exact_match_scores_list_memory_without_a_judge(self):
        _, record, labels = self._record_and_labels()
        with tempfile.TemporaryDirectory() as tmp:
            result, reused = _context_metrics(
                judge=None,
                record=record,
                labels=labels,
                output_path=Path(tmp) / "context_000.json",
                force=False,
                strategy="exact-match",
            )
        self.assertFalse(reused)
        self.assertEqual(result["represented"]["pre_rerank_indices"], [0, 1, 2])
        self.assertEqual(result["represented"]["post_rerank_indices"], [0, 2])
        self.assertEqual(result["judge_call_count"], 0)
        self.assertEqual(result["judge_exact_token_usage"]["total_tokens"], 0)
        self.assertIsNone(result["judge_model"])

    def test_exact_match_rejects_non_attribute_candidate_text(self):
        _, record, labels = self._record_and_labels()
        record["rerank"]["prompt"] = (
            "Candidate memories:\n"
            + json.dumps([{"index": 0, "memory": "Rephrased unknown fact"}])
        )
        record["rerank"]["candidate_count"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "not original attributes"):
                _context_metrics(
                    judge=None,
                    record=record,
                    labels=labels,
                    output_path=Path(tmp) / "context_000.json",
                    force=False,
                    strategy="exact-match",
                )

    def test_batch_results_are_matched_by_attribute_index_and_ignore_unknown_extra(self):
        parsed = {
            "results": [
                {
                    "attribute_index": index,
                    "pre_rerank_present": False,
                    "post_rerank_present": False,
                }
                for index in (52, 53, 54, 55, 56)
            ]
        }
        pre, post, normalized = _parse_targeted_results(parsed, [52, 53, 54, 55])
        self.assertEqual(pre, set())
        self.assertEqual(post, set())
        self.assertEqual([item["attribute_index"] for item in normalized], [52, 53, 54, 55])

    @staticmethod
    def _record_and_labels():
        attributes = ["Income is 72k", "Has hypertension", "Is an engineer", "Owns a boat"]
        record = {
            "user_idx": 0,
            "persona_name": "Test Person",
            "memory_mode": 3,
            "context_idx": 0,
            "recipient": "Bank",
            "task": "Loan",
            "attributes": attributes,
            "rerank": {
                "candidate_count": 3,
                "candidate_limit": 50,
                "prompt": "Candidate memories:\n" + json.dumps([
                    {"index": 0, "memory": "Income is 72k"},
                    {"index": 1, "memory": "Has hypertension"},
                    {"index": 2, "memory": "Is an engineer"},
                ]),
                "selected_memories": ["Income is 72k", "Is an engineer"],
            },
        }
        labels = {
            "necessary_attributes": [attributes[0]],
            "inappropriate_attributes": [attributes[1]],
            "ambiguous_attributes": [attributes[2], attributes[3]],
            "context_discarded": False,
            "metrics": {
                "repeat_count": 10,
                "necessary_recall": 0.8,
                "inappropriate_leak_rate": 0.1,
                "ambiguous_exposure_rate": 0.2,
            },
        }
        return attributes, record, labels

    def test_scores_pre_post_and_reuses_checkpoint(self):
        attributes, record, labels = self._record_and_labels()
        judge = FakeJudge()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "context_000.json"
            result, reused = _context_metrics(
                judge=judge, record=record, labels=labels, output_path=output, force=False
            )
            self.assertFalse(reused)
            self.assertEqual(result["pre_rerank"]["necessary"]["availability_rate"], 1.0)
            self.assertEqual(result["pre_rerank"]["private"]["availability_rate"], 1.0)
            self.assertEqual(result["post_rerank"]["private"]["availability_rate"], 0.0)
            self.assertEqual(result["reranker"]["private"]["removal_rate"], 1.0)
            self.assertEqual(result["final_response"]["necessary_recall"], 0.8)
            self.assertEqual(result["judge_consistency"]["post_only_indices_removed"], [3])

            cached, reused = _context_metrics(
                judge=judge, record=record, labels=labels, output_path=output, force=False
            )
            self.assertTrue(reused)
            self.assertEqual(cached["input_fingerprint"], result["input_fingerprint"])
            self.assertEqual(judge.calls, 1)

    def test_targeted_strategies_checkpoint_calls_and_aggregate_results(self):
        _, record, labels = self._record_and_labels()
        for strategy, batch_size, expected_calls in (
            ("per-attribute", 1, 4),
            ("attribute-batch", 2, 2),
        ):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as tmp:
                judge = TargetedFakeJudge()
                output = Path(tmp) / "context_000.json"
                result, reused = _context_metrics(
                    judge=judge,
                    record=record,
                    labels=labels,
                    output_path=output,
                    force=False,
                    strategy=strategy,
                    attribute_batch_size=batch_size,
                )
                self.assertFalse(reused)
                self.assertEqual(judge.calls, expected_calls)
                self.assertEqual(result["represented"]["pre_rerank_indices"], [0, 1, 2])
                self.assertEqual(result["represented"]["post_rerank_indices"], [0, 2])
                self.assertEqual(result["judge_call_count"], expected_calls)
                self.assertEqual(result["computed_judge_call_count"], expected_calls)
                call_files = sorted(Path(result["judge_calls_dir"]).glob("*.json"))
                self.assertEqual(len(call_files), expected_calls)
                self.assertIn(
                    "retained_after_rerank", call_files[0].read_text(encoding="utf-8")
                )

                cached, reused = _context_metrics(
                    judge=judge,
                    record=record,
                    labels=labels,
                    output_path=output,
                    force=False,
                    strategy=strategy,
                    attribute_batch_size=batch_size,
                )
                self.assertTrue(reused)
                self.assertEqual(cached["input_fingerprint"], result["input_fingerprint"])
                self.assertEqual(judge.calls, expected_calls)

    def test_attribute_batch_falls_back_only_for_omitted_indices(self):
        _, record, labels = self._record_and_labels()
        judge = TargetedFakeJudge(omit_last_in_batches=True)
        with tempfile.TemporaryDirectory() as tmp:
            result, reused = _context_metrics(
                judge=judge,
                record=record,
                labels=labels,
                output_path=Path(tmp) / "context_000.json",
                force=False,
                strategy="attribute-batch",
                attribute_batch_size=2,
            )
            self.assertFalse(reused)
            self.assertEqual(judge.calls, 4)
            self.assertEqual(result["judge_call_count"], 4)
            self.assertEqual(result["computed_judge_call_count"], 4)
            self.assertEqual(result["represented"]["pre_rerank_indices"], [0, 1, 2])
            call_files = sorted(Path(result["judge_calls_dir"]).glob("*.json"))
            self.assertEqual(
                [json.loads(path.read_text())["fallback_attribute_indices"] for path in call_files],
                [[1], [3]],
            )

    def test_malformed_batch_falls_back_for_all_batch_attributes(self):
        _, record, labels = self._record_and_labels()
        judge = MalformedBatchFakeJudge()
        with tempfile.TemporaryDirectory() as tmp:
            result, reused = _context_metrics(
                judge=judge,
                record=record,
                labels=labels,
                output_path=Path(tmp) / "context_000.json",
                force=False,
                strategy="attribute-batch",
                attribute_batch_size=2,
            )
            self.assertFalse(reused)
            self.assertEqual(judge.calls, 6)
            self.assertEqual(result["judge_call_count"], 6)
            self.assertEqual(result["represented"]["pre_rerank_indices"], [0, 1, 2])
            call_files = sorted(Path(result["judge_calls_dir"]).glob("*.json"))
            self.assertTrue(
                all(json.loads(path.read_text())["malformed_batch_response"] for path in call_files)
            )

    def test_aggregate_reports_macro_and_micro(self):
        def result(total, represented):
            category = {
                "total": total,
                "represented_count": represented,
                "availability_rate": represented / total,
            }
            stages = {name: dict(category) for name in ("necessary", "private", "ambiguous")}
            reranker = {
                name: {
                    "pre_rerank_represented_count": represented,
                    "post_rerank_represented_count": represented,
                    "retention_rate": 1.0,
                    "removal_rate": 0.0,
                }
                for name in ("necessary", "private", "ambiguous")
            }
            return {
                "status": "complete",
                "pre_rerank": stages,
                "post_rerank": stages,
                "reranker": reranker,
                "final_response": {
                    "necessary_recall": 0.5,
                    "private_leak_rate": 0.25,
                    "ambiguous_exposure_rate": 0.1,
                },
                "judge_consistency": {"post_only_indices_removed": []},
            }

        aggregate = _aggregate([result(1, 1), result(3, 0)])
        necessary = aggregate["pre_rerank"]["necessary"]
        self.assertEqual(necessary["macro_availability_rate"], 0.5)
        self.assertEqual(necessary["micro_availability_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
