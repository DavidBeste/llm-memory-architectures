from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from letta_research_chat.cli import (
    _cimemories_progress_entries,
    _estimate_token_usage_cost,
    _missing_memory_stage_entries,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class CimemoriesExperimentProgressTests(unittest.TestCase):
    def test_cost_estimate_applies_cached_input_rate(self) -> None:
        cost = _estimate_token_usage_cost(
            "gpt-5.2",
            {
                "input_tokens": 1_000_000,
                "cached_input_tokens": 600_000,
                "output_tokens": 100_000,
            },
        )
        self.assertAlmostEqual(cost or 0.0, 2.205)

    def test_connects_pre_stage_to_exact_pipeline_and_counts_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = root / "privacy-pipeline-cimemories_example"
            query = root / "query"
            responses = [
                {
                    "context_idx": context_idx,
                    "repeat_idx": repeat_idx,
                    "efficiency": {"timings_ms": {"generation": 2000.0}},
                }
                for context_idx in range(2)
                for repeat_idx in range(3)
            ]
            exposures = [{**record, "judge_error": None} for record in responses]
            write_jsonl(query / "responses.jsonl", responses)
            write_jsonl(query / "exposed.jsonl", exposures)
            pipeline_manifest = pipeline / "manifest.json"
            write_json(
                pipeline_manifest,
                {
                    "dataset_name": "cimemories_raw.json",
                    "persona_idx": 0,
                    "persona_name": "Test Person",
                    "agent_model": "test/model",
                    "memory_mode": 3,
                    "memory_mode_name": "list-rerank",
                    "context_count": 2,
                    "scenario_repeats": 3,
                    "contexts": [{"context_idx": 0}, {"context_idx": 1}],
                    "artifacts": {
                        "responses_jsonl": str((query / "responses.jsonl").resolve()),
                        "exposed_attributes_jsonl": str((query / "exposed.jsonl").resolve()),
                    },
                },
            )
            write_json(
                pipeline / "memory_stage_metrics_exact_match_summary.json",
                {
                    "source_pipeline_manifest": str(pipeline_manifest.resolve()),
                    "context_count": 2,
                    "strategy": "exact-match",
                    "judge_model": None,
                    "pilot_contexts": None,
                },
            )

            pre = pipeline / "pre_rerank_response_evaluation/personas/persona_000"
            pre_responses = pre / "responses.jsonl"
            write_jsonl(
                pre_responses,
                [record for record in responses if record["repeat_idx"] < 2],
            )
            write_json(
                pre / "manifest.json",
                {
                    "source_pipeline_manifest": str(pipeline_manifest.resolve()),
                    "status": "complete",
                    "evaluation_complete": True,
                    "artifacts": {"responses_jsonl": str(pre_responses.resolve())},
                },
            )

            entries = _cimemories_progress_entries(
                output_root=root,
                dataset_filter="cimemories_raw",
                model_filter=None,
            )

            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertTrue(entry["post_complete"])
            self.assertEqual((entry["post_contexts"], entry["post_repeats"]), (2, "3"))
            self.assertEqual(entry["post_generation_mean_ms"], 2000.0)
            self.assertTrue(entry["pre_complete"])
            self.assertEqual((entry["pre_contexts"], entry["pre_repeats"]), (2, "2"))
            self.assertEqual(entry["pre_generation_mean_ms"], 2000.0)
            self.assertTrue(entry["memory_stage_complete"])
            self.assertEqual(entry["memory_stage_contexts"], 2)
            self.assertEqual(entry["memory_stage_label"], "exact")

    def test_detects_namespaced_semantic_memory_stage_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = root / "privacy-pipeline-cimemories_semantic"
            query = root / "query"
            response = {"context_idx": 0, "repeat_idx": 0}
            write_jsonl(query / "responses.jsonl", [response])
            write_jsonl(query / "exposed.jsonl", [{**response, "judge_error": None}])
            pipeline_manifest = pipeline / "manifest.json"
            write_json(
                pipeline_manifest,
                {
                    "dataset_name": "cimemories_raw.json",
                    "persona_idx": 0,
                    "persona_name": "Test Person",
                    "agent_model": "test/model",
                    "memory_mode": 6,
                    "memory_mode_name": "graph-rerank",
                    "context_count": 1,
                    "scenario_repeats": 1,
                    "contexts": [{"context_idx": 0}],
                    "artifacts": {
                        "responses_jsonl": str((query / "responses.jsonl").resolve()),
                        "exposed_attributes_jsonl": str((query / "exposed.jsonl").resolve()),
                    },
                },
            )
            write_json(
                pipeline
                / "memory_stage_metrics_attribute_batch_4__judge_qwen_123_summary.json",
                {
                    "source_pipeline_manifest": str(pipeline_manifest.resolve()),
                    "context_count": 1,
                    "strategy": "attribute-batch",
                    "judge_model": "Qwen/Qwen3.8-Flash",
                    "judge_configuration": {
                        "configuration_source": "dedicated_memory_stage_environment"
                    },
                },
            )

            entries = _cimemories_progress_entries(
                output_root=root,
                dataset_filter=None,
                model_filter=None,
            )

            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0]["memory_stage_complete"])
            self.assertEqual(
                entries[0]["memory_stage_label"], "Qwen/Qwen3.8-Flash"
            )

    def test_incomplete_judging_marks_post_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = root / "privacy-pipeline-cimemories_partial"
            query = root / "query"
            responses = [{"context_idx": 0, "repeat_idx": 0}]
            write_jsonl(query / "responses.jsonl", responses)
            write_jsonl(query / "exposed.jsonl", [])
            write_json(
                pipeline / "manifest.json",
                {
                    "dataset_name": "cimemories_raw.json",
                    "persona_idx": 1,
                    "persona_name": "Test Person",
                    "agent_model": "test/model",
                    "memory_mode": 6,
                    "memory_mode_name": "graph-rerank",
                    "context_count": 1,
                    "scenario_repeats": 1,
                    "contexts": [{"context_idx": 0}],
                    "artifacts": {
                        "responses_jsonl": str((query / "responses.jsonl").resolve()),
                        "exposed_attributes_jsonl": str((query / "exposed.jsonl").resolve()),
                    },
                },
            )

            entries = _cimemories_progress_entries(
                output_root=root,
                dataset_filter=None,
                model_filter="test/model",
            )
            self.assertEqual(len(entries), 1)
            self.assertFalse(entries[0]["post_complete"])
            missing = _missing_memory_stage_entries(
                output_root=root,
                dataset_filter=None,
                model_filter="test/model",
                architecture="graph",
            )
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]["persona_idx"], 1)

    def test_missing_batch_skips_completed_memory_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = root / "privacy-pipeline-cimemories_complete_memory"
            query = root / "query"
            response = {"context_idx": 0, "repeat_idx": 0}
            write_jsonl(query / "responses.jsonl", [response])
            write_jsonl(query / "exposed.jsonl", [{**response, "judge_error": None}])
            write_json(
                pipeline / "manifest.json",
                {
                    "dataset_name": "cimemories_raw.json",
                    "persona_idx": 2,
                    "persona_name": "Complete Person",
                    "agent_model": "test/model",
                    "memory_mode": 3,
                    "memory_mode_name": "list-rerank",
                    "context_count": 1,
                    "scenario_repeats": 1,
                    "contexts": [{"context_idx": 0}],
                    "artifacts": {
                        "responses_jsonl": str((query / "responses.jsonl").resolve()),
                        "exposed_attributes_jsonl": str((query / "exposed.jsonl").resolve()),
                    },
                },
            )
            write_json(
                pipeline / "memory_stage_metrics_exact_match_summary.json",
                {
                    "context_count": 1,
                    "strategy": "exact-match",
                    "judge_model": None,
                },
            )

            missing = _missing_memory_stage_entries(
                output_root=root,
                dataset_filter="cimemories_raw",
                model_filter=None,
                architecture="list",
            )
            self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
