import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from letta_research_chat.cli import (
    _localize_privacy_pipeline,
    _rewrite_local_artifact_paths,
    run_privacy_pipeline_cimemories_dataset,
)


class DatasetPrivacyPipelineTests(unittest.TestCase):
    def test_dataset_run_propagates_labels_file_to_every_persona(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels = root / "labels.json"
            labels.write_text("[]", encoding="utf-8")

            with (
                patch("letta_research_chat.cli.resolve_dataset_filename", return_value="dataset.json"),
                patch("letta_research_chat.cli.load_persona_dataset", return_value=[{}, {}]),
                patch("letta_research_chat.cli.persona_label", return_value="Persona"),
                patch("letta_research_chat.cli.extract_persona_memory_statements", return_value=["memory"]),
                patch("letta_research_chat.cli._iter_valid_contexts", return_value=[(0, {})]),
                patch("letta_research_chat.cli.load_embedded_cimemories_labelings") as load_labels,
                patch("letta_research_chat.cli._privacy_pipeline_reuse_signature", return_value={}),
                patch("letta_research_chat.cli.run_privacy_pipeline_cimemories") as run_persona,
                patch("letta_research_chat.cli._localize_privacy_pipeline") as localize,
                patch("letta_research_chat.cli.compute_memory_stage_metrics") as stage_metrics,
            ):
                run_persona.side_effect = [root / "source_0", root / "source_1"]
                localize.side_effect = lambda source, persona_root, copy_primary: persona_root / "pipeline"
                stage_metrics.side_effect = lambda http, path, **kwargs: Path(path).parent / "memory_stage_metrics_summary.json"
                cfg = type("Config", (), {"agent_model": "agent-model", "agent_reasoning_effort": "medium"})()

                output = run_privacy_pipeline_cimemories_dataset(
                    cfg=cfg,
                    http=object(),
                    agent=object(),
                    convos=object(),
                    mem=object(),
                    dataset_name="dataset.json",
                    memory_mode=3,
                    use_convos=False,
                    labels_filename=str(labels),
                    output_root=root,
                )

            self.assertEqual(run_persona.call_count, 2)
            self.assertEqual(load_labels.call_count, 2)
            self.assertEqual(stage_metrics.call_count, 3)
            for call in run_persona.call_args_list:
                self.assertEqual(call.kwargs["labels_filename"], str(labels))
                self.assertFalse(call.kwargs["compute_stage_metrics"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["context_labels_file"], str(labels.resolve()))

    def test_rewrite_paths_handles_nested_json_values(self):
        payload = {
            "artifact": "old/query/responses.jsonl",
            "nested": [{"history": "old/query/histories/one.json"}],
            "unrelated": "old/queryish/file.json",
        }
        rewritten = _rewrite_local_artifact_paths(payload, {"old/query": "new/query"})
        self.assertEqual(rewritten["artifact"], "new/query/responses.jsonl")
        self.assertEqual(rewritten["nested"][0]["history"], "new/query/histories/one.json")
        self.assertEqual(rewritten["unrelated"], "old/queryish/file.json")

    def test_localize_copies_all_referenced_artifacts_and_rewrites_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_pipeline = root / "old_pipeline"
            source_query = root / "old_query"
            source_labeling = root / "old_labeling"
            source_metrics = root / "old_metrics"
            for directory in (source_pipeline, source_query, source_labeling, source_metrics):
                directory.mkdir()

            responses = source_query / "responses.jsonl"
            history = source_query / "history.json"
            exposed = source_query / "exposed.jsonl"
            labeling = source_labeling / "context_labeling_cimemories.json"
            metrics = source_metrics / "privacy_metrics_cimemories.json"
            history.write_text("{}", encoding="utf-8")
            responses.write_text(
                json.dumps({"history_json_path": str(history)}) + "\n",
                encoding="utf-8",
            )
            exposed.write_text(
                json.dumps({"source_responses_jsonl": str(responses)}) + "\n",
                encoding="utf-8",
            )
            labeling.write_text("{}", encoding="utf-8")
            metrics.write_text("{}", encoding="utf-8")
            summary = {
                "artifacts": {
                    "query_output_dir": str(source_query),
                    "responses_jsonl": str(responses),
                    "exposed_attributes_jsonl": str(exposed),
                },
                "contexts": [{
                    "context_idx": 0,
                    "context_labeling_cimemories_json": str(labeling),
                    "privacy_metrics_cimemories_json": str(metrics),
                }],
            }
            (source_pipeline / "privacy_pipeline_cimemories.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            destination = root / "dataset_run" / "personas" / "persona_000"

            localized = _localize_privacy_pipeline(source_pipeline, destination, copy_primary=True)

            self.assertTrue(source_pipeline.is_dir())
            self.assertTrue((destination / "query" / "responses.jsonl").is_file())
            self.assertTrue((destination / "context_labelings/context_000" / labeling.name).is_file())
            self.assertTrue((destination / "privacy_metrics/context_000" / metrics.name).is_file())
            localized_summary = json.loads(
                (localized / "privacy_pipeline_cimemories.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                localized_summary["artifacts"]["responses_jsonl"],
                str(destination / "query" / "responses.jsonl"),
            )
            localized_response = json.loads(
                (destination / "query" / "responses.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(
                localized_response["history_json_path"],
                str(destination / "query" / "history.json"),
            )

    def test_localize_keeps_pipeline_internal_imported_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_pipeline = root / "old_pipeline"
            source_query = root / "old_query"
            source_metrics = root / "old_metrics"
            labeling_dir = source_pipeline / "context_labelings" / "context_000"
            for directory in (source_pipeline, source_query, source_metrics, labeling_dir):
                directory.mkdir(parents=True, exist_ok=True)

            responses = source_query / "responses.jsonl"
            exposed = source_query / "exposed.jsonl"
            labeling = labeling_dir / "context_labeling_cimemories.json"
            metrics = source_metrics / "privacy_metrics_cimemories.json"
            responses.write_text("{}\n", encoding="utf-8")
            exposed.write_text("{}\n", encoding="utf-8")
            labeling.write_text("{}", encoding="utf-8")
            metrics.write_text("{}", encoding="utf-8")
            summary = {
                "artifacts": {
                    "query_output_dir": str(source_query),
                    "responses_jsonl": str(responses),
                    "exposed_attributes_jsonl": str(exposed),
                },
                "contexts": [{
                    "context_idx": 0,
                    "context_labeling_cimemories_json": str(labeling),
                    "privacy_metrics_cimemories_json": str(metrics),
                }],
            }
            (source_pipeline / "privacy_pipeline_cimemories.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            destination = root / "dataset_run" / "personas" / "persona_000"

            localized = _localize_privacy_pipeline(source_pipeline, destination, copy_primary=False)

            localized_labeling = localized / "context_labelings/context_000" / labeling.name
            self.assertTrue(localized_labeling.is_file())
            localized_summary = json.loads(
                (localized / "privacy_pipeline_cimemories.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                localized_summary["contexts"][0]["context_labeling_cimemories_json"],
                str(localized_labeling),
            )


if __name__ == "__main__":
    unittest.main()
