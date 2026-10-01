from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from letta_research_chat.anonymized_artifacts import (
    PathRewriter,
    Sanitizer,
    audit_anonymized_directory,
    export_anonymized_artifacts,
    verify_anonymized_artifact,
)
from letta_research_chat.comparative_report import _resolve


class AnonymizedArtifactsTest(unittest.TestCase):
    def test_artifact_uri_resolves_within_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifact_manifest.json").write_text("{}", encoding="utf-8")
            target = root / "pipelines/example/manifest.json"
            target.parent.mkdir(parents=True)
            target.write_text("{}", encoding="utf-8")
            lockfile = root / "paper_report/integrated_artifact_inputs.json"
            lockfile.parent.mkdir()
            lockfile.write_text("{}", encoding="utf-8")

            self.assertEqual(
                _resolve("artifact://pipelines/example/manifest.json", lockfile),
                target,
            )
            self.assertIsNone(_resolve("artifact://../outside", lockfile))

    def test_sanitizer_removes_operational_metadata_but_preserves_synthetic_data(self) -> None:
        sanitizer = Sanitizer(PathRewriter({}, Path("/work/repository")))
        payload = {
            "persona_idx": 0,
            "persona_name": "Synthetic Person",
            "address": "123 Fictional Road",
            "response_id": "resp_0123456789abcdefghijkl",
            "conversation_id": "conv-11111111-1111-4111-8111-111111111111",
            "input_fingerprint": "abcdef0123456789",
            "credential_source": "OPENAI_API_KEY",
            "I earned a professional credential in 2023.": "Synthetic fact",
            "source": "/home/researcher/project/output.json",
            "nested": {
                "id": "agent-22222222-2222-4222-8222-222222222222",
                "attribute_idx": 17,
            },
        }

        result = sanitizer.sanitize_value(payload)

        self.assertEqual(result["persona_name"], "Synthetic Person")
        self.assertEqual(result["address"], "123 Fictional Road")
        self.assertEqual(result["persona_idx"], 0)
        self.assertEqual(result["nested"]["attribute_idx"], 17)
        self.assertEqual(
            result["I earned a professional credential in 2023."], "Synthetic fact"
        )
        self.assertNotIn("response_id", result)
        self.assertNotIn("conversation_id", result)
        self.assertNotIn("input_fingerprint", result)
        self.assertNotIn("credential_source", result)
        self.assertNotIn("id", result["nested"])
        self.assertEqual(result["source"], "<ABSOLUTE_PATH_REMOVED>")

    def test_export_copies_external_pre_stage_and_passes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "paper-report_20260929_000041"
            report.mkdir()
            persona = root / "dataset-run_20260912_054751" / "personas" / "persona_000"
            pipeline = persona / "pipeline"
            pipeline.mkdir(parents=True)
            query = persona / "query"
            query.mkdir()
            pre = (
                root
                / "dataset-run_20260912_054751"
                / "pre_rerank_response_evaluation"
                / "personas"
                / "persona_000"
            )
            pre.mkdir(parents=True)

            pipeline_manifest = pipeline / "manifest.json"
            pipeline_manifest.write_text(
                json.dumps(
                    {
                        "created_at": "2026-09-12T05:47:51Z",
                        "artifacts": {
                            "query_output_dir": str(query),
                            "api_key_configured": True,
                        },
                        "persona_name": "Synthetic Person",
                    }
                ),
                encoding="utf-8",
            )
            (query / "responses_20260912_060615.jsonl").write_text(
                json.dumps(
                    {
                        "persona_name": "Synthetic Person",
                        "assistant_response": "Lives at 123 Fictional Road.",
                        "response_id": "resp_0123456789abcdefghijkl",
                        "history": [
                            {
                                "content": (
                                    "AGENT_ID: agent-11111111-1111-4111-8111-111111111111\n"
                                    "System prompt last recompiled: 2026-09-12 05:00 UTC"
                                )
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pre_manifest = pre / "manifest.json"
            pre_manifest.write_text(
                json.dumps(
                    {
                        "source_pipeline_manifest": str(pipeline_manifest),
                        "conversation_id": "conv-22222222-2222-4222-8222-222222222222",
                    }
                ),
                encoding="utf-8",
            )
            locked = {
                "selected_runs": [
                    {
                        "model": "model/example",
                        "architecture": "list",
                        "persona_idx": 0,
                        "persona_name": "Synthetic Person",
                        "pipeline_manifest": str(pipeline_manifest),
                        "pre_manifest": str(pre_manifest),
                        "memory_stage_summaries": "",
                        "post_records": 49,
                        "pre_records": 49,
                    }
                ]
            }
            inputs = report / "integrated_artifact_inputs.json"
            inputs.write_text(json.dumps(locked), encoding="utf-8")
            (report / "REPORT.html").write_text(
                f"<p>{pipeline_manifest}</p>", encoding="utf-8"
            )
            output = root / "anonymous"

            result = export_anonymized_artifacts(inputs, output)

            self.assertEqual(result, output.resolve())
            exported_root = output / "pipelines/model-example/list/persona_000"
            exported_response = next((exported_root / "query").glob("responses_run-*.jsonl"))
            self.assertTrue(exported_response.is_file())
            self.assertTrue(
                (
                    exported_root
                    / "pre_rerank_response_evaluation/personas/persona_000/manifest.json"
                ).is_file()
            )
            content = exported_response.read_text(encoding="utf-8")
            self.assertIn("Synthetic Person", content)
            self.assertIn("123 Fictional Road", content)
            self.assertNotIn("resp_", content)
            self.assertNotIn("agent-11111111", content)
            self.assertNotIn(str(root), content)
            exported_readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("analysis and visualization aids", exported_readme)
            self.assertIn("authoritative numerical results", exported_readme)
            self.assertEqual(audit_anonymized_directory(output)["status"], "passed")
            self.assertIn(str(root), pipeline_manifest.read_text(encoding="utf-8"))

    def test_github_scope_keeps_only_context_zero_jsonl_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            report.mkdir()
            pipeline = root / "run"
            query = pipeline / "query"
            pipeline.mkdir()
            query.mkdir()
            (query / "responses.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {"context_idx": index, "repeat_idx": 0, "text": str(index)}
                    )
                    for index in (0, 1)
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = pipeline / "manifest.json"
            manifest.write_text(
                json.dumps({"artifacts": {"query_output_dir": str(query)}}),
                encoding="utf-8",
            )
            inputs = report / "integrated_artifact_inputs.json"
            inputs.write_text(
                json.dumps(
                    {
                        "selected_runs": [
                            {
                                "model": "model-a",
                                "architecture": "list",
                                "persona_idx": 0,
                                "pipeline_manifest": str(manifest),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (report / "REPORT.html").write_text("report", encoding="utf-8")

            output = root / "anonymous"
            export_anonymized_artifacts(inputs, output, scope="github")

            exported = (
                output
                / "pipelines/model-a/list/persona_000/query/responses.jsonl"
            )
            self.assertTrue(
                (output / "pipelines/model-a/list/persona_000/manifest.json").is_file()
            )
            rows = [json.loads(line) for line in exported.read_text().splitlines()]
            self.assertEqual([row["context_idx"] for row in rows], [0])
            artifact_manifest = json.loads(
                (output / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(artifact_manifest["scope"], "github")
            verification = verify_anonymized_artifact(output)
            self.assertEqual(verification["status"], "passed")
            self.assertEqual(verification["pipeline_count"], 1)
            self.assertEqual(verification["sample_row_count"], 1)

            unchecked = output / "unexpected.txt"
            unchecked.write_text("not covered by the release checksums", encoding="utf-8")
            failed = verify_anonymized_artifact(output)
            self.assertEqual(failed["status"], "failed")
            self.assertIn(
                "unchecksummed artifact file: unexpected.txt", failed["issues"]
            )

    def test_export_follows_external_query_and_privacy_metric_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            report.mkdir()
            pipeline = root / "pipeline-run"
            pipeline.mkdir()
            query = root / "query-run_20260925_005036"
            query.mkdir()
            metrics = root / "metrics-run_20260925_005306"
            metrics.mkdir()
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps([{"persona": "Synthetic Person"}]), encoding="utf-8")
            labels = root / "labels.json"
            labels.write_text(json.dumps([{"labels_combined": []}]), encoding="utf-8")
            (query / "responses.jsonl").write_text(
                json.dumps({"assistant_response": "Synthetic response"}) + "\n",
                encoding="utf-8",
            )
            metrics_file = metrics / "privacy_metrics_cimemories.json"
            metrics_file.write_text(
                json.dumps({"context_idx": 0, "necessary_recall": 0.5}),
                encoding="utf-8",
            )
            manifest = pipeline / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset_path": str(dataset),
                        "context_labels_file": str(labels),
                        "artifacts": {"query_output_dir": str(query)},
                        "contexts": [
                            {
                                "context_idx": 0,
                                "privacy_metrics_cimemories_json": str(metrics_file),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            inputs = report / "integrated_artifact_inputs.json"
            inputs.write_text(
                json.dumps(
                    {
                        "selected_runs": [
                            {
                                "model": "model-a",
                                "architecture": "graph",
                                "persona_idx": 0,
                                "pipeline_manifest": str(manifest),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "anonymous"

            export_anonymized_artifacts(inputs, output)

            exported = output / "pipelines/model-a/graph/persona_000"
            self.assertTrue((exported / "query/responses.jsonl").is_file())
            self.assertTrue((output / "inputs/dataset.json").is_file())
            self.assertTrue((output / "inputs/context_labels.json").is_file())
            self.assertTrue(
                (
                    exported
                    / "privacy_metrics/context_000/privacy_metrics_cimemories.json"
                ).is_file()
            )
            self.assertEqual(audit_anonymized_directory(output)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
