from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from letta_research_chat.cli import (
    export_privacy_pipeline_cimemories_efficiency_latex,
    export_privacy_pipeline_cimemories_latex,
)


def _write_pipeline(
    root: Path,
    name: str,
    *,
    memory_mode: int,
    completion_counts: tuple[float, float],
    private_counts: tuple[float, float],
    ambiguous_counts: tuple[float, float],
) -> Path:
    pipeline_dir = root / name
    pipeline_dir.mkdir()
    contexts = []
    recipients = ("Bank & Lender", "Doctor_Office")
    for context_idx, recipient in enumerate(recipients):
        metrics_dir = root / f"{name}-metrics-{context_idx}"
        metrics_dir.mkdir()
        metrics_path = metrics_dir / "privacy_metrics_cimemories.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "context_idx": context_idx,
                    "recipient": recipient,
                    "metrics": {
                        "repeat_count": 10,
                        "necessary_total": 5,
                        "average_successfully_necessary_count": completion_counts[context_idx],
                        "inappropriate_total": 20,
                        "average_leaked_inappropriate_count": private_counts[context_idx],
                        "ambiguous_total": 25,
                        "average_exposed_ambiguous_count": ambiguous_counts[context_idx],
                    },
                }
            ),
            encoding="utf-8",
        )
        contexts.append(
            {
                "context_idx": context_idx,
                "recipient": recipient,
                "privacy_metrics_cimemories_json": str(metrics_path),
            }
        )
    responses_path = pipeline_dir / "responses.jsonl"
    response_records = []
    for context_idx, recipient in enumerate(recipients):
        for repeat_idx in range(10):
            response_records.append(
                {
                    "context_idx": context_idx,
                    "repeat_idx": repeat_idx,
                    "recipient": recipient,
                    "efficiency": {
                        "timings_ms": {
                            "generation": 90.0 + 100.0 * context_idx,
                            "online_observed": 100.0 + 100.0 * context_idx,
                        },
                        "tokens": {
                            "generation_exact": {
                                "exact": True,
                                "total_tokens": 100 + 100 * context_idx,
                            }
                        },
                    },
                }
            )
    responses_path.write_text(
        "\n".join(json.dumps(record) for record in response_records) + "\n",
        encoding="utf-8",
    )
    (pipeline_dir / "privacy_pipeline_cimemories.json").write_text(
        json.dumps(
            {
                "memory_mode": memory_mode,
                "context_count": 2,
                "scenario_repeats": 10,
                "contexts": contexts,
                "artifacts": {"responses_jsonl": str(responses_path)},
                "efficiency": {
                    "one_time": {"memory_initialization_ms": 3000.0},
                    "timings_ms": {
                        "generation": {"mean": 140.0},
                        "memory_preparation": {"mean": 10.0},
                        "online_deployed_estimate": {"mean": 160.0, "p95": 210.0},
                        "experiment_wall": 5000.0,
                    },
                    "tokens": {"deployed_exact_tokens_per_query": 200.0},
                    "context_preparation": {
                        "0": {
                            "duration_ms": 10.0,
                            "exact_model_tokens_expected": True,
                            "exact_model_tokens": {"exact": True, "total_tokens": 50},
                        },
                        "1": {
                            "duration_ms": 10.0,
                            "exact_model_tokens_expected": True,
                            "exact_model_tokens": {"exact": True, "total_tokens": 50},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return pipeline_dir


class PrivacyLatexTests(unittest.TestCase):
    def test_single_run_writes_recipient_and_aggregate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = _write_pipeline(
                root,
                "pipeline",
                memory_mode=1,
                completion_counts=(4.0, 3.0),
                private_counts=(0.0, 1.0),
                ambiguous_counts=(2.0, 3.0),
            )

            output = export_privacy_pipeline_cimemories_latex([str(pipeline)])
            latex = output.read_text(encoding="utf-8")

            self.assertIn("Bank \\& Lender & 80.0 & 0.00 & 8.00", latex)
            self.assertIn("Doctor\\_Office & 60.0 & 5.00 & 12.00", latex)
            self.assertIn("Overall & 70.0 & 2.50 & 10.00", latex)

    def test_multiple_runs_write_recipient_rows_and_grouped_architectures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archival = _write_pipeline(
                root,
                "archival",
                memory_mode=1,
                completion_counts=(4.0, 4.0),
                private_counts=(0.0, 0.0),
                ambiguous_counts=(1.0, 1.0),
            )
            graph = _write_pipeline(
                root,
                "graph",
                memory_mode=6,
                completion_counts=(5.0, 5.0),
                private_counts=(0.5, 0.5),
                ambiguous_counts=(2.0, 3.0),
            )

            output = export_privacy_pipeline_cimemories_latex([str(archival), str(graph)])
            latex = output.read_text(encoding="utf-8")

            self.assertIn(r"\multicolumn{3}{c}{Archival search}", latex)
            self.assertIn(r"\multicolumn{3}{c}{Graphiti + reranking}", latex)
            self.assertIn(
                r"Bank \& Lender & 80.0 & \textbf{0.00} & \textbf{4.00} & "
                r"\textbf{100.0} & 2.50 & 8.00",
                latex,
            )
            self.assertIn(
                r"Doctor\_Office & 80.0 & \textbf{0.00} & \textbf{4.00} & "
                r"\textbf{100.0} & 2.50 & 12.00",
                latex,
            )
            self.assertIn(r"\specialrule{1.1pt}{0.6ex}{0.4ex}", latex)
            self.assertIn(
                r"\textbf{Overall} & 80.0 & \textbf{0.00} & \textbf{4.00} & "
                r"\textbf{100.0} & 2.50 & 10.00",
                latex,
            )

    def test_incomplete_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = _write_pipeline(
                root,
                "pipeline",
                memory_mode=15,
                completion_counts=(1.0, 1.0),
                private_counts=(0.0, 0.0),
                ambiguous_counts=(0.0, 0.0),
            )
            summary_path = pipeline / "privacy_pipeline_cimemories.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["scenario_repeats"] = 20
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "incomplete"):
                export_privacy_pipeline_cimemories_latex([str(pipeline)])

    def test_compact_efficiency_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline = _write_pipeline(
                root,
                "pipeline",
                memory_mode=1,
                completion_counts=(4.0, 3.0),
                private_counts=(0.0, 0.0),
                ambiguous_counts=(1.0, 1.0),
            )

            output = export_privacy_pipeline_cimemories_efficiency_latex(
                [str(pipeline)], "compact"
            )
            latex = output.read_text(encoding="utf-8")

            self.assertIn("Tokens/query", latex)
            self.assertIn(
                r"Archival search & \textbf{200} & n/a & n/a & \textbf{0.14} & \textbf{0.01} & "
                r"\textbf{0.16} & \textbf{0.21} & \textbf{3.00} & \textbf{0.08}",
                latex,
            )

    def test_detailed_efficiency_table_has_recipients_and_overall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archival = _write_pipeline(
                root,
                "archival",
                memory_mode=1,
                completion_counts=(4.0, 3.0),
                private_counts=(0.0, 0.0),
                ambiguous_counts=(1.0, 1.0),
            )
            graph = _write_pipeline(
                root,
                "graph",
                memory_mode=6,
                completion_counts=(5.0, 5.0),
                private_counts=(0.0, 0.0),
                ambiguous_counts=(2.0, 2.0),
            )

            output = export_privacy_pipeline_cimemories_efficiency_latex(
                [str(archival), str(graph)], "detailed"
            )
            latex = output.read_text(encoding="utf-8")

            self.assertIn(
                r"Bank \& Lender & \textbf{150} & \textbf{0.11} & \textbf{0.11} & "
                r"\textbf{150} & \textbf{0.11} & \textbf{0.11}",
                latex,
            )
            self.assertIn(
                r"Doctor\_Office & \textbf{250} & \textbf{0.21} & \textbf{0.21} & "
                r"\textbf{250} & \textbf{0.21} & \textbf{0.21}",
                latex,
            )
            self.assertIn(r"\specialrule{1.1pt}{0.6ex}{0.4ex}", latex)
            self.assertIn(
                r"\textbf{Overall} & \textbf{200} & \textbf{0.16} & \textbf{0.21} & "
                r"\textbf{200} & \textbf{0.16} & \textbf{0.21}",
                latex,
            )


if __name__ == "__main__":
    unittest.main()
