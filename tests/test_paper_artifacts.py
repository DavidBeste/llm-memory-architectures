import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from letta_research_chat.paper_artifacts import (  # noqa: E402
    discover_privacy_dataset_runs,
    export_privacy_paper_artifacts,
)


class PaperArtifactTests(unittest.TestCase):
    def _run(
        self,
        root: Path,
        mode: int,
        name: str,
        created: str,
        offset: float,
        discarded_contexts: set[int] | None = None,
    ) -> Path:
        discarded_contexts = discarded_contexts or set()
        run = root / f"privacy-pipeline-cimemories-dataset_data_mem{mode}_{created}"
        pipeline = run / "personas" / "persona_000" / "pipeline"
        metrics_root = run / "personas" / "persona_000" / "privacy_metrics"
        pipeline.mkdir(parents=True)
        contexts = []
        for context_idx in range(2):
            metrics_dir = metrics_root / f"context_{context_idx:03d}"
            metrics_dir.mkdir(parents=True)
            metrics = metrics_dir / "privacy_metrics_cimemories.json"
            metrics.write_text(
                json.dumps(
                    {
                        "persona_idx": 0,
                        "persona_name": "Test Person",
                        "context_idx": context_idx,
                        "recipient": f"Recipient {context_idx}",
                        "task": f"Task {context_idx}",
                        "context_discarded": context_idx in discarded_contexts,
                        "metrics": {
                            "necessary_recall": 0.5 + offset + context_idx * 0.1,
                            "inappropriate_leak_rate": 0.02 + offset / 10,
                            "ambiguous_exposure_rate": 0.1 + offset / 10,
                        },
                    }
                ),
                encoding="utf-8",
            )
            contexts.append(
                {
                    "context_idx": context_idx,
                    "privacy_metrics_cimemories_json": str(metrics),
                }
            )
        (pipeline / "manifest.json").write_text(
            json.dumps(
                {
                    "persona_idx": 0,
                    "persona_name": "Test Person",
                    "context_count": 2,
                    "scenario_repeats": 10,
                    "contexts": contexts,
                    "efficiency": {
                        "counts": {"contexts": 2, "responses": 20},
                        "one_time": {"memory_initialization_ms": 1000},
                        "timings_ms": {
                            "generation": {"count": 20, "mean": 200},
                            "memory_preparation": {"count": 2, "mean": 50},
                            "online_deployed_estimate": {"count": 20, "mean": 250, "p95": 300},
                            "experiment_wall": 5000,
                        },
                        "tokens": {
                            "deployed_exact_tokens_per_query": 100,
                            "generation_exact_coverage": 1,
                            "memory_preparation_exact_coverage": 1,
                        },
                    },
                    "evaluation_overhead": {
                        "exposure_judging_ms": 500,
                        "context_labeling_and_privacy_metrics_ms": 100,
                    },
                    "pipeline_usage_ledger": {
                        "logical_exact_total": {"total_tokens": 2500},
                        "components": [
                            {"component": "response_generation", "availability": "exact"},
                            {"component": "backend_internal", "availability": "unavailable"},
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "command": "run_privacy_pipeline_cimemories_dataset",
                    "status": "complete",
                    "created_at": created,
                    "dataset_name": "data.json",
                    "context_labels_file": "/safe/labels_qwen.json",
                    "memory_mode": mode,
                    "memory_mode_name": name,
                    "persona_count": 1,
                    "personas": [{"persona_idx": 0, "local_pipeline": str(pipeline)}],
                }
            ),
            encoding="utf-8",
        )
        return run

    def test_exports_complete_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = self._run(root, 3, "list-rerank", "20260101_000000", 0.0)
            right = self._run(root, 6, "graph-rerank", "20260102_000000", 0.05)
            output = export_privacy_paper_artifacts(
                [str(left), str(right)], output_dir=str(root / "paper"), baseline_name="list-rerank"
            )
            expected = {
                "aggregate_metrics.csv",
                "paired_comparisons.csv",
                "persona_metrics.csv",
                "context_metrics.csv",
                "context_deltas.csv",
                "figures",
                "includes",
                "efficiency_summary.csv",
                "efficiency_by_persona.csv",
                "manifest.json",
                "README.md",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertEqual(
                {path.name for path in (output / "figures").iterdir()},
                {
                    "context_difference_heatmap.svg",
                    "persona_completion.svg",
                    "privacy_utility.svg",
                    "efficiency_tradeoff.svg",
                },
            )
            self.assertEqual(
                {path.name for path in (output / "includes").iterdir()},
                {
                    "aggregate_metrics.tex",
                    "paired_comparisons.tex",
                    "efficiency_summary.tex",
                    "context_difference_heatmap.tex",
                    "persona_completion.tex",
                    "privacy_utility.tex",
                    "efficiency_tradeoff.tex",
                    "overleaf_preamble.tex",
                    "overleaf_figures.tex",
                },
            )
            self.assertIn("graph-rerank", (output / "aggregate_metrics.csv").read_text())
            self.assertIn("Green is better", (output / "figures/context_difference_heatmap.svg").read_text())
            heatmap_svg = (output / "figures/context_difference_heatmap.svg").read_text()
            self.assertIn(">Graph</text>", heatmap_svg)
            self.assertNotIn("graph-rerank − baseline", heatmap_svg)
            tradeoff_svg = (output / "figures/privacy_utility.svg").read_text()
            efficiency_svg = (output / "figures/efficiency_tradeoff.svg").read_text()
            for rendered in (tradeoff_svg, efficiency_svg):
                self.assertIn(">List</text>", rendered)
                self.assertIn(">Graph</text>", rendered)
                self.assertNotIn(">graph-rerank</text>", rendered)
            efficiency = (output / "efficiency_summary.csv").read_text()
            self.assertIn("deployed_exact_tokens_per_query", efficiency)
            self.assertIn("backend_internal", efficiency)
            heatmap_tex = (output / "includes/context_difference_heatmap.tex").read_text()
            self.assertIn(r"\includesvg", heatmap_tex)
            self.assertIn(r"{figures/context_difference_heatmap.svg}", heatmap_tex)
            self.assertIn(r"\label{fig:privacy-context-differences}", heatmap_tex)
            self.assertIn(r"\usepackage{svg}", (output / "includes/overleaf_preamble.tex").read_text())
            self.assertIn(
                r"\input{includes/context_difference_heatmap.tex}",
                (output / "includes/overleaf_figures.tex").read_text(),
            )

    def test_complete_case_export_excludes_discarded_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = self._run(
                root, 3, "list-rerank", "20260101_000000", 0.0, discarded_contexts={1}
            )
            right = self._run(
                root, 6, "graph-rerank", "20260102_000000", 0.05, discarded_contexts={1}
            )
            output = export_privacy_paper_artifacts(
                [str(left), str(right)],
                output_dir=str(root / "paper"),
                baseline_name="list-rerank",
                exclude_discarded_contexts=True,
            )
            aggregate = (output / "aggregate_metrics.csv").read_text()
            self.assertIn("list-rerank,3,1,50.0", aggregate)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["evaluation_cells"],
                {
                    "policy": "exclude-context-discarded",
                    "original": 2,
                    "excluded": 1,
                    "included": 1,
                },
            )
            self.assertIn("`1` included, `1` excluded", (output / "README.md").read_text())

    def test_discovery_selects_newest_complete_run_per_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self._run(root, 3, "list-rerank", "20260101_000000", 0.0)
            newest = self._run(root, 3, "list-rerank", "20260103_000000", 0.0)
            graph = self._run(root, 6, "graph-rerank", "20260102_000000", 0.0)
            selected = discover_privacy_dataset_runs(
                str(root), dataset_filter="data", labels_filter="qwen", modes=[3, 6]
            )
            self.assertEqual(selected, [str(newest), str(graph)])
            self.assertNotIn(str(old), selected)


if __name__ == "__main__":
    unittest.main()
