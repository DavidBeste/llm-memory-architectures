from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from letta_research_chat.integrated_paper_artifacts import (
    _architecture_efficiency,
    _join_response_rows,
    _memory_bootstrap_estimate_count,
    _memory_conditions,
    _memory_interactions,
    _memory_paired_contrasts,
    _summary_matches,
)


class IntegratedPaperArtifactsTest(unittest.TestCase):
    def test_architecture_efficiency_macro_averages_pipeline_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for index, (model, preparation, generation, tokens) in enumerate(
                (("model-a", 1000, 3000, 4000), ("model-b", 3000, 5000, 6000))
            ):
                manifest = root / f"manifest-{index}.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "efficiency": {
                                "one_time": {"memory_initialization_ms": 120000 + index * 60000},
                                "timings_ms": {
                                    "memory_preparation": {"mean": preparation},
                                    "generation": {"mean": generation},
                                    "online_deployed_estimate": {
                                        "mean": preparation + generation,
                                        "p95": preparation + generation + 1000,
                                    },
                                },
                                "tokens": {
                                    "deployed_exact_tokens_per_query": tokens,
                                    "generation_exact_coverage": 1,
                                    "memory_preparation_exact_coverage": 1,
                                },
                                "usage_ledger": {"complete": index == 0},
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                runs.append(
                    {
                        "model": model,
                        "architecture": "list",
                        "persona_idx": index,
                        "pipeline_manifest": str(manifest),
                    }
                )

            summary, details = _architecture_efficiency(runs)

        self.assertEqual(len(details), 2)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["pipeline_cells"], 2)
        self.assertEqual(row["model_count"], 2)
        self.assertAlmostEqual(row["memory_preparation_seconds"], 2.0)
        self.assertAlmostEqual(row["generation_seconds"], 4.0)
        self.assertAlmostEqual(row["deployed_seconds"], 6.0)
        self.assertAlmostEqual(row["reported_exact_tokens_per_query"], 5000)
        self.assertAlmostEqual(row["initialization_seconds"], 150)
        self.assertEqual(row["available_token_cells"], 2)
        self.assertEqual(row["reported_token_cells"], 2)
        self.assertEqual(row["complete_usage_ledger_cells"], 1)

    def test_summary_matching_separates_exact_and_semantic_methods(self) -> None:
        exact = {"context_count": 49, "strategy": "exact-match"}
        semantic = {
            "context_count": 49,
            "strategy": "monolithic",
            "judge_model": "gpt-6-sol",
            "pilot_contexts": None,
        }
        pilot = {**semantic, "pilot_contexts": 10}
        self.assertTrue(
            _summary_matches(exact, architecture="list", judge_model="gpt-6-sol")
        )
        self.assertFalse(
            _summary_matches(exact, architecture="graph", judge_model="gpt-6-sol")
        )
        self.assertTrue(
            _summary_matches(semantic, architecture="profile", judge_model="gpt-6-sol")
        )
        self.assertFalse(
            _summary_matches(pilot, architecture="profile", judge_model="gpt-6-sol")
        )

    def test_memory_conditions_preserve_pre_post_and_density(self) -> None:
        rows = []
        for context, pre, post in ((0, 0.8, 0.6), (1, 1.0, 0.9)):
            row = {
                "model": "model-a",
                "model_label": "model-a",
                "architecture": "graph",
                "persona_idx": 0,
                "context_idx": context,
                "context_discarded": False,
                "pre_fact_count": 50,
                "post_fact_count": 10,
                "pre_attribute_count": 60,
                "post_attribute_count": 20,
                "matching_method": "monolithic",
                "judge_model": "judge-a",
            }
            for category in ("necessary", "private", "ambiguous"):
                row[f"pre_{category}_availability"] = pre
                row[f"post_{category}_availability"] = post
                row[f"delta_{category}_availability"] = post - pre
                row[f"{category}_retention"] = post / pre
            rows.append(row)
        result = _memory_conditions(rows)
        necessary = next(
            row
            for row in result
            if row["scope"] == "all" and row["category"] == "necessary"
        )
        self.assertAlmostEqual(necessary["pre_availability"], 0.9)
        self.assertAlmostEqual(necessary["post_availability"], 0.75)
        self.assertEqual(necessary["mean_pre_facts"], 50)
        self.assertEqual(necessary["mean_post_attributes"], 20)

    def test_cross_stage_join_uses_exact_model_architecture_persona_context(self) -> None:
        memory = {
            "model": "model-a",
            "model_label": "model-a",
            "architecture": "list",
            "persona_idx": 0,
            "context_idx": 3,
        }
        for category in ("necessary", "private", "ambiguous"):
            memory[f"pre_{category}_availability"] = 0.8
            memory[f"post_{category}_availability"] = 0.4
            memory[f"delta_{category}_availability"] = -0.4
        response = {
            "model": "model-a",
            "architecture": "list",
            "persona_idx": "0",
            "context_idx": "3",
            "pre_necessary_recall": "0.5",
            "post_necessary_recall": "0.25",
            "pre_inappropriate_leak_rate": "0.2",
            "post_inappropriate_leak_rate": "0.1",
            "pre_ambiguous_exposure_rate": "0.3",
            "post_ambiguous_exposure_rate": "0.15",
        }
        joined, alignment = _join_response_rows([memory], [response])
        self.assertAlmostEqual(joined[0]["post_necessary_response_minus_memory"], -0.15)
        self.assertAlmostEqual(joined[0]["delta_private_response"], -0.1)
        post_private = next(
            row
            for row in alignment
            if row["stage"] == "post" and row["category"] == "private"
        )
        self.assertEqual(post_private["context_pairs"], 1)
        self.assertAlmostEqual(post_private["memory_availability"], 0.4)
        self.assertAlmostEqual(post_private["response_rate"], 0.1)

    def test_memory_contrasts_use_right_minus_left_and_paired_contexts(self) -> None:
        rows = []
        for context_idx in (0, 1):
            for architecture, pre, post in (
                ("list", 0.8, 0.6),
                ("graph", 0.5, 0.4),
            ):
                row = {
                    "model": "model-a",
                    "architecture": architecture,
                    "persona_idx": 0,
                    "context_idx": context_idx,
                }
                for category in ("necessary", "private", "ambiguous"):
                    row[f"pre_{category}_availability"] = pre
                    row[f"post_{category}_availability"] = post
                    row[f"delta_{category}_availability"] = post - pre
                rows.append(row)

        contrasts = _memory_paired_contrasts(
            rows, dimension="architecture", iterations=20
        )
        pre_necessary = next(
            row
            for row in contrasts
            if row["stage"] == "pre" and row["category"] == "necessary"
        )
        self.assertEqual(pre_necessary["left_architecture"], "list")
        self.assertEqual(pre_necessary["right_architecture"], "graph")
        self.assertEqual(pre_necessary["context_pairs"], 2)
        self.assertAlmostEqual(pre_necessary["estimate"], -0.3)

        interactions = _memory_interactions(
            rows, dimension="architecture", iterations=20
        )
        necessary = next(
            row for row in interactions if row["category"] == "necessary"
        )
        # Graph loses 0.1 under reranking while list loses 0.2, so the
        # graph-minus-list difference in reranking effects is +0.1.
        self.assertEqual(necessary["context_pairs"], 2)
        self.assertAlmostEqual(necessary["estimate"], 0.1)
        self.assertEqual(_memory_bootstrap_estimate_count(rows), 15)


if __name__ == "__main__":
    unittest.main()
