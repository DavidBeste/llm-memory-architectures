from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from letta_research_chat.comparative_report import METRICS
from letta_research_chat.study_artifacts import (
    _bootstrap_estimate_count,
    _coverage,
    _interaction_contrasts,
    _limited_post_metrics,
    _paired_contrasts,
    _reranking_effects,
)


def _row(
    model: str,
    architecture: str,
    persona: int,
    context: int,
    pre: float,
    post: float,
) -> dict:
    row = {
        "model": model,
        "model_label": model,
        "architecture": architecture,
        "persona_idx": persona,
        "persona_name": f"Persona {persona}",
        "context_idx": context,
        "context_discarded": False,
    }
    for metric in METRICS:
        row[f"pre_{metric}"] = pre
        row[f"post_{metric}"] = post
        row[f"delta_{metric}"] = post - pre
    return row


class CimemoriesStudyArtifactsTest(unittest.TestCase):
    def test_limited_post_metrics_selects_lowest_repeat_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "privacy.json"
            metrics_path.write_text(
                json.dumps({
                    "necessary_attributes": ["needed"],
                    "inappropriate_attributes": ["private"],
                    "ambiguous_attributes": ["mixed"],
                    "repeated_exposed_records": [
                        {"repeat_idx": 1, "exposed_attributes": {"private": "later"}},
                        {"repeat_idx": 0, "exposed_attributes": {"needed": "first"}},
                    ],
                }),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest = {
                "contexts": [{
                    "context_idx": 0,
                    "privacy_metrics_cimemories_json": str(metrics_path),
                }]
            }
            result = _limited_post_metrics(manifest_path, manifest, limit=1)[0]
        self.assertEqual(result["necessary_recall"], 1.0)
        self.assertEqual(result["inappropriate_leak_rate"], 0.0)
        self.assertEqual(result["ambiguous_exposure_rate"], 0.0)
        self.assertEqual(result["average_exposed_total"], 1.0)

    def test_architecture_contrast_uses_only_exact_shared_cells(self) -> None:
        rows = [
            _row("model-a", "list", 0, 0, 0.1, 0.2),
            _row("model-a", "list", 1, 0, 0.1, 0.2),
            _row("model-a", "graph", 0, 0, 0.3, 0.5),
        ]
        contrasts = _paired_contrasts(rows, dimension="architecture", iterations=100)
        result = next(
            row for row in contrasts
            if row["metric"] == "necessary_recall" and row["stage"] == "post"
        )
        self.assertEqual(result["persona_count"], 1)
        self.assertEqual(result["context_pairs"], 1)
        self.assertAlmostEqual(result["estimate"], -0.3)

    def test_model_interaction_is_difference_between_reranking_effects(self) -> None:
        rows = [
            _row("model-a", "list", 0, 0, 0.1, 0.2),
            _row("model-b", "list", 0, 0, 0.1, 0.4),
        ]
        interactions = _interaction_contrasts(rows, dimension="model", iterations=100)
        result = next(row for row in interactions if row["metric"] == "necessary_recall")
        self.assertEqual(result["left_model"], "model-a")
        self.assertEqual(result["right_model"], "model-b")
        self.assertAlmostEqual(result["estimate"], 0.2)
        self.assertTrue(result["ci_excludes_zero"])

    def test_reranking_effect_retains_persona_and_context_counts(self) -> None:
        rows = [
            _row("model-a", "profile", persona, context, 0.4, 0.3)
            for persona in (0, 1) for context in (0, 1)
        ]
        effects = _reranking_effects(rows, iterations=100)
        result = next(row for row in effects if row["metric"] == "average_exposed_total")
        self.assertEqual(result["persona_count"], 2)
        self.assertEqual(result["context_pairs"], 4)
        self.assertAlmostEqual(result["estimate"], -0.1)

    def test_coverage_reports_missing_personas(self) -> None:
        inventory = [
            {"model": "model-a", "architecture": "list", "persona_idx": 0},
            {"model": "model-a", "architecture": "list", "persona_idx": 2},
        ]
        result = _coverage(inventory, expected_personas=3)[0]
        self.assertEqual(result["complete_personas"], 2)
        self.assertEqual(result["missing_persona_indices"], "1")
        self.assertFalse(result["complete"])

    def test_bootstrap_progress_count_matches_all_factorial_estimates(self) -> None:
        rows = [
            _row("model-a", architecture, 0, 0, 0.1, 0.2)
            for architecture in ("list", "graph")
        ]
        metric_count = len(METRICS)
        # Two conditions each have pre, post, and reranking estimates. The
        # architecture pair adds pre, post, and interaction estimates.
        self.assertEqual(
            _bootstrap_estimate_count(rows, metric_count),
            (2 * 3 * metric_count) + (1 * 3 * metric_count),
        )


if __name__ == "__main__":
    unittest.main()
