from __future__ import annotations

import unittest

from letta_research_chat.comparative_report import (
    METRICS,
    _cohort_robustness,
    _matched_rows,
    _summary_rows,
)


def _row(model: str, persona: int, context: int, pre: float, post: float) -> dict:
    row = {
        "model": model,
        "architecture": "list",
        "persona_idx": persona,
        "persona_name": f"Persona {persona}",
        "context_idx": context,
        "recipient": "recipient",
        "task": "task",
    }
    for metric in METRICS:
        row[f"pre_{metric}"] = pre
        row[f"post_{metric}"] = post
        row[f"delta_{metric}"] = post - pre
    return row


class ComparativeReportTest(unittest.TestCase):
    def test_summary_pairs_pre_and_post_with_hierarchical_interval(self) -> None:
        rows = [_row("model-a", persona, context, 0.2, 0.3)
                for persona in (0, 1) for context in (0, 1)]
        summary = _summary_rows(rows, iterations=100)

        necessary = next(row for row in summary if row["metric"] == "necessary_recall")
        self.assertEqual(necessary["persona_count"], 2)
        self.assertEqual(necessary["context_pairs"], 4)
        self.assertAlmostEqual(necessary["mean_delta"], 0.1)
        self.assertAlmostEqual(necessary["ci95_low"], 0.1)
        self.assertAlmostEqual(necessary["ci95_high"], 0.1)

    def test_cross_model_comparison_uses_only_shared_cells(self) -> None:
        rows = [
            _row("model-a", 0, 0, 0.1, 0.2),
            _row("model-a", 1, 0, 0.1, 0.2),
            _row("model-b", 0, 0, 0.3, 0.5),
        ]
        matched = _matched_rows(rows)

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["persona_idx"], 0)
        self.assertAlmostEqual(
            matched[0]["post_right_minus_left_necessary_recall"], 0.3
        )

    def test_cohort_robustness_reports_full_and_common_personas(self) -> None:
        rows = [
            _row("model-a", 0, 0, 0.0, 0.1),
            _row("model-a", 1, 0, 0.0, 0.5),
            _row("model-b", 0, 0, 0.0, 0.2),
        ]
        robustness = _cohort_robustness(rows)
        result = next(
            row for row in robustness
            if row["model"] == "model-a"
            and row["architecture"] == "list"
            and row["metric"] == "necessary_recall"
        )

        self.assertEqual(result["full_personas"], 2)
        self.assertEqual(result["matched_personas"], 1)
        self.assertAlmostEqual(result["full_mean_delta"], 0.3)
        self.assertAlmostEqual(result["matched_mean_delta"], 0.1)
        self.assertAlmostEqual(result["matched_minus_full"], -0.2)


if __name__ == "__main__":
    unittest.main()
