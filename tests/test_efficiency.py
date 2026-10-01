from __future__ import annotations

import unittest

from letta_research_chat.efficiency import (
    aggregate_usage_components,
    estimate_visible_tokens,
    extract_token_usage,
    summarize_query_efficiency,
    summarize_samples,
    token_usage_delta,
    usage_component,
)


class EfficiencyTests(unittest.TestCase):
    def test_visible_token_estimator_is_deterministic(self) -> None:
        self.assertEqual(estimate_visible_tokens("Hello, world!"), 4)
        self.assertEqual(estimate_visible_tokens("Café costs €5."), 5)

    def test_extract_token_usage_prefers_top_level_aggregate(self) -> None:
        payload = {
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "messages": [
                {"usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
            ],
        }
        self.assertEqual(
            extract_token_usage(payload),
            {
                "exact": True,
                "source": "response.usage",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cached_input_tokens": None,
                "reasoning_output_tokens": None,
            },
        )

    def test_nearest_rank_p95(self) -> None:
        summary = summarize_samples(range(1, 101))
        self.assertEqual(summary["mean"], 50.5)
        self.assertEqual(summary["median"], 50.5)
        self.assertEqual(summary["p95"], 95.0)

    def test_usage_ledger_never_counts_unavailable_as_zero(self) -> None:
        ledger = aggregate_usage_components(
            [
                usage_component(
                    "generation",
                    stage="online",
                    modality="generation",
                    operation="respond",
                    usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    expected_calls=1,
                ),
                usage_component(
                    "embeddings",
                    stage="initialization",
                    modality="embedding",
                    operation="embed",
                    expected_calls=3,
                    reason="backend omitted usage",
                ),
            ]
        )
        self.assertEqual(ledger["additive_exact_total"]["total_tokens"], 15)
        self.assertFalse(ledger["complete"])
        self.assertEqual(ledger["unavailable_components"], ["embeddings"])

    def test_cumulative_usage_delta(self) -> None:
        delta = token_usage_delta(
            {"usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}},
            {"usage": {"input_tokens": 160, "output_tokens": 35, "total_tokens": 195}},
            source="test.delta",
        )
        self.assertEqual(delta["input_tokens"], 60)
        self.assertEqual(delta["output_tokens"], 15)
        self.assertEqual(delta["total_tokens"], 75)

    def test_summary_separates_observed_and_deployed_cost(self) -> None:
        usage = {
            "exact": True,
            "input_tokens": 40,
            "output_tokens": 10,
            "total_tokens": 50,
        }
        records = []
        for context_idx in (0, 1):
            for _ in range(2):
                records.append(
                    {
                        "context_idx": context_idx,
                        "efficiency": {
                            "timings_ms": {"generation": 20.0, "online_observed": 25.0},
                            "tokens": {
                                "generation_exact": usage,
                                "visible_prompt_estimate": 30,
                                "visible_response_estimate": 10,
                            },
                        },
                    }
                )
        context_preparation = {
            0: {
                "duration_ms": 100.0,
                "exact_model_tokens": {**usage, "total_tokens": 100},
                "exact_model_tokens_expected": True,
            },
            1: {
                "duration_ms": 200.0,
                "exact_model_tokens": {**usage, "total_tokens": 100},
                "exact_model_tokens_expected": True,
            },
        }
        summary = summarize_query_efficiency(
            records,
            context_preparation,
            initialization_ms=300.0,
            agent_creation_ms=50.0,
            generation_batch_ms=80.0,
            experiment_elapsed_ms=500.0,
            repeats_per_context=2,
            concurrency=4,
        )
        self.assertEqual(summary["timings_ms"]["online_observed"]["mean"], 25.0)
        self.assertEqual(summary["timings_ms"]["online_deployed_estimate"]["mean"], 175.0)
        self.assertEqual(summary["tokens"]["observed_exact_total"], 400)
        self.assertEqual(summary["tokens"]["deployed_exact_tokens_per_query"], 150.0)
        self.assertEqual(summary["tokens"]["generation_exact_coverage"], 1.0)
        self.assertEqual(summary["tokens"]["memory_preparation_exact_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
