from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from letta_research_chat.ground_truth_artifacts import (
    _comparison_rows,
    _exposure_metrics,
    load_ground_truth_file,
    memory_row_transform,
)


class GroundTruthArtifactsTest(unittest.TestCase):
    def _labels(self, root: Path) -> Path:
        path = root / "labels.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "bio": {"name": "Test Person"},
                        "information_attributes": {
                            "a": {"memory_statement": "Necessary fact."},
                            "b": {"memory_statement": "Private fact."},
                            "c": {"memory_statement": "Mixed fact."},
                        },
                        "contexts": [
                            {
                                "recipient": "Recipient",
                                "task": "Task",
                                "labels_combined": {
                                    "Necessary fact.": {"necessary": 1.0, "private": 0.0, "unlabeled": 0.0},
                                    "Private fact.": {"necessary": 0.0, "private": 1.0, "unlabeled": 0.0},
                                    "Mixed fact.": {"necessary": 0.5, "private": 0.5, "unlabeled": 0.0},
                                },
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        return path

    def test_loader_uses_generic_combined_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = load_ground_truth_file(self._labels(Path(tmp)), name="alternate")
        context = labels.contexts[("Test Person", 0)]
        self.assertEqual(context.categories["necessary"], {"Necessary fact."})
        self.assertEqual(context.categories["private"], {"Private fact."})
        self.assertEqual(context.categories["ambiguous"], {"Mixed fact."})
        self.assertEqual(labels.name, "alternate")

    def test_exposure_rescore_preserves_total_and_changes_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = load_ground_truth_file(self._labels(Path(tmp)))
        context = labels.contexts[("Test Person", 0)]
        attributes = list(context.attributes)
        records = [
            {"attributes": attributes, "exposed_attributes": {"Necessary fact.": "evidence"}},
            {
                "attributes": attributes,
                "exposed_attributes": {"Necessary fact.": "evidence", "Mixed fact.": "evidence"},
            },
        ]
        metrics = _exposure_metrics(records, context)
        self.assertEqual(metrics["necessary_recall"], 1.0)
        self.assertEqual(metrics["inappropriate_leak_rate"], 0.0)
        self.assertEqual(metrics["ambiguous_exposure_rate"], 0.5)
        self.assertEqual(metrics["average_exposed_total"], 1.5)

    def test_memory_rescore_reuses_saved_represented_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = load_ground_truth_file(self._labels(root))
            context_path = root / "context.json"
            context_path.write_text(
                json.dumps(
                    {
                        "attributes": list(labels.contexts[("Test Person", 0)].attributes),
                        "represented": {
                            "pre_rerank_indices": [0, 1, 2],
                            "post_rerank_indices": [0, 2],
                        },
                    }
                ),
                encoding="utf-8",
            )
            source = {
                "persona_idx": 0,
                "persona_name": "Test Person",
                "context_idx": 0,
                "recipient": "Recipient",
                "task": "Task",
                "source_context": str(context_path),
            }
            result = memory_row_transform(labels)([source])[0]
        self.assertEqual(result["pre_necessary_availability"], 1.0)
        self.assertEqual(result["post_private_availability"], 0.0)
        self.assertEqual(result["ambiguous_retention"], 1.0)

    def test_comparison_marks_direction_flips(self) -> None:
        base = [{
            "model": "m", "architecture": "list", "persona_idx": 0,
            "context_idx": 0, "delta_necessary_recall": -0.2,
            "delta_inappropriate_leak_rate": None,
            "delta_ambiguous_exposure_rate": 0.1,
            "delta_average_exposed_total": 2.0,
        }]
        alternate = [{**base[0], "delta_necessary_recall": 0.1}]
        rows = _comparison_rows(base, alternate)
        necessary = next(row for row in rows if row["metric"] == "necessary_recall")
        self.assertFalse(necessary["direction_stable"])
        exposed = next(row for row in rows if row["metric"] == "average_exposed_total")
        self.assertTrue(exposed["direction_stable"])


if __name__ == "__main__":
    unittest.main()
