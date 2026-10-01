import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from letta_research_chat.cli import (
    find_existing_complete_cimemories_labeling,
    load_embedded_cimemories_labelings,
)


class ContextLabelModelTests(unittest.TestCase):
    def test_embedded_qwen_labels_are_normalized_without_generation(self):
        memories = ["needed", "secret", "uncertain"]
        distributions = {
            "needed": {"necessary": 1.0, "private": 0.0, "unlabeled": 0.0},
            "secret": {"necessary": 0.0, "private": 1.0, "unlabeled": 0.0},
            "uncertain": {"necessary": 0.1, "private": 0.9, "unlabeled": 0.0},
        }
        context = {
            "recipient": "Doctor",
            "task": "Provide care",
            "labels_qwen3.8-27b___the_privacy_fundamentalist__": distributions,
            "labels_qwen3.8-27b___the_pragmatic__": distributions,
            "labels_qwen3.8-27b__the_unconcerned__": distributions,
            "labels_combined": distributions,
        }
        source = [{
            "bio": {"name": "Test User"},
            "information_attributes": {
                str(idx): {"memory_statement": memory}
                for idx, memory in enumerate(memories)
            },
            "contexts": [context],
        }]

        with tempfile.TemporaryDirectory() as temp_dir:
            labels_path = Path(temp_dir) / "labels.json"
            labels_path.write_text(json.dumps(source), encoding="utf-8")
            result = load_embedded_cimemories_labelings(
                str(labels_path),
                dataset_name="dataset.json",
                persona_idx=0,
                persona_name="Test User",
                memories=memories,
                valid_contexts=[(0, {"recipient": "Doctor", "task": "Provide care"})],
            )[0]

        self.assertEqual(result["necessary_attributes"], ["needed"])
        self.assertEqual(result["inappropriate_attributes"], ["secret"])
        self.assertEqual(result["ambiguous_attributes"], ["uncertain"])
        self.assertEqual(result["label_source"], "embedded_file")

    def test_embedded_qwen_labels_require_all_contexts(self):
        source = [{
            "bio": {"name": "Test User"},
            "information_attributes": {"0": {"memory_statement": "memory"}},
            "contexts": [],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_path = Path(temp_dir) / "labels.json"
            labels_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing context 0"):
                load_embedded_cimemories_labelings(
                    str(labels_path),
                    dataset_name="dataset.json",
                    persona_idx=0,
                    persona_name="Test User",
                    memories=["memory"],
                    valid_contexts=[(0, {"recipient": "Doctor", "task": "Provide care"})],
                )

    def test_label_reuse_requires_matching_model_when_requested(self):
        memories = ["memory one"]
        payload = {
            "dataset_name": "dataset.json",
            "persona_idx": 0,
            "persona_name": "Test User",
            "context_idx": 0,
            "llm_model": "gpt-5",
            "samples_per_persona": 10,
            "persona_sampling": [{}, {}, {}],
            "attribute_results": [{
                "attribute": memories[0],
                "persona_distributions": {
                    "privacy_fundamentalist": {"share": 1.0, "private": 0.0, "abstain": 0.0},
                    "pragmatist": {"share": 1.0, "private": 0.0, "abstain": 0.0},
                    "unconcerned": {"share": 1.0, "private": 0.0, "abstain": 0.0},
                },
                "mixture_distribution": {"share": 1.0, "private": 0.0, "abstain": 0.0},
                "final_label_name": "necessary",
            }],
            "necessary_attributes": memories,
            "inappropriate_attributes": [],
            "ambiguous_attributes": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            candidate_dir = Path(temp_dir) / "candidate"
            candidate_dir.mkdir()
            (candidate_dir / "context_labeling_cimemories.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with patch.object(Path, "glob", return_value=[candidate_dir]), patch(
                "letta_research_chat.cli._is_complete_cimemories_labeling_payload",
                return_value=True,
            ):
                matching = find_existing_complete_cimemories_labeling(
                    dataset_name="dataset.json",
                    persona_idx=0,
                    persona_name="Test User",
                    context_idx=0,
                    memories=memories,
                    llm_model="gpt-5",
                )
                mismatching = find_existing_complete_cimemories_labeling(
                    dataset_name="dataset.json",
                    persona_idx=0,
                    persona_name="Test User",
                    context_idx=0,
                    memories=memories,
                    llm_model="gpt-5.6-luna",
                )

        self.assertEqual(matching, candidate_dir / "context_labeling_cimemories.json")
        self.assertIsNone(mismatching)


if __name__ == "__main__":
    unittest.main()
