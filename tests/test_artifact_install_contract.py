from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ArtifactInstallContractTest(unittest.TestCase):
    def test_unrelated_code_style_datasets_are_excluded_from_archives(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        excluded = (
            "cweval_core_c_sections.jsonl",
            "cweval_lang_c_sections.jsonl",
            "cweval_schema_sections.jsonl",
            "dataset_full_functions_final.json",
            "dataset_functions.json",
        )
        for filename in excluded:
            self.assertIn(f"{filename} export-ignore", attributes)

    def test_memobase_is_conditional_on_supported_python(self) -> None:
        requirements = (ROOT / "requirements-artifact.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'memobase==0.0.27; python_version >= "3.11"', requirements
        )

    def test_reviewer_guides_use_python_311_for_full_reproduction(self) -> None:
        for filename in ("ARTIFACT.md", "REPRODUCING_EXPERIMENTS.md"):
            guide = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("python3.11 -m venv .venv", guide)
            self.assertIn("offline", guide.lower())
            self.assertIn("Python 3.10", guide)


if __name__ == "__main__":
    unittest.main()
