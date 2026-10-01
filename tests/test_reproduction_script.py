from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_cimemories_experiments.sh"


class ReproductionScriptTest(unittest.TestCase):
    def run_plan(self, *args: str) -> str:
        completed = subprocess.run(
            [str(SCRIPT), *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    def test_gpt_plan_preserves_asymmetric_repeat_policy(self) -> None:
        output = self.run_plan("gpt", "responses")
        self.assertEqual(output.count("/run_privacy_pipeline_cimemories_dataset"), 3)
        self.assertEqual(output.count("--repeats 10"), 3)
        self.assertNotIn("--with-pre-rerank", output)

    def test_open_model_plan_generates_both_response_stages(self) -> None:
        output = self.run_plan("deepseek", "responses")
        self.assertIn("0,1,2,3,4,5,6,7,8,9", output)
        self.assertIn("--repeats 1", output)
        self.assertIn("--with-pre-rerank", output)

    def test_memory_plan_fixes_exact_and_semantic_strategies(self) -> None:
        output = self.run_plan(
            "glm",
            "memory",
            "--list-pipeline",
            "LIST",
            "--graph-pipeline",
            "GRAPH",
            "--profile-pipeline",
            "PROFILE",
        )
        self.assertIn("LIST --strategy exact-match", output)
        self.assertIn("GRAPH --strategy monolithic", output)
        self.assertIn("PROFILE --strategy monolithic", output)

    def test_report_plan_uses_locked_matched_repeat_design(self) -> None:
        output = self.run_plan("all", "report")
        self.assertIn("--bootstrap-iterations 5000", output)
        self.assertIn("--first-post-repeat", output)
        self.assertIn("--memory-persona 0", output)
        self.assertIn("--memory-judge-model gpt-6-sol", output)
        self.assertIn("--require-complete", output)
        self.assertIn("--require-memory-complete", output)


if __name__ == "__main__":
    unittest.main()
