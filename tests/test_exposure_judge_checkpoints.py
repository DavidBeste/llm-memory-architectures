import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from letta_research_chat.cli import (
    _configuration_fingerprint,
    _judge_exposed_attributes_async,
    _response_exposure_judge_configuration,
)


class FakeExposureJudge:
    model = "gpt-5.2"
    reasoning_effort = "none"
    api_style = "responses"
    base_url = "https://api.openai.com/v1"
    max_output_tokens = 4096
    use_indexed_exposed_judge = False

    def __init__(self, fail_response: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_response = fail_response

    def build_prompt(self, attributes, response):
        return f"attributes={attributes!r}\nresponse={response}"

    def build_indexed_prompt(self, attributes, response):
        return self.build_prompt(attributes, response)

    def judge_exposed_attributes(self, attributes, response):
        self.calls.append(response)
        if response == self.fail_response:
            raise RuntimeError("temporary provider failure")
        return (
            {attributes[0]: response},
            {
                "id": f"response-{len(self.calls)}",
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
            self.build_prompt(attributes, response),
        )


def records() -> list[dict]:
    return [
        {
            "experiment_id": "experiment",
            "user_idx": 0,
            "persona_name": "Ada",
            "memory_mode": 3,
            "agent_model": "model",
            "agent_reasoning_effort": None,
            "context_idx": index,
            "repeat_idx": 0,
            "recipient": "Bank",
            "task": f"Task {index}",
            "assistant_response": f"Response {index}",
        }
        for index in range(2)
    ]


class ExposureJudgeCheckpointTests(unittest.TestCase):
    def run_judging(self, judge, calls_dir, rows=None):
        configuration_fingerprint = _configuration_fingerprint(
            _response_exposure_judge_configuration(judge)
        )

        async def direct_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch(
            "letta_research_chat.cli.asyncio.to_thread", new=direct_to_thread
        ):
            return asyncio.run(
                _judge_exposed_attributes_async(
                    judge=judge,
                    attributes=["My name is Ada."],
                    records=rows or records(),
                    concurrency=2,
                    calls_dir=calls_dir,
                    judge_configuration_fingerprint=configuration_fingerprint,
                )
            )

    def test_reuses_each_successful_response_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls_dir = Path(tmp) / "exposure_calls"
            first_judge = FakeExposureJudge()
            first = self.run_judging(first_judge, calls_dir)
            self.assertEqual(len(first_judge.calls), 2)
            self.assertTrue(all(not row["checkpoint_reused"] for row in first))

            second_judge = FakeExposureJudge()
            second = self.run_judging(second_judge, calls_dir)
            self.assertEqual(second_judge.calls, [])
            self.assertTrue(all(row["checkpoint_reused"] for row in second))
            self.assertEqual(len(list(calls_dir.glob("*.json"))), 2)

    def test_changed_response_recomputes_only_that_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls_dir = Path(tmp) / "exposure_calls"
            self.run_judging(FakeExposureJudge(), calls_dir)
            changed = records()
            changed[1] = {**changed[1], "assistant_response": "Changed response"}

            judge = FakeExposureJudge()
            results = self.run_judging(judge, calls_dir, changed)
            self.assertEqual(judge.calls, ["Changed response"])
            reused_by_context = {
                row["record"]["context_idx"]: row["checkpoint_reused"]
                for row in results
            }
            self.assertEqual(reused_by_context, {0: True, 1: False})

    def test_failed_checkpoint_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls_dir = Path(tmp) / "exposure_calls"
            failed = self.run_judging(
                FakeExposureJudge(fail_response="Response 1"), calls_dir
            )
            self.assertEqual(sum(bool(row["judge_error"]) for row in failed), 1)

            retry_judge = FakeExposureJudge()
            retried = self.run_judging(retry_judge, calls_dir)
            self.assertEqual(retry_judge.calls, ["Response 1"])
            self.assertEqual(sum(bool(row["judge_error"]) for row in retried), 0)

    def test_changed_judge_configuration_does_not_reuse_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls_dir = Path(tmp) / "exposure_calls"
            self.run_judging(FakeExposureJudge(), calls_dir)

            changed_judge = FakeExposureJudge()
            changed_judge.max_output_tokens = 8192
            results = self.run_judging(changed_judge, calls_dir)
            self.assertEqual(changed_judge.calls, ["Response 0", "Response 1"])
            self.assertTrue(all(not row["checkpoint_reused"] for row in results))
            self.assertEqual(len(list(calls_dir.glob("*.json"))), 4)


if __name__ == "__main__":
    unittest.main()
