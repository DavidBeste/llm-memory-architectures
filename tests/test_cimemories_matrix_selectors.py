import unittest

from letta_research_chat.cli import parse_memory_mode_selector, parse_persona_selector


class CimemoriesMatrixSelectorTests(unittest.TestCase):
    def test_parses_and_deduplicates_personas_in_order(self) -> None:
        self.assertEqual(parse_persona_selector("5, 6,5,9"), [5, 6, 9])

    def test_rejects_invalid_persona_selector(self) -> None:
        for value in ("", "5,,6", "-1", "five"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_persona_selector(value)

    def test_parses_named_and_legacy_memory_modes(self) -> None:
        self.assertEqual(
            parse_memory_mode_selector(
                "list-rerank,graph-rerank,profile-locomo-events-rerank,3"
            ),
            [3, 6, 19],
        )

    def test_rejects_empty_memory_mode_component(self) -> None:
        with self.assertRaises(ValueError):
            parse_memory_mode_selector("list-rerank,")


if __name__ == "__main__":
    unittest.main()
