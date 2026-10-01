import re
import unittest

from letta_research_chat.cli import _user_input_prompt


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class CliPromptTests(unittest.TestCase):
    def test_readline_prompt_marks_ansi_sequences_as_nonprinting(self) -> None:
        prompt = _user_input_prompt(readline_enabled=True)

        self.assertEqual(prompt.count("\001"), 2)
        self.assertEqual(prompt.count("\002"), 2)
        visible = ANSI_ESCAPE.sub("", prompt.replace("\001", "").replace("\002", ""))
        self.assertEqual(visible, "USER: ")

    def test_plain_prompt_omits_readline_markers(self) -> None:
        prompt = _user_input_prompt(readline_enabled=False)

        self.assertNotIn("\001", prompt)
        self.assertNotIn("\002", prompt)
        self.assertEqual(ANSI_ESCAPE.sub("", prompt), "USER: ")


if __name__ == "__main__":
    unittest.main()
