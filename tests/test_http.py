import os
import unittest
from unittest.mock import patch

from letta_research_chat.http import HttpClient


class HttpClientTimeoutTests(unittest.TestCase):
    def test_server_timeout_defaults_to_120_seconds(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(HttpClient().timeout_default, 120)

    def test_server_timeout_can_be_configured(self) -> None:
        with patch.dict(os.environ, {"LETTA_SERVER_TIMEOUT": "300"}, clear=True):
            self.assertEqual(HttpClient().timeout_default, 300)

    def test_server_timeout_rejects_invalid_values(self) -> None:
        for value in ("", "invalid", "0", "-1"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {"LETTA_SERVER_TIMEOUT": value}, clear=True
                ):
                    with self.assertRaisesRegex(
                        ValueError, "LETTA_SERVER_TIMEOUT must be a positive integer"
                    ):
                        HttpClient()

    def test_explicit_timeout_still_overrides_environment_default(self) -> None:
        with patch.dict(os.environ, {"LETTA_SERVER_TIMEOUT": "300"}, clear=True):
            self.assertEqual(HttpClient(timeout_default=15).timeout_default, 15)


if __name__ == "__main__":
    unittest.main()
