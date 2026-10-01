from __future__ import annotations

import unittest

from letta_research_chat.memory import MemoryClient


class _RecordingHttp:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str):
        self.urls.append(url)
        return {"count": 0, "results": []}


class MemoryClientTests(unittest.TestCase):
    def test_archival_search_sends_letta_top_k_parameter(self):
        http = _RecordingHttp()
        memory = MemoryClient("http://localhost:8283/v1", http)  # type: ignore[arg-type]

        result = memory.search_archival_memory(
            "agent-example", "income & employment", limit=50
        )

        self.assertEqual(result, {"count": 0, "results": []})
        self.assertEqual(
            http.urls,
            [
                "http://localhost:8283/v1/agents/agent-example/archival-memory/search"
                "?query=income%20%26%20employment&top_k=50"
            ],
        )
        self.assertNotIn("&limit=", http.urls[0])


if __name__ == "__main__":
    unittest.main()
