import logging
import unittest

from letta_research_chat.cli import _GraphitiIngestionWarningHandler


class GraphitiIngestionWarningHandlerTests(unittest.TestCase):
    def test_extracts_unresolved_edge_details(self) -> None:
        handler = _GraphitiIngestionWarningHandler()
        record = logging.LogRecord(
            name="graphiti_core.utils.maintenance.edge_operations",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="Target entity not found in nodes for edge relation: HAS_SALARY",
            args=(),
            exc_info=None,
        )

        handler.handle(record)

        self.assertEqual(len(handler.records), 1)
        captured = handler.records[0]
        self.assertEqual(captured["category"], "unresolved_edge_endpoint")
        self.assertEqual(captured["missing_endpoint"], "target")
        self.assertEqual(captured["relation"], "HAS_SALARY")
        self.assertEqual(captured["logger"], record.name)

    def test_preserves_other_graphiti_warnings(self) -> None:
        handler = _GraphitiIngestionWarningHandler()
        record = logging.LogRecord(
            name="graphiti_core.test",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="Another Graphiti warning",
            args=(),
            exc_info=None,
        )

        handler.handle(record)

        self.assertEqual(handler.records[0]["message"], "Another Graphiti warning")
        self.assertNotIn("category", handler.records[0])


if __name__ == "__main__":
    unittest.main()
