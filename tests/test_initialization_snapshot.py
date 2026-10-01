from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from letta_research_chat.initialization_snapshot import capture_initialized_memory_snapshot
from letta_research_chat.persona import PersonaInitResult
from letta_research_chat.profile_memory import ProfileMemoryInitResult


class _ListMemory:
    def list_archival_passages(self, agent_id: str, limit: int, ascending: bool):
        self.call = (agent_id, limit, ascending)
        return {
            "data": [
                {"id": "one", "text": "First fact", "embedding": [0.1, 0.2]},
                {"id": "two", "text": "Second fact"},
            ]
        }


class InitializationSnapshotTests(unittest.TestCase):
    def test_list_snapshot_uses_backend_records_and_omits_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = _ListMemory()
            result = capture_initialized_memory_snapshot(
                output_dir=Path(tmp),
                memory_mode=3,
                memory_mode_name="list-rerank",
                persona_idx=0,
                persona_name="Example",
                source_statements=["First fact", "Second fact"],
                init_result=PersonaInitResult(0, "Example", 2, 2, 0),
                mem=memory,
                agent_id="agent-1",
            )
            self.assertTrue(result["complete"])
            self.assertEqual(memory.call, ("agent-1", 2, True))
            payload = json.loads(
                (Path(tmp) / "initialized_memory" / "full_list_memory.json").read_text()
            )
            self.assertEqual(payload["atomic_entries"][0]["memory_statement"], "First fact")
            self.assertNotIn("embedding", payload["atomic_entries"][0]["backend_record"])
            self.assertTrue((Path(tmp) / "initialized_memory" / "full_list_memory.html").is_file())

    def test_profile_completeness_uses_settled_entries_not_input_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_init = ProfileMemoryInitResult(
                backend="memobase",
                user_id="user-1",
                inserted_count=3,
                flush_sync=True,
                profile={"identity": {"name": "Example", "city": "Berlin"}},
                entries_at_flush=1,
                entries_settled=2,
                settle_converged=True,
            )
            result = capture_initialized_memory_snapshot(
                output_dir=Path(tmp),
                memory_mode=17,
                memory_mode_name="profile-locomo-rerank",
                persona_idx=0,
                persona_name="Example",
                source_statements=["a", "b", "c"],
                init_result=PersonaInitResult(0, "Example", 3, 3, 0),
                profile_init_result=profile_init,
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["backend_observed_primary_count"], 2)
            self.assertTrue((Path(tmp) / "initialized_memory" / "full_profile.html").is_file())

    def test_graph_snapshot_exports_all_visual_formats(self):
        graph = {
            "group_id": "group-1",
            "nodes": [
                {"element_id": "e", "labels": ["Entity"], "properties": {"name": "Example"}},
                {"element_id": "p", "labels": ["Episodic"], "properties": {"content": "fact"}},
            ],
            "relationships": [
                {
                    "element_id": "r",
                    "type": "MENTIONS",
                    "source_element_id": "p",
                    "target_element_id": "e",
                    "properties": {},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "letta_research_chat.initialization_snapshot._read_full_graph", return_value=graph
        ) as read:
            result = capture_initialized_memory_snapshot(
                output_dir=Path(tmp),
                memory_mode=6,
                memory_mode_name="graph-rerank",
                persona_idx=0,
                persona_name="Example",
                source_statements=["fact"],
                init_result=PersonaInitResult(0, "Example", 1, 1, 0),
                graph_group_id="group-1",
                neo4j_uri="bolt://example",
                neo4j_user="neo4j",
                neo4j_password="secret",
            )
            self.assertTrue(result["complete"])
            self.assertFalse(read.call_args.kwargs["include_embeddings"])
            for filename in ("full_graph.json", "full_graph.graphml", "full_graph.svg", "full_graph.html"):
                self.assertTrue((Path(tmp) / "initialized_memory" / filename).is_file())


if __name__ == "__main__":
    unittest.main()
