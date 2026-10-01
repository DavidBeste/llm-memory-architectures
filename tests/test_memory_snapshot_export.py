import json
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from letta_research_chat.memory_snapshot_export import (  # noqa: E402
    _candidates_from_rerank_prompt,
    _full_graph_svg,
    _full_graph_html,
    _graphml,
    _memory_dump_html,
    _pre_rerank_candidates,
    _retrieval_svg,
    export_memory_snapshots,
    export_pre_rerank_candidates,
)


class MemorySnapshotExportTests(unittest.TestCase):
    def test_extracts_exact_candidates_for_all_three_reranked_architectures(self):
        prompt = 'Header\nCandidate memories:\n[{"index":0,"memory":"Fact A"},{"index":1,"memory":"Fact B"}]\n\nRespond now'
        self.assertEqual(
            [item["memory"] for item in _candidates_from_rerank_prompt(prompt)],
            ["Fact A", "Fact B"],
        )
        cases = [
            (
                {"rerank": {"candidate_count": 2, "candidate_limit": 200, "prompt": prompt, "selected_memories": ["Fact B"]}},
                "list",
                "rerank.prompt",
            ),
            (
                {"graph_memory": {"candidate_count": 1, "facts": [{"fact": "Graph fact", "uuid": "u"}], "selected_memories": []}},
                "graph",
                "graph_memory.facts",
            ),
            (
                {"profile_memory": {"candidates": [{"fact": "Profile fact", "category": "basic"}], "selected_memories": ["Profile fact"], "rerank": {"candidate_count": 1, "candidate_limit": 200}}},
                "profile",
                "profile_memory.candidates",
            ),
        ]
        for record, expected_architecture, expected_source in cases:
            architecture, candidates, status = _pre_rerank_candidates(record)
            self.assertEqual(architecture, expected_architecture)
            self.assertEqual(status["recovery_source"], expected_source)
            self.assertTrue(status["complete_reranker_input"])
            self.assertTrue(candidates)

    def test_mode_19_raw_scope_includes_profile_and_events(self):
        record = {
            "profile_memory": {
                "pre_rerank_memory_items": [
                    {
                        "fact": "Profile fact",
                        "component": "persistent_profile",
                        "reranker_input": False,
                        "retained_after_rerank": True,
                    },
                    {
                        "fact": "Relevant event",
                        "component": "query_selected_event",
                        "reranker_input": True,
                        "retained_after_rerank": True,
                    },
                    {
                        "fact": "Discarded event",
                        "component": "query_selected_event",
                        "reranker_input": True,
                        "retained_after_rerank": False,
                    },
                ],
                "retrieval_options": {"pre_rerank_memory_item_count": 3},
                "selected_memories": ["Relevant event"],
                "rerank": {"candidate_count": 2, "candidate_limit": 2},
            }
        }
        architecture, candidates, status = _pre_rerank_candidates(record)
        self.assertEqual(architecture, "profile")
        self.assertEqual(status["recovery_source"], "profile_memory.pre_rerank_memory_items")
        self.assertEqual(
            status["metric_scope"],
            "persistent_profile_plus_all_query_selected_events",
        )
        self.assertTrue(status["complete_reranker_input"])
        self.assertIsNone(status["candidate_limit"])
        self.assertEqual(len(candidates), 3)
        self.assertTrue(candidates[0]["selected_after_rerank"])
        self.assertFalse(candidates[2]["selected_after_rerank"])
        self.assertEqual(
            candidates[0]["provenance"]["component"], "persistent_profile"
        )

    def test_pre_rerank_export_writes_searchable_and_analysis_ready_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query = root / "query"
            query.mkdir()
            prompt = 'Candidate memories:\n[{"index":0,"memory":"Fact A"},{"index":1,"memory":"Fact B"}]\nRespond'
            record = {
                "user_idx": 0,
                "persona_name": "Test Person",
                "memory_mode": 3,
                "context_idx": 0,
                "repeat_idx": 0,
                "recipient": "Bank",
                "task": "Apply",
                "rerank": {
                    "candidate_count": 2,
                    "candidate_limit": 2,
                    "prompt": prompt,
                    "selected_memories": ["Fact B"],
                },
            }
            (query / "manifest.json").write_text(
                json.dumps({
                    "user_idx": 0,
                    "persona_name": "Test Person",
                    "memory_mode": 3,
                    "memory_mode_name": "list-rerank",
                    "records": [record],
                }),
                encoding="utf-8",
            )

            output = export_pre_rerank_candidates(str(query), output_dir=str(root / "export"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            context = json.loads((output / "personas/persona_000/contexts/context_000.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["all_candidate_sets_complete"])
            self.assertEqual(manifest["candidate_occurrence_count"], 2)
            self.assertTrue(context["candidate_set"]["candidate_limit_reached"])
            self.assertTrue(context["candidates"][1]["selected_after_rerank"])
            self.assertIn("Fact A", (output / "candidates.csv").read_text(encoding="utf-8"))
            self.assertEqual(len((output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            self.assertIn("selected only", (output / "personas/persona_000/contexts/context_000.html").read_text(encoding="utf-8"))

    def test_list_and_profile_memory_dump_html_is_portable_and_searchable(self):
        list_page = _memory_dump_html(
            persona_name="Test <Person>",
            persona_idx=0,
            mode_name="list-rerank",
            backend={"atomic_entries": [{"index": 0, "memory_statement": "Lives at A & B"}]},
        )
        profile_page = _memory_dump_html(
            persona_name="Test Person",
            persona_idx=0,
            mode_name="profile-locomo-rerank",
            backend={"profile": {"basic_info": {"occupation": "Engineer"}}},
        )

        self.assertIn("Test &lt;Person&gt;", list_page)
        self.assertIn("Lives at A &amp; B", list_page)
        self.assertIn("document.querySelectorAll", list_page)
        self.assertNotIn("http://", list_page)
        self.assertNotIn("https://", list_page)
        self.assertIn("basic_info", profile_page)
        self.assertIn("Engineer", profile_page)

    def test_graph_visualizations_are_valid_xml_and_preserve_topology(self):
        graph = {
            "group_id": "example-group",
            "nodes": [
                {"element_id": "person", "labels": ["Entity"], "properties": {"name": "Person & Co"}},
                {"element_id": "employer", "labels": ["Entity"], "properties": {"name": "Employer"}},
                {"element_id": "episode", "labels": ["Episodic"], "properties": {"name": "Episode 1"}},
            ],
            "relationships": [
                {"element_id": "fact", "type": "RELATES_TO", "source_element_id": "person", "target_element_id": "employer", "properties": {"name": "WORKS_FOR", "fact": "Person works for Employer"}},
                {"element_id": "mention", "type": "MENTIONS", "source_element_id": "episode", "target_element_id": "person", "properties": {}},
            ],
        }
        graphml = _graphml(graph)
        full_svg = _full_graph_svg(graph, "Person & Co")
        interactive_html = _full_graph_html(graph, "Person & Co")
        retrieval_svg = _retrieval_svg(
            {"candidate_count": 2, "selected_memories": ["Person works for Employer"]},
            "Person & Co",
            0,
            "Bank <Officer>",
        )

        ElementTree.fromstring(graphml)
        ElementTree.fromstring(full_svg)
        ElementTree.fromstring(retrieval_svg)
        self.assertIn("RELATES_TO", graphml)
        self.assertIn("Person works for Employer", full_svg)
        self.assertIn("Person works for Employer", interactive_html)
        self.assertIn("Relationship names", interactive_html)
        self.assertIn("e.properties.name||e.type", interactive_html)
        self.assertIn("Show episodes and MENTIONS", interactive_html)
        self.assertIn("type=\"application/json\"", interactive_html)
        self.assertNotIn("<script src=", interactive_html)
        self.assertNotIn("<link rel=", interactive_html)

    def test_dataset_export_deduplicates_repeated_context_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data.json"
            dataset.write_text(
                json.dumps([{"bio": {"name": "Test <Person>"}, "information_attributes": {"name": {"memory_statement": "My name is Test Person."}}, "contexts": [{"recipient": "Bank", "task": "Apply"}]}]),
                encoding="utf-8",
            )
            run = root / "dataset-run"
            pipeline = run / "personas/persona_000/pipeline"
            query = run / "personas/persona_000/query"
            pipeline.mkdir(parents=True)
            query.mkdir()
            records = []
            for repeat in range(2):
                records.append({
                    "dataset_name": str(dataset), "user_idx": 0, "persona_name": "Test <Person>",
                    "memory_mode": 3, "context_idx": 0, "repeat_idx": repeat,
                    "recipient": "Bank & Lender", "task": "Apply <now>",
                    "attributes": ["My name is Test Person."],
                    "rerank": {"candidate_count": 1, "output_limit": 20, "selected_memories": ["My name is Test Person."], "raw_response": {"large": True}, "prompt": "large"},
                })
            responses = query / "responses.jsonl"
            responses.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            (query / "manifest.json").write_text(json.dumps({
                "source_dataset": str(dataset), "user_idx": 0, "persona_name": "Test <Person>",
                "memory_mode": 3, "memory_mode_name": "list-rerank", "records": records,
                "memory_initialization": {"memory_statements_inserted": 1},
                "artifacts": {"responses_jsonl": str(responses)},
            }), encoding="utf-8")
            (pipeline / "manifest.json").write_text(json.dumps({
                "dataset_name": str(dataset), "persona_idx": 0, "persona_name": "Test <Person>",
                "memory_mode": 3, "memory_mode_name": "list-rerank", "contexts": [{}],
                "artifacts": {"query_output_dir": str(query)},
            }), encoding="utf-8")
            run.mkdir(exist_ok=True)
            (run / "manifest.json").write_text(json.dumps({
                "command": "run_privacy_pipeline_cimemories_dataset", "status": "complete",
                "persona_count": 1, "personas": [{"local_pipeline": str(pipeline)}],
            }), encoding="utf-8")

            output = export_memory_snapshots(str(run), output_dir=str(root / "export"))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["persona_count"], 1)
            self.assertEqual(manifest["context_snapshot_count"], 1)
            snapshot = json.loads((output / "personas/persona_000/contexts/context_000.json").read_text())
            self.assertEqual(snapshot["repeat_count"], 2)
            self.assertTrue(snapshot["shared_across_repeats"])
            self.assertNotIn("raw_response", snapshot["retrieval"])
            page = (output / "index.html").read_text()
            self.assertIn("Test &lt;Person&gt;", page)
            self.assertIn("Bank &amp; Lender", page)
            self.assertIn("Formatted memory dump", page)
            memory_page = output / "personas/persona_000/memory/full_memory.html"
            self.assertTrue(memory_page.is_file())
            self.assertIn("My name is Test Person.", memory_page.read_text(encoding="utf-8"))
            self.assertTrue((output / "architecture_flow.svg").is_file())


if __name__ == "__main__":
    unittest.main()
