import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from letta_research_chat.cli import (  # noqa: E402
    atomic_memobase_context_fact_candidates,
    atomic_memobase_profile_fact_candidates,
    _diagnostic_best_fact,
    _diagnostic_label_source,
    _diagnostic_memory_evidence,
    _diagnostic_stage,
    _diagnostic_winner_cells,
    _pipeline_catalog_label_source,
    _pipeline_label_filter_matches,
    resolve_pipeline_reference,
    RerankResult,
    _profile_config_fingerprint,
    inspect_memobase_server_config_file,
    _profile_document_domain_rank,
    configure_profile_memory_for_mode,
    restore_profile_memory_config,
    retrieve_profile_memories_for_task,
)
from letta_research_chat.persona import persona_label  # noqa: E402
from letta_research_chat.profile_memory import (  # noqa: E402
    FULL_PROFILE_TOKEN_SIZE,
    ProfileMemoryClient,
    ProfileMemoryResult,
)


class FakeProfileClient:
    """Stand-in for ProfileMemoryClient covering only the config surface."""

    def __init__(self, config="language: en\n"):
        self.config = config
        self.updates = []

    def get_config(self):
        return self.config

    def update_config(self, config):
        self.updates.append(config)
        self.config = config
        return True


class PersonaLabelTests(unittest.TestCase):
    def test_prefers_bio_name(self):
        entry = {"bio": {"name": "Douglas Perry"}, "information_attributes": {}}
        self.assertEqual(persona_label(entry), "Douglas Perry")

    def test_missing_event_does_not_return_arbitrary_attribute(self):
        """`"name" in (event or "name")` was always true when event was absent."""
        entry = {
            "information_attributes": {
                "address": {"value": "503 Ward Underpass"},
                "name": {"value": "Douglas Perry"},
            }
        }
        self.assertEqual(persona_label(entry), "Douglas Perry")

    def test_unknown_when_no_name_attribute(self):
        entry = {"information_attributes": {"address": {"value": "503 Ward Underpass"}}}
        self.assertEqual(persona_label(entry), "<unknown persona>")


class DomainRankTests(unittest.TestCase):
    def test_privacy_sensitive_domains_are_ranked(self):
        """Both were previously absent and fell to the rank-100 fallback."""
        for domain in ("mental_health", "relationships"):
            self.assertLess(_profile_document_domain_rank(domain)[0], 100, domain)

    def test_memobase_topic_vocabulary_is_ranked(self):
        """Mode 15 ranks Memobase topics, not dataset information_domain values."""
        for topic in ("basic_info", "contact_info", "psychological", "life_event", "interest"):
            self.assertLess(_profile_document_domain_rank(topic)[0], 100, topic)

    def test_identity_outranks_life_event(self):
        self.assertLess(
            _profile_document_domain_rank("basic_info")[0],
            _profile_document_domain_rank("life_event")[0],
        )

    def test_unknown_domain_falls_back(self):
        self.assertEqual(_profile_document_domain_rank("nonsense")[0], 100)

    def test_rank_is_case_insensitive(self):
        self.assertEqual(
            _profile_document_domain_rank("Mental_Health")[0],
            _profile_document_domain_rank("mental_health")[0],
        )


class ProjectConfigRestoreTests(unittest.TestCase):
    def test_non_locomo_modes_do_not_touch_project_config(self):
        client = FakeProfileClient()
        for mode in (9, 10, 11, 12, 13, 14, 15):
            applied, previous = configure_profile_memory_for_mode(client, mode)
            self.assertFalse(applied)
            self.assertIsNone(previous)
        self.assertEqual(client.updates, [])

    def test_locomo_captures_previous_config(self):
        client = FakeProfileClient(config="language: en\noriginal: true\n")
        applied, previous = configure_profile_memory_for_mode(client, 16)
        self.assertTrue(applied)
        self.assertEqual(previous, "language: en\noriginal: true\n")
        self.assertIn("overwrite_user_profiles", client.config)

    def test_locomo_rerank_uses_and_restores_the_same_schema(self):
        original = "language: en\noriginal: true\n"
        client = FakeProfileClient(config=original)
        applied, previous = configure_profile_memory_for_mode(client, 17)
        self.assertTrue(applied)
        self.assertEqual(previous, original)
        self.assertIn("overwrite_user_profiles", client.config)
        self.assertTrue(restore_profile_memory_config(client, previous))
        self.assertEqual(client.config, original)

    def test_locomo_context_rerank_uses_and_restores_the_same_schema(self):
        original = "language: en\noriginal: true\n"
        client = FakeProfileClient(config=original)
        applied, previous = configure_profile_memory_for_mode(client, 18)
        self.assertTrue(applied)
        self.assertEqual(previous, original)
        self.assertIn("overwrite_user_profiles", client.config)
        self.assertTrue(restore_profile_memory_config(client, previous))
        self.assertEqual(client.config, original)

    def test_locomo_event_rerank_uses_and_restores_the_same_schema(self):
        original = "language: en\noriginal: true\n"
        client = FakeProfileClient(config=original)
        applied, previous = configure_profile_memory_for_mode(client, 19)
        self.assertTrue(applied)
        self.assertEqual(previous, original)
        self.assertIn("overwrite_user_profiles", client.config)
        self.assertTrue(restore_profile_memory_config(client, previous))
        self.assertEqual(client.config, original)

    def test_restore_puts_the_original_config_back(self):
        original = "language: en\noriginal: true\n"
        client = FakeProfileClient(config=original)
        _, previous = configure_profile_memory_for_mode(client, 16)
        self.assertNotEqual(client.config, original)
        self.assertTrue(restore_profile_memory_config(client, previous))
        self.assertEqual(client.config, original)

    def test_restore_is_a_noop_without_a_snapshot(self):
        client = FakeProfileClient()
        self.assertFalse(restore_profile_memory_config(client, None))
        self.assertFalse(restore_profile_memory_config(None, "language: en"))
        self.assertEqual(client.updates, [])


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_and_distinguishing(self):
        self.assertEqual(
            _profile_config_fingerprint("language: en"),
            _profile_config_fingerprint("language: en"),
        )
        self.assertNotEqual(
            _profile_config_fingerprint("language: en"),
            _profile_config_fingerprint("language: zh"),
        )

    def test_empty_config_has_no_fingerprint(self):
        self.assertIsNone(_profile_config_fingerprint(None))
        self.assertIsNone(_profile_config_fingerprint(""))


class ProfileConfigDiagnosticTests(unittest.TestCase):
    def _client(self, sdk_client):
        client = ProfileMemoryClient.__new__(ProfileMemoryClient)
        client._client = sdk_client
        return client

    @patch("letta_research_chat.profile_memory.metadata.version", return_value="1.2.3")
    def test_reports_unsupported_client_method(self, _version):
        result = self._client(SimpleNamespace()).inspect_config()
        self.assertEqual(result["status"], "unsupported")
        self.assertFalse(result["get_config_supported"])
        self.assertEqual(result["sdk_version"], "1.2.3")

    @patch("letta_research_chat.profile_memory.metadata.version", return_value="1.2.3")
    def test_distinguishes_empty_config(self, _version):
        sdk_client = SimpleNamespace(get_config=lambda: None, update_config=lambda value: value)
        result = self._client(sdk_client).inspect_config()
        self.assertEqual(result["status"], "empty")
        self.assertTrue(result["get_config_supported"])
        self.assertTrue(result["update_config_supported"])

    @patch("letta_research_chat.profile_memory.metadata.version", return_value="1.2.3")
    def test_preserves_config_error_details(self, _version):
        def fail():
            raise RuntimeError("endpoint unavailable")

        result = self._client(SimpleNamespace(get_config=fail)).inspect_config()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["error_message"], "endpoint unavailable")
        self.assertIsNone(self._client(SimpleNamespace(get_config=fail)).get_config())

    @patch("letta_research_chat.profile_memory.metadata.version", return_value="1.2.3")
    def test_reports_available_config(self, _version):
        result = self._client(SimpleNamespace(get_config=lambda: "language: en\n")).inspect_config()
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["config"], "language: en\n")


class ServerConfigFileDiagnosticTests(unittest.TestCase):
    def test_only_returns_allowlisted_settings(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "llm_api_key: extremely-secret\n"
                "best_llm_model: gpt-test\n"
                "max_chat_blob_buffer_token_size: 100 # tuned\n"
                "additional_user_profiles:\n"
                "  - private: value\n",
                encoding="utf-8",
            )
            result = inspect_memobase_server_config_file(str(path))

        self.assertEqual(result["visible_settings"]["best_llm_model"], "gpt-test")
        self.assertEqual(result["visible_settings"]["max_chat_blob_buffer_token_size"], "100")
        self.assertNotIn("llm_api_key", result["visible_settings"])
        self.assertNotIn("extremely-secret", str(result))
        self.assertEqual(result["hidden_sections_present"], ["additional_user_profiles"])
        self.assertEqual(len(result["sha256"]), 64)


class SettleTests(unittest.TestCase):
    """The profile keeps growing after flush(sync=True) returns."""

    def _client_yielding(self, counts):
        client = ProfileMemoryClient.__new__(ProfileMemoryClient)
        seq = list(counts)

        def payload(_user, *, max_token_size=None):
            value = seq.pop(0) if seq else counts[-1]
            return {"topic": {f"s{i}": {"content": "x"} for i in range(value)}}

        client._profile_payload = payload  # type: ignore[assignment]
        return client

    def test_waits_until_entry_count_is_stable(self):
        client = self._client_yielding([1, 5, 20, 20, 20, 20])
        waited, converged = client._wait_for_settle(
            object(), timeout_s=60, interval_s=0, stable_checks=3
        )
        self.assertTrue(converged)
        self.assertEqual(client._entry_count(object()), 20)

    def test_reports_non_convergence_on_timeout(self):
        client = ProfileMemoryClient.__new__(ProfileMemoryClient)
        counter = {"n": 0}

        def payload(_user, *, max_token_size=None):
            counter["n"] += 1
            return {"topic": {f"s{i}": {"content": "x"} for i in range(counter["n"])}}

        client._profile_payload = payload  # type: ignore[assignment]
        _, converged = client._wait_for_settle(
            object(), timeout_s=0, interval_s=0, stable_checks=3
        )
        self.assertFalse(converged)

    def test_count_entries_handles_missing_payload(self):
        self.assertEqual(ProfileMemoryClient._count_entries(None), 0)
        self.assertEqual(ProfileMemoryClient._count_entries({}), 0)
        self.assertEqual(ProfileMemoryClient._count_entries({"a": {"x": {}, "y": {}}}), 2)


class QueryAppliedTests(unittest.TestCase):
    def test_default_records_query_as_not_applied(self):
        result = ProfileMemoryResult(
            backend="memobase", user_id="u", query="q", context="c", max_token_size=1000
        )
        self.assertFalse(result.query_applied)
        self.assertFalse(result.to_json()["query_applied"])

    def test_full_profile_budget_exceeds_sdk_default(self):
        self.assertGreater(FULL_PROFILE_TOKEN_SIZE, 1000)


class AtomicProfileFactTests(unittest.TestCase):
    def test_splits_consolidated_entries_and_preserves_provenance(self):
        payload = {
            "life_circumstances": {
                "family_status": {
                    "id": "entry-1",
                    "content": "User is married.; User has one child.\nUser plans to move.",
                }
            }
        }
        candidates = atomic_memobase_profile_fact_candidates(payload)
        self.assertEqual(
            [candidate["fact"] for candidate in candidates],
            ["User is married.", "User has one child.", "User plans to move."],
        )
        self.assertTrue(all(candidate["category"] == "life_circumstances" for candidate in candidates))
        self.assertTrue(all(candidate["key"] == "family_status" for candidate in candidates))

    def test_deduplicates_facts_without_splitting_periods(self):
        payload = {
            "basic_info": {
                "name": {"content": "User is Dr. Smith. User prefers Sam."},
                "duplicate": {"content": "User is Dr. Smith. User prefers Sam."},
            }
        }
        candidates = atomic_memobase_profile_fact_candidates(payload)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["fact"], "User is Dr. Smith. User prefers Sam.")

    def test_locomo_rerank_limits_candidates_by_count_and_selects_facts(self):
        payload = {
            "basic_info": {
                "a": {"content": "Fact A"},
                "b": {"content": "Fact B"},
                "c": {"content": "Fact C"},
            }
        }
        cfg = SimpleNamespace(rerank_candidate_limit=2, rerank_output_limit=1)
        profile = SimpleNamespace(backend="memobase")
        reranked = RerankResult(
            query="q",
            mode="llm",
            candidate_source="all_profile_facts",
            candidate_limit=2,
            output_limit=1,
            candidate_count=2,
            selected_memories=["Fact A"],
            scores=[],
        )
        with patch(
            "letta_research_chat.cli.rerank_memory_candidates_for_task",
            return_value=reranked,
        ) as rerank:
            result = retrieve_profile_memories_for_task(
                cfg=cfg,
                http=object(),
                profile=profile,
                user_id="user-1",
                recipient="Bank",
                task="Apply",
                memory_mode=17,
                full_profile_payload=payload,
            )
        self.assertEqual(rerank.call_args.kwargs["candidates"], ["Fact A", "Fact B"])
        self.assertEqual(result.selected_memories, ["Fact A"])
        self.assertEqual(result.context, "- Fact A")
        self.assertIsNone(result.max_token_size)
        self.assertEqual(result.retrieval_options["candidate_count_before_limit"], 3)


class AtomicNativeContextFactTests(unittest.TestCase):
    CONTEXT = """---
# Memory
Use only relevant memory.
## User Current Profile:
- life_circumstances::career: User is a supervisor.; User works 8 am to 4 pm.
- basic_info::name: User is Douglas Perry.
## Past Events:
- User moved to Kaufmanberg. // event
- User is Douglas Perry. // info
---"""

    def test_parses_only_bullets_and_preserves_native_provenance(self):
        candidates = atomic_memobase_context_fact_candidates(self.CONTEXT)
        self.assertEqual(
            [candidate["fact"] for candidate in candidates],
            [
                "User is a supervisor.",
                "User works 8 am to 4 pm.",
                "User is Douglas Perry.",
                "User moved to Kaufmanberg.",
            ],
        )
        self.assertEqual(candidates[0]["category"], "life_circumstances")
        self.assertEqual(candidates[0]["key"], "career")
        self.assertEqual(candidates[-1]["section"], "Past Events:")
        self.assertEqual(candidates[-1]["event_kind"], "event")

    def test_mode_18_retrieves_native_context_then_reranks_its_facts(self):
        class NativeContextProfile:
            backend = "memobase"

            def __init__(self):
                self.context_kwargs = None

            def context(self, user_id, **kwargs):
                self.context_kwargs = kwargs
                return ProfileMemoryResult(
                    backend=self.backend,
                    user_id=user_id,
                    query=kwargs["query"],
                    context=AtomicNativeContextFactTests.CONTEXT,
                    max_token_size=kwargs["max_token_size"],
                    query_applied=True,
                )

        cfg = SimpleNamespace(
            rerank_candidate_limit=3,
            rerank_output_limit=2,
            memobase_context_max_token_size=1000,
        )
        profile = NativeContextProfile()
        reranked = RerankResult(
            query="q",
            mode="llm",
            candidate_source="memobase_native_context",
            candidate_limit=3,
            output_limit=2,
            candidate_count=3,
            selected_memories=["User is a supervisor."],
            scores=[],
        )
        with patch(
            "letta_research_chat.cli.rerank_memory_candidates_for_task",
            return_value=reranked,
        ) as rerank:
            result = retrieve_profile_memories_for_task(
                cfg=cfg,
                http=object(),
                profile=profile,
                user_id="user-1",
                recipient="Employer",
                task="Confirm work details",
                memory_mode=18,
                full_profile_payload={"cached": "profile"},
            )

        self.assertEqual(profile.context_kwargs["max_token_size"], 3000)
        self.assertEqual(profile.context_kwargs["event_similarity_threshold"], 0.2)
        self.assertTrue(profile.context_kwargs["fill_window_with_events"])
        self.assertEqual(
            rerank.call_args.kwargs["candidates"],
            [
                "User is a supervisor.",
                "User works 8 am to 4 pm.",
                "User is Douglas Perry.",
            ],
        )
        self.assertEqual(result.context, "- User is a supervisor.")
        self.assertEqual(result.source_context, self.CONTEXT)
        self.assertEqual(result.rendering_mode, "locomo_context_atomic_llm_rerank")
        self.assertEqual(result.retrieval_options["candidate_count_before_limit"], 4)

    def test_mode_19_preserves_profile_and_reranks_all_selected_events(self):
        class NativeContextProfile:
            backend = "memobase"

            def __init__(self):
                self.context_kwargs = None

            def context(self, user_id, **kwargs):
                self.context_kwargs = kwargs
                return ProfileMemoryResult(
                    backend=self.backend,
                    user_id=user_id,
                    query=kwargs["query"],
                    context=AtomicNativeContextFactTests.CONTEXT,
                    max_token_size=kwargs["max_token_size"],
                    query_applied=True,
                )

        cfg = SimpleNamespace(
            rerank_candidate_limit=1,
            rerank_output_limit=1,
            memobase_context_max_token_size=1000,
        )
        profile = NativeContextProfile()
        reranked = RerankResult(
            query="q",
            mode="llm",
            candidate_source="memobase_query_selected_events",
            candidate_limit=1,
            output_limit=1,
            candidate_count=2,
            selected_memories=["User moved to Kaufmanberg."],
            scores=[],
        )
        with patch(
            "letta_research_chat.cli.rerank_memory_candidates_for_task",
            return_value=reranked,
        ) as rerank:
            result = retrieve_profile_memories_for_task(
                cfg=cfg,
                http=object(),
                profile=profile,
                user_id="user-1",
                recipient="Employer",
                task="Confirm work details",
                memory_mode=19,
                full_profile_payload={"cached": "profile"},
            )

        self.assertEqual(profile.context_kwargs["max_token_size"], 3000)
        self.assertEqual(
            rerank.call_args.kwargs["candidates"],
            ["User moved to Kaufmanberg.", "User is Douglas Perry."],
        )
        self.assertIn("User is a supervisor.", result.context)
        self.assertIn("User works 8 am to 4 pm.", result.context)
        self.assertIn("## Past Events:", result.context)
        self.assertIn("- User moved to Kaufmanberg.", result.context)
        self.assertNotIn("- User is Douglas Perry. // info", result.context)
        self.assertEqual(result.source_context, self.CONTEXT)
        self.assertEqual(
            result.rendering_mode, "locomo_profile_plus_reranked_events"
        )
        self.assertIsNone(result.retrieval_options["event_candidate_limit"])
        self.assertEqual(result.retrieval_options["profile_fact_count"], 3)
        self.assertEqual(result.retrieval_options["event_candidate_count"], 2)
        self.assertEqual(result.retrieval_options["selected_event_count"], 1)
        self.assertEqual(result.rerank["candidate_limit"], 2)
        self.assertEqual(len(result.pre_rerank_memory_items), 5)
        self.assertEqual(len(result.post_rerank_memory_items), 4)
        self.assertEqual(
            [item["component"] for item in result.pre_rerank_memory_items],
            [
                "persistent_profile",
                "persistent_profile",
                "persistent_profile",
                "query_selected_event",
                "query_selected_event",
            ],
        )
        self.assertTrue(
            all(
                item["retained_after_rerank"]
                for item in result.pre_rerank_memory_items[:3]
            )
        )
        self.assertFalse(result.pre_rerank_memory_items[-1]["retained_after_rerank"])
        serialized = result.to_json()
        self.assertEqual(
            serialized["retrieval_options"]["pre_rerank_metric_scope"],
            "persistent_profile_plus_all_query_selected_events",
        )
        self.assertEqual(len(serialized["pre_rerank_memory_items"]), 5)


class ArchitectureDiagnosticTests(unittest.TestCase):
    def test_pipeline_catalog_label_source_uses_manifest_fallback(self):
        self.assertEqual(
            _pipeline_catalog_label_source(
                {"summary": {"context_label_model": "gpt-5", "contexts": []}}
            ),
            "model:gpt-5",
        )

    def test_pipeline_label_filter_is_exact_for_versioned_gpt_models(self):
        self.assertTrue(_pipeline_label_filter_matches("model:gpt-5", "gpt-5"))
        self.assertFalse(_pipeline_label_filter_matches("model:gpt-5.6-luna", "gpt-5"))
        self.assertTrue(
            _pipeline_label_filter_matches(
                "external:labels_qwen.json (qwen3.8-27b)",
                "qwen",
            )
        )

    def test_label_source_distinguishes_generated_and_external_labels(self):
        self.assertEqual(
            _diagnostic_label_source({}, [{"llm_model": "gpt-5"}]),
            "model:gpt-5",
        )
        self.assertEqual(
            _diagnostic_label_source(
                {},
                [{
                    "llm_model": "qwen3.8-27b",
                    "label_source": "embedded_file",
                    "source_labels_file": "/tmp/labels_qwen.json",
                }],
            ),
            "external:labels_qwen.json (qwen3.8-27b)",
        )

    def test_label_source_marks_mixed_context_provenance(self):
        source = _diagnostic_label_source(
            {},
            [{"llm_model": "gpt-5"}, {"llm_model": "gpt-5.6-luna"}],
        )
        self.assertTrue(source.startswith("MIXED:"))
        self.assertIn("model:gpt-5", source)
        self.assertIn("model:gpt-5.6-luna", source)

    def test_winner_cells_mark_each_category_and_all_ties(self):
        cells = _diagnostic_winner_cells(
            [(0.9, 0.01, 0.03), (0.8, 0.01, 0.04), (0.7, 0.04, 0.02)]
        )
        self.assertIn("C★90.0", cells[0].text)
        self.assertIn("L★1.0", cells[0].text)
        self.assertIn("L★1.0", cells[1].text)
        self.assertIn("A★2.0", cells[2].text)

    def test_best_fact_matches_rephrased_profile_fact(self):
        fact, score = _diagnostic_best_fact(
            "My annual income is $72,000.",
            ["User owns a home", "User's annual income is $72,000."],
        )
        self.assertEqual(fact, "User's annual income is $72,000.")
        self.assertEqual(score, 1.0)

    def test_profile_evidence_uses_candidates_and_selected_facts(self):
        candidates, selected = _diagnostic_memory_evidence(
            {
                "profile_memory": {
                    "candidates": [{"fact": "Fact A"}, {"fact": "Fact B"}],
                    "selected_memories": ["Fact B"],
                }
            }
        )
        self.assertEqual(candidates, ["Fact A", "Fact B"])
        self.assertEqual(selected, ["Fact B"])

    def test_graph_evidence_reads_structured_fact_candidates(self):
        candidates, selected = _diagnostic_memory_evidence(
            {
                "graph_memory": {
                    "facts": [{"fact": "Fact A"}, {"fact": "Fact B"}],
                    "selected_memories": ["Fact A"],
                }
            }
        )
        self.assertEqual(candidates, ["Fact A", "Fact B"])
        self.assertEqual(selected, ["Fact A"])

    def test_stage_heuristic_distinguishes_pipeline_failures(self):
        self.assertEqual(_diagnostic_stage(0, 0.2, 0.0), "likely preprocessing/ingestion gap")
        self.assertEqual(_diagnostic_stage(0, 1.0, 0.2), "likely retrieval/reranking gap")
        self.assertEqual(_diagnostic_stage(0, 1.0, 1.0), "likely response omission")
        self.assertEqual(_diagnostic_stage(40, 1.0, 1.0), "intermittent response use")
        self.assertEqual(_diagnostic_stage(90, 0.2, 0.0), "candidate absent; response inferred")
        self.assertEqual(_diagnostic_stage(50, 1.0, 0.2), "selection mismatch; response inferred")
        self.assertEqual(_diagnostic_stage(100, 0.0, 0.0), "shared consistently")

    def test_pipeline_references_resolve_index_latest_and_mode(self):
        catalog = [
            {
                "index": 1,
                "path": Path("research_outputs/new"),
                "summary": {"memory_mode": 17},
            },
            {
                "index": 2,
                "path": Path("research_outputs/older"),
                "summary": {"memory_mode": 3},
            },
        ]
        with patch("letta_research_chat.cli._pipeline_run_catalog", return_value=catalog), patch(
            "letta_research_chat.cli._pipeline_catalog_metrics",
            return_value=("1", "0", "0", "complete"),
        ):
            self.assertEqual(resolve_pipeline_reference("@latest"), "research_outputs/new")
            self.assertEqual(resolve_pipeline_reference("@2"), "research_outputs/older")
            self.assertEqual(resolve_pipeline_reference("@mode:list-rerank"), "research_outputs/older")

    def test_pipeline_reference_errors_are_actionable(self):
        catalog = [{"index": 1, "path": Path("one"), "summary": {"memory_mode": 3}}]
        with patch("letta_research_chat.cli._pipeline_run_catalog", return_value=catalog):
            with self.assertRaisesRegex(ValueError, "/pipeline_runs"):
                resolve_pipeline_reference("@2")
            with self.assertRaisesRegex(ValueError, "@latest"):
                resolve_pipeline_reference("@wat")


if __name__ == "__main__":
    unittest.main()
