from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class GraphMemoryFact:
    fact: str
    uuid: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    created_at: str | None = None
    source_node_uuid: str | None = None
    target_node_uuid: str | None = None
    kind: str = "edge"
    score: float | None = None


@dataclass(frozen=True)
class GraphRetrievalResult:
    query: str
    group_id: str
    search_limit: int
    facts: list[GraphMemoryFact]
    backend: str = "graphiti"
    retrieval_strategy: str = "edge_hybrid_rrf"
    center_node_uuid: str | None = None
    selected_fact_texts: list[str] | None = None
    rerank: dict[str, Any] | None = None

    @property
    def selected_memories(self) -> list[str]:
        if self.selected_fact_texts is not None:
            return self.selected_fact_texts
        return [fact.fact for fact in self.facts]

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "query": self.query,
            "group_id": self.group_id,
            "search_limit": self.search_limit,
            "retrieval_strategy": self.retrieval_strategy,
            "center_node_uuid": self.center_node_uuid,
            "candidate_count": len(self.facts),
            "fact_count": len(self.facts),
            "facts": [fact.__dict__ for fact in self.facts],
            "selected_memories": self.selected_memories,
            "rerank": self.rerank,
        }


class GraphMemoryClient:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        *,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        llm_reasoning_effort: str | None = None,
        llm_structured_output_mode: str = "json_object",
        llm_max_tokens: int = 4096,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int = 1024,
        reranker_base_url: str | None = None,
        reranker_api_key: str | None = None,
        reranker_model: str | None = None,
        build_indices: bool = True,
    ) -> None:
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
        try:
            from graphiti_core import Graphiti
        except ImportError as e:
            raise RuntimeError(
                "Graph memory mode requires Graphiti. Install it with `pip install -e '.[graph]'` "
                "and ensure Neo4j 5.26+ plus OPENAI_API_KEY are configured."
            ) from e

        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._loop = asyncio.new_event_loop()
        self._usage_recorder = _ProviderUsageRecorder()
        graphiti_kwargs = _build_graphiti_provider_kwargs(
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_reasoning_effort=llm_reasoning_effort,
            llm_structured_output_mode=llm_structured_output_mode,
            llm_max_tokens=llm_max_tokens,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            reranker_base_url=reranker_base_url,
            reranker_api_key=reranker_api_key,
            reranker_model=reranker_model,
            usage_recorder=self._usage_recorder,
        )
        self._client = Graphiti(neo4j_uri, neo4j_user, neo4j_password, **graphiti_kwargs)
        if build_indices:
            self._run(self._client.build_indices_and_constraints())

    def usage_snapshot(self) -> dict[str, Any]:
        """Best-effort snapshot of counters exposed by Graphiti providers.

        Graphiti 0.29 does not define a stable usage API.  This intentionally
        reads only conventional telemetry attributes/methods and never provider
        configuration, headers, or API keys.
        """
        snapshots: dict[str, Any] = self._usage_recorder.snapshot()
        providers = {
            "graphiti": self._client,
            "llm": getattr(self._client, "llm_client", None),
            "embedder": getattr(self._client, "embedder", None),
            "reranker": getattr(self._client, "cross_encoder", None),
        }
        for name, provider in providers.items():
            if provider is None:
                continue
            value = None
            for attr in ("get_usage", "usage", "usage_statistics", "token_usage"):
                candidate = getattr(provider, attr, None)
                try:
                    value = candidate() if callable(candidate) else candidate
                except Exception as exc:
                    value = {"telemetry_error": f"{type(exc).__name__}: {exc}"}
                if value is not None:
                    break
            if value is not None:
                snapshots[name] = _usage_json_safe(value)
        return snapshots

    def usage_events(self) -> list[dict[str, Any]]:
        return self._usage_recorder.event_snapshot()

    def close(self) -> None:
        try:
            close = getattr(self._client, "close", None)
            if close is not None:
                self._run(close())
        finally:
            if not self._loop.is_closed():
                self._loop.close()

    def insert_text(
        self,
        *,
        group_id: str,
        text: str,
        name: str,
        source_description: str,
        reference_time: datetime | None = None,
    ) -> None:
        self._run(
            self._add_episode(
                group_id=group_id,
                text=text,
                name=name,
                source_description=source_description,
                reference_time=reference_time or datetime.now(timezone.utc),
            )
        )

    def insert_texts(
        self,
        *,
        group_id: str,
        texts: list[str],
        source_description: str,
        progress: Any = None,
    ) -> tuple[int, int]:
        ok = 0
        failed = 0
        total = len(texts)
        for index, text in enumerate(texts, start=1):
            try:
                self.insert_text(
                    group_id=group_id,
                    text=text,
                    name=f"persona_memory_{index:04d}_{_short_hash(text)}",
                    source_description=source_description,
                )
                ok += 1
            except Exception as e:
                failed += 1
                if progress is not None:
                    progress(index, total, ok, failed, e)
                continue
            if progress is not None:
                progress(index, total, ok, failed, None)
        return ok, failed

    def search(self, *, group_id: str, query: str, limit: int) -> GraphRetrievalResult:
        facts = self._run(self._search(group_id=group_id, query=query, limit=limit))
        if not facts:
            facts = self._fallback_search_graph_nodes(group_id=group_id, query=query, limit=limit)
        return GraphRetrievalResult(query=query, group_id=group_id, search_limit=limit, facts=facts)

    def search_advanced(
        self,
        *,
        group_id: str,
        query: str,
        limit: int,
        persona_name: str,
    ) -> GraphRetrievalResult:
        center_node_uuid = self._find_entity_uuid(group_id=group_id, entity_name=persona_name)
        facts = self._run(
            self._search_advanced(
                group_id=group_id,
                query=query,
                limit=limit,
                center_node_uuid=center_node_uuid,
            )
        )
        if not facts:
            facts = self._run(self._search(group_id=group_id, query=query, limit=limit))
        if not facts:
            facts = self._fallback_search_graph_nodes(group_id=group_id, query=query, limit=limit)
        return GraphRetrievalResult(
            query=query,
            group_id=group_id,
            search_limit=limit,
            facts=facts,
            retrieval_strategy="combined_hybrid_rrf_bfs",
            center_node_uuid=center_node_uuid,
        )

    def list_candidates(
        self,
        *,
        group_id: str,
        limit: int,
        persona_name: str,
    ) -> GraphRetrievalResult:
        center_node_uuid = self._find_entity_uuid(group_id=group_id, entity_name=persona_name)
        facts = self._list_graph_candidates(group_id=group_id, limit=limit)
        return GraphRetrievalResult(
            query="",
            group_id=group_id,
            search_limit=limit,
            facts=facts,
            retrieval_strategy="all_graph_episodes_edges_nodes",
            center_node_uuid=center_node_uuid,
        )

    async def _add_episode(
        self,
        *,
        group_id: str,
        text: str,
        name: str,
        source_description: str,
        reference_time: datetime,
    ) -> None:
        from graphiti_core.nodes import EpisodeType

        await self._client.add_episode(
            name=name,
            episode_body=text,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=reference_time,
            group_id=group_id,
        )

    async def _search(self, *, group_id: str, query: str, limit: int) -> list[GraphMemoryFact]:
        results = await self._client.search(query, group_ids=[group_id], num_results=limit)
        facts: list[GraphMemoryFact] = []
        for item in results:
            fact = getattr(item, "fact", None)
            if not fact:
                continue
            facts.append(
                GraphMemoryFact(
                    fact=str(fact),
                    uuid=_maybe_str(getattr(item, "uuid", None)),
                    valid_at=_maybe_iso(getattr(item, "valid_at", None)),
                    invalid_at=_maybe_iso(getattr(item, "invalid_at", None)),
                    created_at=_maybe_iso(getattr(item, "created_at", None)),
                    source_node_uuid=_maybe_str(getattr(item, "source_node_uuid", None)),
                    target_node_uuid=_maybe_str(getattr(item, "target_node_uuid", None)),
                )
            )
        return facts

    async def _search_advanced(
        self,
        *,
        group_id: str,
        query: str,
        limit: int,
        center_node_uuid: str | None,
    ) -> list[GraphMemoryFact]:
        from graphiti_core.search.search_config import EdgeSearchMethod, NodeSearchMethod
        from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

        config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = limit
        if center_node_uuid is not None:
            if config.edge_config is not None and EdgeSearchMethod.bfs not in config.edge_config.search_methods:
                config.edge_config.search_methods.append(EdgeSearchMethod.bfs)
            if config.node_config is not None and NodeSearchMethod.bfs not in config.node_config.search_methods:
                config.node_config.search_methods.append(NodeSearchMethod.bfs)

        results = await self._client.search_(
            query,
            config=config,
            group_ids=[group_id],
            center_node_uuid=center_node_uuid,
            bfs_origin_node_uuids=[center_node_uuid] if center_node_uuid is not None else None,
        )

        edge_facts = [
            GraphMemoryFact(
                fact=str(edge.fact),
                uuid=_maybe_str(getattr(edge, "uuid", None)),
                valid_at=_maybe_iso(getattr(edge, "valid_at", None)),
                invalid_at=_maybe_iso(getattr(edge, "invalid_at", None)),
                created_at=_maybe_iso(getattr(edge, "created_at", None)),
                source_node_uuid=_maybe_str(getattr(edge, "source_node_uuid", None)),
                target_node_uuid=_maybe_str(getattr(edge, "target_node_uuid", None)),
                kind="edge",
                score=_score_at(results.edge_reranker_scores, index),
            )
            for index, edge in enumerate(results.edges)
            if getattr(edge, "fact", None)
        ]
        node_facts = []
        for index, node in enumerate(results.nodes):
            name = str(getattr(node, "name", "") or "").strip()
            summary = str(getattr(node, "summary", "") or "").strip()
            text = f"{name}: {summary}" if name and summary else summary or name
            if text:
                node_facts.append(
                    GraphMemoryFact(
                        fact=text,
                        uuid=_maybe_str(getattr(node, "uuid", None)),
                        created_at=_maybe_iso(getattr(node, "created_at", None)),
                        source_node_uuid=_maybe_str(getattr(node, "uuid", None)),
                        kind="node",
                        score=_score_at(results.node_reranker_scores, index),
                    )
                )

        episode_facts = []
        for index, episode in enumerate(results.episodes):
            content = str(getattr(episode, "content", "") or "").strip()
            statements = _episode_statements(content)
            for statement in statements:
                episode_facts.append(
                    GraphMemoryFact(
                        fact=statement,
                        uuid=_maybe_str(getattr(episode, "uuid", None)),
                        valid_at=_maybe_iso(getattr(episode, "valid_at", None)),
                        created_at=_maybe_iso(getattr(episode, "created_at", None)),
                        source_node_uuid=_maybe_str(getattr(episode, "uuid", None)),
                        kind="episode",
                        score=_score_at(results.episode_reranker_scores, index),
                    )
                )

        return _round_robin_dedupe([edge_facts, node_facts, episode_facts], limit)

    def _find_entity_uuid(self, *, group_id: str, entity_name: str) -> str | None:
        try:
            from neo4j import GraphDatabase
        except ImportError:
            return None

        driver = GraphDatabase.driver(self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password))
        try:
            with driver.session(database="neo4j") as session:
                record = session.run(
                    """
                    MATCH (n:Entity {group_id: $group_id})
                    WHERE toLower(n.name) = toLower($entity_name)
                    RETURN n.uuid AS uuid
                    ORDER BY n.created_at ASC
                    LIMIT 1
                    """,
                    group_id=group_id,
                    entity_name=entity_name,
                ).single()
                return _maybe_str(record.get("uuid")) if record is not None else None
        finally:
            driver.close()

    def _list_graph_candidates(self, *, group_id: str, limit: int) -> list[GraphMemoryFact]:
        try:
            from neo4j import GraphDatabase
        except ImportError:
            return []

        driver = GraphDatabase.driver(self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password))
        try:
            with driver.session(database="neo4j") as session:
                episodes = session.run(
                    """
                    MATCH (e:Episodic {group_id: $group_id})
                    RETURN e.uuid AS uuid, e.content AS content, e.created_at AS created_at,
                           e.valid_at AS valid_at
                    ORDER BY e.created_at ASC
                    """,
                    group_id=group_id,
                ).data()
                edges = session.run(
                    """
                    MATCH (source:Entity)-[r:RELATES_TO {group_id: $group_id}]->(target:Entity)
                    RETURN r.uuid AS uuid, r.fact AS fact, r.created_at AS created_at,
                           r.valid_at AS valid_at, r.invalid_at AS invalid_at,
                           source.uuid AS source_uuid, target.uuid AS target_uuid
                    ORDER BY r.created_at ASC
                    """,
                    group_id=group_id,
                ).data()
                nodes = session.run(
                    """
                    MATCH (n:Entity {group_id: $group_id})
                    RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
                           n.created_at AS created_at
                    ORDER BY n.created_at ASC
                    """,
                    group_id=group_id,
                ).data()
        finally:
            driver.close()

        candidates: list[GraphMemoryFact] = []
        for episode in episodes:
            for statement in _episode_statements(str(episode.get("content") or "")):
                candidates.append(
                    GraphMemoryFact(
                        fact=statement,
                        uuid=_maybe_str(episode.get("uuid")),
                        valid_at=_maybe_iso(episode.get("valid_at")),
                        created_at=_maybe_iso(episode.get("created_at")),
                        source_node_uuid=_maybe_str(episode.get("uuid")),
                        kind="episode",
                    )
                )
        for edge in edges:
            if edge.get("fact"):
                candidates.append(
                    GraphMemoryFact(
                        fact=str(edge["fact"]),
                        uuid=_maybe_str(edge.get("uuid")),
                        valid_at=_maybe_iso(edge.get("valid_at")),
                        invalid_at=_maybe_iso(edge.get("invalid_at")),
                        created_at=_maybe_iso(edge.get("created_at")),
                        source_node_uuid=_maybe_str(edge.get("source_uuid")),
                        target_node_uuid=_maybe_str(edge.get("target_uuid")),
                        kind="edge",
                    )
                )
        for node in nodes:
            name = str(node.get("name") or "").strip()
            summary = str(node.get("summary") or "").strip()
            text = f"{name}: {summary}" if name and summary else summary or name
            if text:
                candidates.append(
                    GraphMemoryFact(
                        fact=text,
                        uuid=_maybe_str(node.get("uuid")),
                        created_at=_maybe_iso(node.get("created_at")),
                        source_node_uuid=_maybe_str(node.get("uuid")),
                        kind="node",
                    )
                )
        return _dedupe_facts(candidates, limit)

    def _run(self, value: Any) -> Any:
        if not inspect.isawaitable(value):
            return value
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._loop.run_until_complete(value)
        raise RuntimeError("GraphMemoryClient sync methods cannot be called from an active event loop.")

    def _fallback_search_graph_nodes(self, *, group_id: str, query: str, limit: int) -> list[GraphMemoryFact]:
        try:
            from neo4j import GraphDatabase
        except ImportError:
            return []

        records: list[dict[str, Any]] = []
        driver = GraphDatabase.driver(self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password))
        try:
            with driver.session(database="neo4j") as session:
                records.extend(
                    session.run(
                        """
                        MATCH (e:Episodic {group_id: $group_id})
                        RETURN e.uuid AS uuid, e.content AS text, e.created_at AS created_at,
                               e.valid_at AS valid_at, "episodic" AS source
                        LIMIT 500
                        """,
                        group_id=group_id,
                    ).data()
                )
                records.extend(
                    session.run(
                        """
                        MATCH (n:Entity {group_id: $group_id})
                        RETURN n.uuid AS uuid,
                               CASE
                                 WHEN n.summary IS NULL OR n.summary = "" THEN n.name
                                 ELSE n.name + ": " + n.summary
                               END AS text,
                               n.created_at AS created_at,
                               NULL AS valid_at,
                               "entity" AS source
                        LIMIT 500
                        """,
                        group_id=group_id,
                    ).data()
                )
        finally:
            driver.close()

        query_tokens = _tokenize(query)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, record in enumerate(records):
            text = record.get("text")
            if not text:
                continue
            text_tokens = _tokenize(str(text))
            overlap = len(query_tokens & text_tokens)
            priority = _task_fact_priority(str(text))
            score = float(overlap) + priority
            scored.append((score, index, record))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [record for score, _, record in scored if score > 0][:limit]
        if not selected:
            selected = [record for _, _, record in scored[:limit]]

        return [
            GraphMemoryFact(
                fact=str(record["text"]),
                uuid=_maybe_str(record.get("uuid")),
                valid_at=_maybe_iso(record.get("valid_at")),
                created_at=_maybe_iso(record.get("created_at")),
                source_node_uuid=_maybe_str(record.get("source")),
            )
            for record in selected
            if record.get("text")
        ]


def _build_graphiti_provider_kwargs(
    *,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
    llm_reasoning_effort: str | None,
    llm_structured_output_mode: str,
    llm_max_tokens: int,
    embedding_base_url: str | None,
    embedding_api_key: str | None,
    embedding_model: str | None,
    embedding_dim: int,
    reranker_base_url: str | None,
    reranker_api_key: str | None,
    reranker_model: str | None,
    usage_recorder: "_ProviderUsageRecorder | None" = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if llm_base_url or llm_model:
        if not llm_base_url or not llm_model:
            raise RuntimeError("GRAPHITI_LLM_BASE_URL and GRAPHITI_LLM_MODEL must both be set for local Graphiti LLM support.")
        try:
            from graphiti_core.llm_client.config import LLMConfig
        except ImportError as e:
            raise RuntimeError("Installed graphiti_core version does not expose LLMConfig.") from e
        llm_config = LLMConfig(
            api_key=llm_api_key or "local",
            model=llm_model,
            small_model=llm_model,
            base_url=llm_base_url.rstrip("/"),
            temperature=0,
            max_tokens=llm_max_tokens,
        )
        metered_llm_client = _metered_openai_client(
            api_key=llm_api_key or "local",
            base_url=llm_base_url.rstrip("/"),
            recorder=usage_recorder,
            component="llm",
        )
        if _is_official_openai_base_url(llm_base_url):
            try:
                from graphiti_core.llm_client.openai_client import OpenAIClient
            except ImportError as e:
                raise RuntimeError("Installed graphiti_core version does not expose OpenAIClient.") from e
            openai_kwargs: dict[str, Any] = {
                "config": llm_config,
                "max_tokens": llm_max_tokens,
            }
            reasoning_effort = llm_reasoning_effort
            if not reasoning_effort and llm_model.startswith("gpt-5.6"):
                reasoning_effort = "medium"
            if reasoning_effort:
                openai_kwargs["reasoning"] = reasoning_effort
            openai_kwargs["client"] = metered_llm_client
            kwargs["llm_client"] = OpenAIClient(**openai_kwargs)
        else:
            try:
                from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
            except ImportError as e:
                raise RuntimeError("Installed graphiti_core version does not expose OpenAIGenericClient.") from e
            generic_kwargs = {
                "config": llm_config,
                "client": metered_llm_client,
                "max_tokens": llm_max_tokens,
                "structured_output_mode": _validated_structured_output_mode(
                    llm_structured_output_mode
                ),
            }
            if llm_reasoning_effort and llm_reasoning_effort.lower() != "none":
                kwargs["llm_client"] = _reasoning_compatible_graphiti_client(
                    OpenAIGenericClient,
                    reasoning_effort=llm_reasoning_effort,
                    **generic_kwargs,
                )
            else:
                # Preserve Graphiti's original generic-client behavior when no reasoning
                # effort is requested. This is the explicit legacy/reproduction path.
                kwargs["llm_client"] = OpenAIGenericClient(**generic_kwargs)

    if embedding_base_url or embedding_model:
        if not embedding_base_url or not embedding_model:
            raise RuntimeError(
                "GRAPHITI_EMBEDDING_BASE_URL and GRAPHITI_EMBEDDING_MODEL must both be set for local Graphiti embeddings."
            )
        try:
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        except ImportError as e:
            raise RuntimeError("Installed graphiti_core version does not expose OpenAIEmbedder.") from e
        embedder_config = OpenAIEmbedderConfig(
            api_key=embedding_api_key or "local",
            base_url=embedding_base_url.rstrip("/"),
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
        kwargs["embedder"] = OpenAIEmbedder(
            client=_metered_openai_client(
                api_key=embedding_api_key or "local",
                base_url=embedding_base_url.rstrip("/"),
                recorder=usage_recorder,
                component="embedder",
            ),
            config=embedder_config,
        )

    if reranker_base_url or reranker_model:
        if not reranker_base_url or not reranker_model:
            raise RuntimeError(
                "GRAPHITI_RERANKER_BASE_URL and GRAPHITI_RERANKER_MODEL must both be set for local Graphiti reranking."
            )
        try:
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            from graphiti_core.llm_client.config import LLMConfig
        except ImportError as e:
            raise RuntimeError("Installed graphiti_core version does not expose OpenAIRerankerClient.") from e
        kwargs["cross_encoder"] = OpenAIRerankerClient(
            config=LLMConfig(
                api_key=reranker_api_key or "local",
                model=reranker_model,
                base_url=reranker_base_url.rstrip("/"),
                temperature=0,
                max_tokens=1,
            ),
            client=_metered_openai_client(
                api_key=reranker_api_key or "local",
                base_url=reranker_base_url.rstrip("/"),
                recorder=usage_recorder,
                component="reranker",
            ),
        )

    return kwargs


def _save_graphiti_failed_response(response: Any, metadata: dict[str, Any]) -> Path:
    """Persist provider output and safe request metadata, never headers or credentials."""
    directory = Path(
        os.getenv("GRAPHITI_LLM_ERROR_DIR", "research_outputs/graphiti-error-responses")
    ).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    response_payload = _usage_json_safe(response)
    response_id = "no-id"
    if isinstance(response_payload, dict):
        response_id = str(response_payload.get("id") or response_id)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", response_id).strip("-.") or "no-id"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = directory / f"graphiti-response_{timestamp}_{safe_id[:100]}.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"schema_version": 1, "metadata": metadata, "response": response_payload},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path.resolve()


def _reasoning_compatible_graphiti_client(
    client_class: type,
    *,
    reasoning_effort: str,
    **client_kwargs: Any,
) -> Any:
    """Add reasoning-effort forwarding and actionable failures to Graphiti's generic client.

    Kept as a runtime subclass because Graphiti is an optional dependency. The provider
    response is saved only on parsing/empty-content failures, and request messages,
    headers, and API keys are deliberately excluded.
    """
    import openai
    from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS
    from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError

    class ReasoningCompatibleOpenAIGenericClient(client_class):
        letta_reasoning_effort = reasoning_effort

        async def _generate_response(
            self,
            messages: list[Any],
            response_model: type | None = None,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model_size: Any = None,
        ) -> dict[str, Any]:
            del model_size
            openai_messages: list[dict[str, str]] = []
            for message in messages:
                content = self._clean_input(message.content)
                if message.role in {"user", "system"}:
                    openai_messages.append({"role": message.role, "content": content})
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "response_format": self._build_response_format(response_model),
                "reasoning_effort": reasoning_effort,
            }
            try:
                response = await self.client.chat.completions.create(**payload)
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                normalized = self._strip_code_fences(content)
                metadata = {
                    "model": self.model,
                    "response_model": getattr(response_model, "__name__", None),
                    "reasoning_effort": reasoning_effort,
                    "max_tokens": max_tokens,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "content_present": bool(normalized),
                    "content_characters": len(content),
                    "reasoning_content_present": bool(
                        getattr(message, "reasoning_content", None)
                    ),
                }
                if not normalized:
                    path = _save_graphiti_failed_response(response, metadata)
                    raise EmptyResponseError(
                        "Graphiti LLM returned no parseable visible content "
                        f"(finish_reason={metadata['finish_reason']}; raw_response={path})"
                    )
                try:
                    parsed = json.loads(normalized)
                except json.JSONDecodeError as exc:
                    path = _save_graphiti_failed_response(response, metadata)
                    raise json.JSONDecodeError(
                        f"{exc.msg}; raw_response={path}", exc.doc, exc.pos
                    ) from exc
                if not isinstance(parsed, dict):
                    path = _save_graphiti_failed_response(response, metadata)
                    raise ValueError(
                        "Graphiti LLM response was not a JSON object "
                        f"(raw_response={path})"
                    )
                if response_model is not None:
                    try:
                        response_model.model_validate(parsed)
                    except Exception as exc:
                        metadata["validation_error_type"] = type(exc).__name__
                        path = _save_graphiti_failed_response(response, metadata)
                        raise EmptyResponseError(
                            "Graphiti LLM response failed structured validation "
                            f"({type(exc).__name__}; raw_response={path})"
                        ) from exc
                return parsed
            except openai.RateLimitError as exc:
                raise RateLimitError from exc
            except Exception:
                raise

    return ReasoningCompatibleOpenAIGenericClient(**client_kwargs)


class _ProviderUsageRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, component: str, operation: str, response: Any, request: dict[str, Any]) -> None:
        payload = _usage_json_safe(response)
        usage = _find_usage_payload(payload)
        self.events.append(
            {
                "component": component,
                "operation": operation,
                "model": request.get("model"),
                "usage": usage,
                "usage_available": usage is not None,
            }
        )

    def record_error(self, component: str, operation: str, error: Exception, request: dict[str, Any]) -> None:
        self.events.append(
            {
                "component": component,
                "operation": operation,
                "model": request.get("model"),
                "usage": None,
                "usage_available": False,
                "error_type": type(error).__name__,
            }
        )

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {"event_count": len(self.events)}
        for component in ("llm", "embedder", "reranker"):
            events = [event for event in self.events if event["component"] == component]
            usages = [event["usage"] for event in events if isinstance(event.get("usage"), dict)]
            result[component] = {
                **_sum_provider_usages(usages),
                "calls": len(events),
                "usage_calls": len(usages),
            }
        return result

    def event_snapshot(self) -> list[dict[str, Any]]:
        return list(self.events)


class _MeteredEndpoint:
    def __init__(self, target: Any, recorder: _ProviderUsageRecorder, component: str, operation: str) -> None:
        self._target = target
        self._recorder = recorder
        self._component = component
        self._operation = operation

    async def create(self, **kwargs: Any) -> Any:
        try:
            response = await self._target.create(**kwargs)
        except Exception as exc:
            self._recorder.record_error(self._component, self._operation, exc, kwargs)
            raise
        self._recorder.record(self._component, self._operation, response, kwargs)
        return response

    async def parse(self, **kwargs: Any) -> Any:
        try:
            response = await self._target.parse(**kwargs)
        except Exception as exc:
            self._recorder.record_error(self._component, f"{self._operation}.parse", exc, kwargs)
            raise
        self._recorder.record(self._component, f"{self._operation}.parse", response, kwargs)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _MeteredOpenAIClient:
    def __init__(self, client: Any, recorder: _ProviderUsageRecorder, component: str) -> None:
        self._client = client
        self.chat = SimpleNamespace(
            completions=_MeteredEndpoint(client.chat.completions, recorder, component, "chat.completions")
        )
        self.responses = _MeteredEndpoint(client.responses, recorder, component, "responses")
        self.embeddings = _MeteredEndpoint(client.embeddings, recorder, component, "embeddings")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _metered_openai_client(
    *, api_key: str, base_url: str, recorder: _ProviderUsageRecorder | None, component: str
) -> Any:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _MeteredOpenAIClient(client, recorder, component) if recorder is not None else client


def _find_usage_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("usage", "usage_statistics"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    for value in payload.values():
        found = _find_usage_payload(value)
        if found is not None:
            return found
    return None


def _usage_number(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _sum_provider_usages(usages: list[dict[str, Any]]) -> dict[str, Any]:
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    result: dict[str, Any] = {}
    for output_key, input_keys in aliases.items():
        values = []
        for usage in usages:
            value = next((_usage_number(usage.get(key)) for key in input_keys if _usage_number(usage.get(key)) is not None), None)
            if value is not None:
                values.append(value)
        result[output_key] = sum(values) if values else None
    cached_values = []
    reasoning_values = []
    for usage in usages:
        input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
        output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details")
        cached = _usage_number(input_details.get("cached_tokens")) if isinstance(input_details, dict) else None
        reasoning = _usage_number(output_details.get("reasoning_tokens")) if isinstance(output_details, dict) else None
        if cached is not None:
            cached_values.append(cached)
        if reasoning is not None:
            reasoning_values.append(reasoning)
    result["cached_input_tokens"] = sum(cached_values) if cached_values else None
    result["reasoning_output_tokens"] = sum(reasoning_values) if reasoning_values else None
    return result


def _validated_structured_output_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"json_schema", "json_object"}:
        raise RuntimeError("GRAPHITI_LLM_STRUCTURED_OUTPUT_MODE must be 'json_schema' or 'json_object'.")
    return normalized


def _is_official_openai_base_url(value: str) -> bool:
    return (urlparse(value).hostname or "").lower() == "api.openai.com"


def _maybe_iso(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _maybe_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _usage_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _usage_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_usage_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _usage_json_safe(value.model_dump())
    if hasattr(value, "dict"):
        try:
            return _usage_json_safe(value.dict())
        except Exception:
            pass
    return str(value)


def _score_at(scores: list[float], index: int) -> float | None:
    return float(scores[index]) if index < len(scores) else None


def _episode_statements(content: str) -> list[str]:
    if not content:
        return []
    bullets = [line[2:].strip() for line in content.splitlines() if line.strip().startswith("- ")]
    return bullets or [content]


def _round_robin_dedupe(groups: list[list[GraphMemoryFact]], limit: int) -> list[GraphMemoryFact]:
    selected: list[GraphMemoryFact] = []
    seen: set[str] = set()
    max_length = max((len(group) for group in groups), default=0)
    for index in range(max_length):
        for group in groups:
            if index >= len(group):
                continue
            fact = group[index]
            key = " ".join(fact.fact.lower().split())
            if key in seen:
                continue
            seen.add(key)
            selected.append(fact)
            if len(selected) >= limit:
                return selected
    return selected


def _dedupe_facts(facts: list[GraphMemoryFact], limit: int) -> list[GraphMemoryFact]:
    selected: list[GraphMemoryFact] = []
    seen: set[str] = set()
    for fact in facts:
        key = " ".join(fact.fact.lower().split())
        if key in seen:
            continue
        seen.add(key)
        selected.append(fact)
        if len(selected) >= limit:
            break
    return selected


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _tokenize(text: str) -> set[str]:
    stopwords = {
        "a", "about", "and", "are", "as", "at", "be", "for", "from", "i", "in", "is", "it",
        "my", "of", "on", "or", "the", "this", "to", "user", "with", "write",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9$]+", text.lower())
        if len(token) > 1 and token not in stopwords
    }


def _task_fact_priority(text: str) -> float:
    lowered = text.lower()
    priority_terms = (
        "name", "address", "income", "work", "employer", "job", "title", "married",
        "mortgage", "loan", "debt", "credit", "salary", "employment",
    )
    return sum(0.25 for term in priority_terms if term in lowered)
