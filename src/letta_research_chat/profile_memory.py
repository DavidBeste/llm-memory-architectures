from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from importlib import metadata
from typing import Any
from uuid import UUID

# Memobase extracts profiles asynchronously in a background worker. Reading the
# profile as soon as insert/flush returns yields a partial profile - measured at
# 0-48% of the eventual entry count, and 0 of 123 entries in the worst observed
# case. Poll until the entry count holds steady instead.
DEFAULT_SETTLE_TIMEOUT_S = 1800
DEFAULT_SETTLE_INTERVAL_S = 20
DEFAULT_SETTLE_STABLE_CHECKS = 3

# Budget used when the whole profile is wanted rather than a prompt-sized slice.
FULL_PROFILE_TOKEN_SIZE = 64000


@dataclass(frozen=True)
class ProfileMemoryInitResult:
    backend: str
    user_id: str
    inserted_count: int
    flush_sync: bool
    profile: Any = None
    entries_at_flush: int | None = None
    entries_settled: int | None = None
    settle_wait_seconds: int = 0
    settle_converged: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "user_id": self.user_id,
            "inserted_count": self.inserted_count,
            "flush_sync": self.flush_sync,
            "entries_at_flush": self.entries_at_flush,
            "entries_settled": self.entries_settled,
            "settle_wait_seconds": self.settle_wait_seconds,
            "settle_converged": self.settle_converged,
            "profile": json_safe(self.profile),
        }


@dataclass(frozen=True)
class ProfileMemoryResult:
    backend: str
    user_id: str
    query: str
    context: str
    max_token_size: int | None
    rendering_mode: str = "memobase_context"
    profile: Any = None
    retrieval_options: dict[str, Any] | None = None
    # Whether `query` actually reached the backend. Memobase only conditions
    # retrieval on `chats`; recording a query that was never sent made saved
    # manifests imply task-conditioned retrieval that did not happen.
    query_applied: bool = False
    candidates: list[dict[str, Any]] | None = None
    selected_memories: list[str] | None = None
    rerank: dict[str, Any] | None = None
    # Native Memobase-rendered context before any optional local postprocessing.
    source_context: str | None = None
    # Architecture-level memory visible before and after local reranking. These
    # are separate from ``candidates`` because a persistent profile may bypass
    # the event reranker while still entering the downstream prompt.
    pre_rerank_memory_items: list[dict[str, Any]] | None = None
    post_rerank_memory_items: list[dict[str, Any]] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "user_id": self.user_id,
            "query": self.query,
            "query_applied": self.query_applied,
            "context": self.context,
            "max_token_size": self.max_token_size,
            "rendering_mode": self.rendering_mode,
            "profile": json_safe(self.profile),
            "retrieval_options": json_safe(self.retrieval_options),
            "candidates": json_safe(self.candidates),
            "selected_memories": json_safe(self.selected_memories),
            "rerank": json_safe(self.rerank),
            "source_context": self.source_context,
            "pre_rerank_memory_items": json_safe(self.pre_rerank_memory_items),
            "post_rerank_memory_items": json_safe(self.post_rerank_memory_items),
        }


class ProfileMemoryClient:
    backend = "memobase"

    def __init__(self, project_url: str, api_key: str) -> None:
        try:
            from memobase import ChatBlob, MemoBaseClient
        except ImportError as exc:
            raise RuntimeError(
                "Profile memory mode requires Memobase. Install it with "
                "`pip install -e '.[profile]'` and configure MEMOBASE_PROJECT_URL "
                "plus MEMOBASE_API_KEY."
            ) from exc

        self._chat_blob_cls = ChatBlob
        self._client = MemoBaseClient(project_url=project_url, api_key=api_key)

    def ping(self) -> bool:
        ping = getattr(self._client, "ping", None)
        if ping is None:
            return True
        return bool(ping())

    def get_config(self) -> str | None:
        """Return the project's current profile config, or None if unsupported.

        Used to snapshot state before a mode overwrites it, and to record the
        deployment's extraction schema in run manifests.
        """
        return self.inspect_config()["config"]

    def inspect_config(self) -> dict[str, Any]:
        """Diagnose the optional project-config API without raising.

        Memobase client/server versions differ, and older deployments may not
        expose this API at all.  Keep errors observable for the CLI while
        preserving ``get_config()``'s best-effort behavior for experiment runs.
        """
        client_type = type(self._client)
        try:
            sdk_version = metadata.version("memobase")
        except metadata.PackageNotFoundError:
            sdk_version = "unknown"
        get_config = getattr(self._client, "get_config", None)
        update_config = getattr(self._client, "update_config", None)
        result: dict[str, Any] = {
            "sdk_version": sdk_version,
            "client_class": f"{client_type.__module__}.{client_type.__qualname__}",
            "get_config_supported": callable(get_config),
            "update_config_supported": callable(update_config),
            "status": "unsupported",
            "config": None,
            "error_type": None,
            "error_message": None,
        }
        if not callable(get_config):
            return result
        try:
            config = get_config()
        except Exception as exc:
            result.update(
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return result
        if not config:
            result["status"] = "empty"
            return result
        result.update(status="available", config=str(config))
        return result

    def create_user(self, metadata: dict[str, Any]) -> str:
        user_id = self._client.add_user(metadata)
        return str(user_id)

    def get_user(self, user_id: str) -> Any:
        return self._client.get_user(user_id)

    def update_config(self, config: str) -> Any:
        update = getattr(self._client, "update_config", None)
        if update is None:
            raise RuntimeError("Installed Memobase client does not support update_config(...).")
        return update(config)

    def usage_snapshot(self) -> Any:
        """Return backend billing telemetry when supported by this SDK/server.

        Memobase versions differ: some expose get_usage on the project client,
        while older/self-hosted deployments expose no usage endpoint.  Callers
        must treat None as unavailable rather than zero.
        """
        get_usage = getattr(self._client, "get_usage", None)
        if not callable(get_usage):
            return None
        try:
            return json_safe(get_usage())
        except Exception as exc:
            return {"telemetry_error": f"{type(exc).__name__}: {exc}"}

    def insert_memory_statements(
        self,
        user_id: str,
        statements: list[str],
        *,
        sync_flush: bool,
        wait_for_settle: bool = True,
        settle_timeout_s: int = DEFAULT_SETTLE_TIMEOUT_S,
        settle_interval_s: int = DEFAULT_SETTLE_INTERVAL_S,
        settle_stable_checks: int = DEFAULT_SETTLE_STABLE_CHECKS,
        progress: Any = None,
    ) -> ProfileMemoryInitResult:
        user = self.get_user(user_id)
        inserted = 0
        for statement in statements:
            text = statement.strip()
            if not text:
                continue
            blob = self._chat_blob_cls(messages=[{"role": "user", "content": text}])
            user.insert(blob)
            inserted += 1
        flush = getattr(user, "flush", None)
        if flush is not None:
            flush(sync=sync_flush)

        # Baseline taken immediately after flush: the gap between this and
        # entries_settled is exactly what the pre-fix code silently discarded.
        at_flush = self._entry_count(user)
        waited = 0
        converged = True
        if wait_for_settle:
            waited, converged = self._wait_for_settle(
                user,
                timeout_s=settle_timeout_s,
                interval_s=settle_interval_s,
                stable_checks=settle_stable_checks,
                progress=progress,
            )

        # Record the untruncated profile: at the SDK's 1000-token default the
        # stored payload and entries_settled would disagree with the settle poll.
        profile = self._profile_payload(user, max_token_size=FULL_PROFILE_TOKEN_SIZE)
        return ProfileMemoryInitResult(
            backend=self.backend,
            user_id=user_id,
            inserted_count=inserted,
            flush_sync=sync_flush,
            profile=profile,
            entries_at_flush=at_flush,
            entries_settled=self._count_entries(profile),
            settle_wait_seconds=waited,
            settle_converged=converged,
        )

    def _wait_for_settle(
        self,
        user: Any,
        *,
        timeout_s: int,
        interval_s: int,
        stable_checks: int,
        progress: Any = None,
    ) -> tuple[int, bool]:
        """Poll until the extracted profile stops growing.

        `flush(sync=True)` only covers blobs still sitting in the buffer; blobs
        the server already picked up keep processing in a background worker, so
        an empty buffer is not a completion signal. The entry count is, once it
        holds steady across `stable_checks` polls.
        """
        deadline = time.time() + timeout_s
        waited = 0
        last = -1
        stable = 0
        while time.time() < deadline:
            current = self._entry_count(user)
            stable = stable + 1 if current == last else 0
            last = current
            if progress is not None:
                progress(waited, current, stable)
            if stable >= stable_checks and current > 0:
                return waited, True
            time.sleep(interval_s)
            waited += interval_s
        return waited, False

    def _entry_count(self, user: Any) -> int:
        # Count against the untruncated profile: at the SDK's 1000-token default
        # the count would plateau at the truncation boundary and read as settled.
        return self._count_entries(
            self._profile_payload(user, max_token_size=FULL_PROFILE_TOKEN_SIZE)
        )

    @staticmethod
    def _count_entries(payload: Any) -> int:
        if not isinstance(payload, dict):
            return 0
        return sum(len(value) for value in payload.values() if isinstance(value, dict))

    def context(
        self,
        user_id: str,
        *,
        query: str,
        max_token_size: int,
        chats: list[dict[str, Any]] | None = None,
        event_similarity_threshold: float | None = None,
        fill_window_with_events: bool | None = None,
        rendering_mode: str = "memobase_context",
    ) -> ProfileMemoryResult:
        user = self.get_user(user_id)
        kwargs: dict[str, Any] = {"max_token_size": max_token_size}
        if chats is not None:
            kwargs["chats"] = chats
        if event_similarity_threshold is not None:
            kwargs["event_similarity_threshold"] = event_similarity_threshold
        if fill_window_with_events is not None:
            kwargs["fill_window_with_events"] = fill_window_with_events
        context = user.context(**kwargs)
        return ProfileMemoryResult(
            backend=self.backend,
            user_id=user_id,
            query=query,
            context=str(context or ""),
            max_token_size=max_token_size,
            rendering_mode=rendering_mode,
            retrieval_options=kwargs,
            query_applied=chats is not None,
        )

    def profile(self, user_id: str, *, max_token_size: int | None = None) -> Any:
        """Return the extracted profile JSON.

        `max_token_size` must be threaded through: the SDK defaults to 1000,
        which truncates server-side before any caller-side ranking can see the
        full profile.
        """
        user = self.get_user(user_id)
        return self._profile_payload(user, max_token_size=max_token_size)

    @staticmethod
    def _profile_payload(user: Any, *, max_token_size: int | None = None) -> Any:
        profile = getattr(user, "profile", None)
        if profile is None:
            return None
        kwargs: dict[str, Any] = {"need_json": True}
        if max_token_size is not None:
            kwargs["max_token_size"] = max_token_size
        try:
            return profile(**kwargs)
        except TypeError:
            try:
                return profile(need_json=True)
            except TypeError:
                return profile()


def build_profile_memory_query(recipient: str, task: str) -> str:
    return f"profile memory context needed to write to {recipient} about: {task}"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(json_safe(key)): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if hasattr(value, "dict"):
        try:
            return json_safe(value.dict())
        except Exception:
            pass
    return str(value)


def format_profile_memory_prompt(recipient: str, task: str, result: ProfileMemoryResult) -> str:
    context = result.context.strip() or "(no profile memory context returned)"
    return f"""# Profile-Based User Memory

The following profile memory context came from Memobase, not Letta archival memory
and not Graphiti graph memory. Use only facts necessary for the requested message.

{context}

Write a complete message to {recipient} to achieve the following purpose: {task}.
Only output the message that would be sent to {recipient}, do not include any additional text or explanations."""
