from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


TEXT_SUFFIXES = {
    ".csv",
    ".graphml",
    ".html",
    ".json",
    ".jsonl",
    ".md",
    ".svg",
    ".tex",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

EXPORT_SCOPES = ("github", "paper", "full")
ARTIFACT_SOFTWARE_RELEASE = "letta-research-chat-0.1.0"

REMOVED_KEYS = {
    "agent_id",
    "call_id",
    "checkpoint_fingerprint",
    "conversation_id",
    "created",
    "created_at",
    "credential_configured",
    "credential_source",
    "experiment_agent",
    "experiment_agent_id",
    "experiment_id",
    "group_id",
    "item_id",
    "message_id",
    "organization_id",
    "otid",
    "previous_response_id",
    "project_id",
    "prompt_cache_key",
    "request_id",
    "response_id",
    "run_id",
    "safety_identifier",
    "sender_id",
    "seq_id",
    "source_agent_id",
    "source_experiment_id",
    "step_id",
    "system_fingerprint",
    "tool_call_id",
    "trace_id",
    "updated_at",
    "user_id",
}

SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:api[_-]?key|authorization|credential|password|private[_-]?key|secret)(?:$|_)",
    re.IGNORECASE,
)
FIELD_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
FINGERPRINT_KEY_RE = re.compile(r"(?:^|_)fingerprint$", re.IGNORECASE)
TIMESTAMP_KEY_RE = re.compile(
    r"^(?:created|created_at|creation_time|finished_at|started_at|timestamp|updated_at)$",
    re.IGNORECASE,
)

UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
LETTA_ID_RE = re.compile(
    r"\b(?:agent|conv|run)-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
OPENAI_RESPONSE_ID_RE = re.compile(r"\bresp_[A-Za-z0-9_-]{16,}\b")
TOGETHER_RESPONSE_ID_RE = re.compile(r"\b[A-Za-z0-9_-]{8,}-aws_ec1\b")
RUN_TIMESTAMP_RE = re.compile(
    r"(?<![0-9])20[0-9]{6}(?:T|_)[0-9]{6}(?:_[0-9]{3,6}Z?)?(?![0-9])"
)
CONFIG_HASH_RE = re.compile(r"(?<=_)[0-9a-f]{10}(?=(?:_|\.))", re.IGNORECASE)
LONG_HEX_RE = re.compile(r"\b[0-9a-f]{16,64}\b", re.IGNORECASE)
OPAQUE_ID_VALUE_RE = re.compile(r"^[A-Za-z0-9_-]{16,}$")
ABSOLUTE_HOME_RE = re.compile(r"(?<![A-Za-z0-9._-])/(?:home|Users)/[^\s\"'<>]+")
SYSTEM_COMPILED_RE = re.compile(
    r"(?im)^([ \t]*-?[ \t]*System prompt last recompiled:)[^\r\n]*$"
)

SECRET_PATTERNS = {
    "openai_api_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
    "together_api_key": re.compile(r"\btgp_v1_[A-Za-z0-9_-]{16,}\b"),
    "bearer_token": re.compile(r"(?i)\bBearer[ \t]+[A-Za-z0-9._~-]{16,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "private_key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
}


@dataclass(frozen=True)
class CopyPlan:
    source: Path
    destination: Path
    label: str


@dataclass(frozen=True)
class PlannedFile:
    source: Path
    destination: Path
    label: str


@dataclass(frozen=True)
class TokenAliases:
    run_timestamps: dict[str, str]
    configuration_hashes: dict[str, str]
    opaque_hex: dict[str, str]
    uuids: dict[str, str]
    provider_ids: dict[str, str]

    def rewrite(self, text: str) -> str:
        for mapping in (
            self.run_timestamps,
            self.configuration_hashes,
            self.opaque_hex,
            self.uuids,
            self.provider_ids,
        ):
            for source, destination in mapping.items():
                text = text.replace(source, destination)
        return text


class PathRewriter:
    def __init__(
        self,
        replacements: dict[str, str],
        repository_root: Path,
        token_aliases: TokenAliases | None = None,
    ):
        normalized: dict[str, str] = {}
        for source, destination in replacements.items():
            if source:
                normalized[source.rstrip("/")] = destination.rstrip("/")
        ordered = sorted(normalized, key=len, reverse=True)
        self._replacements = normalized
        self._pattern = (
            re.compile("|".join(re.escape(item) for item in ordered))
            if ordered
            else None
        )
        self._repository_root = str(repository_root.resolve()).rstrip("/")
        self._token_aliases = token_aliases

    def rewrite(self, text: str) -> str:
        if self._pattern is not None:
            text = self._pattern.sub(
                lambda match: self._replacements[match.group(0)], text
            )
        text = text.replace(self._repository_root + "/", "repository://")
        if text == self._repository_root:
            text = "repository://"
        if self._token_aliases is not None:
            text = self._token_aliases.rewrite(text)
        return text


class Sanitizer:
    def __init__(self, path_rewriter: PathRewriter):
        self.path_rewriter = path_rewriter
        self.counts: Counter[str] = Counter()

    def sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if self._remove_key(str(key)):
                    self.counts[f"removed_key:{key}"] += 1
                    continue
                if str(key).lower() == "id" and self._is_operational_id(item):
                    self.counts["removed_key:id"] += 1
                    continue
                result[key] = self.sanitize_value(item)
            return result
        if isinstance(value, list):
            return [self.sanitize_value(item) for item in value]
        if isinstance(value, str):
            return self.sanitize_text(value)
        return value

    def sanitize_text(self, text: str) -> str:
        original = text
        text = self.path_rewriter.rewrite(text)
        if text != original:
            self.counts["rewritten_paths"] += 1

        text = self._sub(LETTA_ID_RE, "<OPERATIONAL_ID_REMOVED>", text, "letta_ids")
        text = self._sub(UUID_RE, "<UUID_REMOVED>", text, "uuids")
        text = self._sub(
            OPENAI_RESPONSE_ID_RE,
            "<PROVIDER_RESPONSE_ID_REMOVED>",
            text,
            "provider_response_ids",
        )
        text = self._sub(
            TOGETHER_RESPONSE_ID_RE,
            "<PROVIDER_RESPONSE_ID_REMOVED>",
            text,
            "provider_response_ids",
        )
        text = self._sub(
            SYSTEM_COMPILED_RE,
            r"\1 <TIMESTAMP_REMOVED>",
            text,
            "system_compilation_timestamps",
        )
        text = self._sub(
            RUN_TIMESTAMP_RE, "<RUN_TIMESTAMP_REMOVED>", text, "run_timestamps"
        )
        text = self._sub(
            CONFIG_HASH_RE, "config", text, "configuration_fingerprints"
        )

        for label, pattern in SECRET_PATTERNS.items():
            text = self._sub(pattern, "<SECRET_REMOVED>", text, f"secrets:{label}")

        text = self._sub(
            ABSOLUTE_HOME_RE, "<ABSOLUTE_PATH_REMOVED>", text, "absolute_paths"
        )
        return text

    def _remove_key(self, key: str) -> bool:
        return _is_removed_key(key)

    @staticmethod
    def _is_operational_id(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return any(
            pattern.search(value) is not None
            for pattern in (
                LETTA_ID_RE,
                UUID_RE,
                OPENAI_RESPONSE_ID_RE,
                TOGETHER_RESPONSE_ID_RE,
            )
        ) or OPAQUE_ID_VALUE_RE.fullmatch(value) is not None

    def _sub(
        self, pattern: re.Pattern[str], replacement: str, text: str, label: str
    ) -> str:
        text, count = pattern.subn(replacement, text)
        if count:
            self.counts[label] += count
        return text


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "unknown"


def _is_removed_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in REMOVED_KEYS
        or (
            FIELD_KEY_RE.fullmatch(lowered) is not None
            and SECRET_KEY_RE.search(lowered) is not None
        )
        or FINGERPRINT_KEY_RE.search(lowered) is not None
        or TIMESTAMP_KEY_RE.search(lowered) is not None
    )


def _resolve_inputs(path: Path) -> tuple[Path, Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        candidate = resolved / "integrated_artifact_inputs.json"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"No integrated_artifact_inputs.json in report directory: {resolved}"
            )
        return candidate, resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"Locked input file does not exist: {resolved}")
    return resolved, resolved.parent


def _load_selected_runs(inputs_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(inputs_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_runs")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"No selected_runs in {inputs_path}")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for raw in selected:
        if not isinstance(raw, dict):
            raise ValueError("Every selected_runs entry must be an object")
        model = str(raw.get("model") or "")
        architecture = str(raw.get("architecture") or "")
        persona_idx = int(raw["persona_idx"])
        key = (model, architecture, persona_idx)
        if not model or not architecture:
            raise ValueError(f"Incomplete selected run key: {key}")
        if key in seen:
            raise ValueError(f"Duplicate selected run: {key}")
        seen.add(key)
        result.append(raw)
    return sorted(
        result,
        key=lambda row: (
            str(row["model"]),
            str(row["architecture"]),
            int(row["persona_idx"]),
        ),
    )


def _main_run_root(pipeline_manifest: Path) -> Path:
    parent = pipeline_manifest.parent
    return parent.parent if parent.name == "pipeline" else parent


def _relative_to_any(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _copy_plans(
    selected_runs: list[dict[str, Any]], report_root: Path
) -> tuple[list[CopyPlan], list[dict[str, Any]]]:
    plans: list[CopyPlan] = []
    public_runs: list[dict[str, Any]] = []
    shared_datasets: set[Path] = set()
    shared_labels: set[Path] = set()
    for run in selected_runs:
        model = str(run["model"])
        architecture = str(run["architecture"])
        persona_idx = int(run["persona_idx"])
        destination = Path("pipelines") / _slug(model) / _slug(architecture) / (
            f"persona_{persona_idx:03d}"
        )
        pipeline_manifest = Path(str(run["pipeline_manifest"])).expanduser().resolve()
        if not pipeline_manifest.is_file():
            raise FileNotFoundError(f"Pipeline manifest is missing: {pipeline_manifest}")
        source_root = _main_run_root(pipeline_manifest)
        plans.append(CopyPlan(source_root, destination, "pipeline"))
        included_roots = [source_root]

        pipeline_payload = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
        dataset_raw = pipeline_payload.get("dataset_path") or pipeline_payload.get(
            "dataset_name"
        )
        if dataset_raw:
            dataset_path = Path(str(dataset_raw)).expanduser().resolve()
            if dataset_path.is_file():
                shared_datasets.add(dataset_path)
        labels_raw = pipeline_payload.get("context_labels_file")
        if labels_raw:
            labels_path = Path(str(labels_raw)).expanduser().resolve()
            if not labels_path.is_file():
                raise FileNotFoundError(
                    f"Referenced context-label file is missing: {labels_path}"
                )
            shared_labels.add(labels_path)
        artifacts = pipeline_payload.get("artifacts") or {}
        query_output_raw = artifacts.get("query_output_dir")
        if query_output_raw:
            query_root = Path(str(query_output_raw)).expanduser().resolve()
            if not query_root.is_dir():
                raise FileNotFoundError(
                    f"Referenced post-response query directory is missing: {query_root}"
                )
            if not _relative_to_any(query_root, included_roots):
                plans.append(CopyPlan(query_root, destination / "query", "post-query"))
                included_roots.append(query_root)

        for context in pipeline_payload.get("contexts") or []:
            if not isinstance(context, dict):
                continue
            metrics_raw = context.get("privacy_metrics_cimemories_json")
            if not metrics_raw:
                continue
            metrics_file = Path(str(metrics_raw)).expanduser().resolve()
            if not metrics_file.is_file():
                raise FileNotFoundError(
                    f"Referenced post-response privacy metrics are missing: {metrics_file}"
                )
            metrics_root = metrics_file.parent
            if not _relative_to_any(metrics_root, included_roots):
                context_idx = int(context.get("context_idx") or 0)
                plans.append(
                    CopyPlan(
                        metrics_root,
                        destination / "privacy_metrics" / f"context_{context_idx:03d}",
                        "post-privacy-metrics",
                    )
                )
                included_roots.append(metrics_root)

        pre_manifest_raw = run.get("pre_manifest")
        if pre_manifest_raw:
            pre_manifest = Path(str(pre_manifest_raw)).expanduser().resolve()
            if not pre_manifest.is_file():
                raise FileNotFoundError(f"Pre-rerank manifest is missing: {pre_manifest}")
            pre_root = pre_manifest.parent
            if not _relative_to_any(pre_root, included_roots):
                pre_destination = (
                    destination
                    / "pre_rerank_response_evaluation"
                    / "personas"
                    / f"persona_{persona_idx:03d}"
                )
                plans.append(CopyPlan(pre_root, pre_destination, "pre-rerank"))
                included_roots.append(pre_root)

        summaries = [
            Path(item).expanduser().resolve()
            for item in str(run.get("memory_stage_summaries") or "").split(";")
            if item
        ]
        for number, summary in enumerate(summaries, 1):
            if not summary.is_file():
                raise FileNotFoundError(f"Memory-stage summary is missing: {summary}")
            if not _relative_to_any(summary, included_roots):
                plans.append(
                    CopyPlan(
                        summary,
                        destination / "memory_stage_summaries" / f"summary_{number:02d}.json",
                        "memory-stage-summary",
                    )
                )

        public_runs.append(
            {
                "model": model,
                "architecture": architecture,
                "persona_idx": persona_idx,
                "pipeline": destination.as_posix(),
                "post_records": run.get("post_records"),
                "pre_records": run.get("pre_records"),
                "post_repeats": run.get("post_repeats"),
                "pre_repeats": run.get("pre_repeats"),
                "memory_stage_summary_count": run.get("memory_stage_summary_count"),
            }
        )

    def add_shared_files(paths: set[Path], stem: str) -> None:
        ordered = sorted(paths)
        for index, path in enumerate(ordered, 1):
            suffix = path.suffix or ".json"
            filename = f"{stem}{suffix}" if len(ordered) == 1 else f"{stem}_{index:02d}{suffix}"
            plans.append(CopyPlan(path, Path("inputs") / filename, f"input-{stem}"))

    add_shared_files(shared_datasets, "dataset")
    add_shared_files(shared_labels, "context_labels")
    plans.append(CopyPlan(report_root, Path("paper_report"), "paper-report"))
    return plans, public_runs


def _token_aliases(plans: list[CopyPlan]) -> TokenAliases:
    names: list[str] = []
    for plan in plans:
        source = plan.source.resolve()
        candidates = [source] if source.is_file() else source.rglob("*")
        for candidate in candidates:
            names.extend(candidate.parts)

    def aliases(values: set[str], prefix: str) -> dict[str, str]:
        return {
            value: f"{prefix}-{index:03d}"
            for index, value in enumerate(sorted(values), 1)
        }

    timestamps = {match.group(0) for name in names for match in RUN_TIMESTAMP_RE.finditer(name)}
    config_hashes = {match.group(0) for name in names for match in CONFIG_HASH_RE.finditer(name)}
    uuids = {match.group(0) for name in names for match in UUID_RE.finditer(name)}
    provider_ids = {
        match.group(0)
        for name in names
        for pattern in (OPENAI_RESPONSE_ID_RE, TOGETHER_RESPONSE_ID_RE)
        for match in pattern.finditer(name)
    }
    opaque_hex = {
        match.group(0)
        for name in names
        for match in LONG_HEX_RE.finditer(name)
        if match.group(0) not in config_hashes
    }
    return TokenAliases(
        run_timestamps=aliases(timestamps, "run"),
        configuration_hashes=aliases(config_hashes, "config"),
        opaque_hex=aliases(opaque_hex, "opaque"),
        uuids=aliases(uuids, "id"),
        provider_ids=aliases(provider_ids, "provider-response"),
    )


def _sanitize_component(name: str, token_aliases: TokenAliases) -> str:
    return token_aliases.rewrite(name)


def _sanitized_relative_path(relative: Path, token_aliases: TokenAliases) -> Path:
    return Path(*(_sanitize_component(part, token_aliases) for part in relative.parts))


def _plan_files(
    plans: list[CopyPlan], token_aliases: TokenAliases, scope: str = "full"
) -> list[PlannedFile]:
    if scope not in EXPORT_SCOPES:
        raise ValueError(f"Unknown export scope: {scope}")
    files: list[PlannedFile] = []
    destinations: dict[Path, Path] = {}
    seen_sources: set[Path] = set()
    for plan in plans:
        source = plan.source.resolve()
        if source.is_symlink():
            raise ValueError(f"Refusing to export a symlink: {source}")
        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(f"Refusing to export a symlink: {candidate}")
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen_sources:
                continue
            if candidate.suffix.lower() not in TEXT_SUFFIXES:
                raise ValueError(
                    f"Unsupported file type {candidate.suffix or '<none>'}; "
                    f"refusing unsafe copy: {candidate}"
                )
            seen_sources.add(resolved)
            if source.is_file():
                destination = plan.destination.parent / _sanitize_component(
                    plan.destination.name, token_aliases
                )
            else:
                relative = candidate.relative_to(source)
                destination = plan.destination / _sanitized_relative_path(
                    relative, token_aliases
                )
            if not _include_in_scope(destination, plan.label, scope):
                continue
            previous = destinations.get(destination)
            if previous is not None and previous != resolved:
                raise ValueError(
                    "Anonymized filename collision: "
                    f"{previous} and {resolved} both map to {destination}"
                )
            destinations[destination] = resolved
            files.append(PlannedFile(resolved, destination, plan.label))
    return files


def _include_in_scope(destination: Path, label: str, scope: str) -> bool:
    """Select release evidence without changing the lossless full export."""
    if scope == "full":
        return True
    if label in {"paper-report", "input-dataset", "input-context_labels"}:
        return True
    if label == "memory-stage-summary":
        return True

    path = destination.as_posix()
    name = destination.name
    parts = set(destination.parts)
    excluded_parts = {
        "histories",
        "calls",
        "exposure_calls",
        "judge_calls",
        "reranker_calls",
        "checkpoints",
    }
    if parts & excluded_parts:
        return False
    if name.startswith("judge-response_") or name.startswith("raw_response"):
        return False

    common_names = {
        "privacy_pipeline_cimemories.json",
        "pipeline_token_usage.md",
        "memory_stage_metrics.csv",
        "memory_stage_metrics_summary.json",
    }
    core_manifest = name == "manifest.json" and (
        path.endswith("/pipeline/manifest.json")
        or re.search(
            r"^pipelines/[^/]+/[^/]+/persona_[0-9]{3}/manifest\.json$",
            path,
        )
        is not None
        or re.search(
            r"/pre_rerank_response_evaluation/personas/persona_[0-9]{3}/manifest\.json$",
            path,
        )
        is not None
    )
    normalized_evidence = (
        name == "responses.jsonl"
        or (name.startswith("responses_") and name.endswith(".jsonl"))
        or (name.startswith("exposed_attributes_") and name.endswith(".jsonl"))
        or name == "privacy_metrics_cimemories.json"
        or name == "context_labeling_cimemories.json"
        or (
            name.startswith("memory_stage_metrics")
            and name.endswith((".json", ".csv"))
        )
    )

    if scope == "paper":
        return core_manifest or name in common_names or normalized_evidence

    if core_manifest or name in common_names:
        return True
    if name == "privacy_metrics_cimemories.json":
        return "/context_000/" in path
    if name == "context_labeling_cimemories.json":
        return "/context_000/" in path
    return (
        name == "responses.jsonl"
        or (name.startswith("responses_") and name.endswith(".jsonl"))
        or (name.startswith("exposed_attributes_") and name.endswith(".jsonl"))
        or (
            name.startswith("memory_stage_metrics")
            and name.endswith((".json", ".csv"))
        )
    )


def _path_replacements(
    plans: list[CopyPlan], repository_root: Path
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for plan in plans:
        source = plan.source.resolve()
        destination = "artifact://" + plan.destination.as_posix()
        replacements[str(source)] = destination
        try:
            replacements[source.relative_to(repository_root).as_posix()] = destination
        except ValueError:
            pass
    return replacements


def _decode_text(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"Binary file cannot be safely anonymized: {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Non-UTF-8 file cannot be safely anonymized: {path}") from exc


def _github_sample_row(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    context = payload.get("context_idx", payload.get("context_index"))
    repeat = payload.get("repeat_idx", payload.get("repeat_index", 0))
    try:
        return int(context) == 0 and int(repeat or 0) == 0
    except (TypeError, ValueError):
        return False


def _sanitize_file(
    source: Path, sanitizer: Sanitizer, scope: str = "full"
) -> str:
    if source.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {source.suffix or '<none>'}; refusing unsafe copy: {source}"
        )
    text = _decode_text(source)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
        return json.dumps(
            sanitizer.sanitize_value(payload), ensure_ascii=False, indent=2
        ) + "\n"
    if suffix == ".jsonl":
        rows: list[str] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}") from exc
            if (
                scope == "github"
                and (
                    source.name == "responses.jsonl"
                    or source.name.startswith("responses_")
                    or source.name.startswith("exposed_attributes_")
                )
                and not _github_sample_row(payload)
            ):
                continue
            rows.append(
                json.dumps(
                    sanitizer.sanitize_value(payload),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        return ("\n".join(rows) + "\n") if rows else ""
    return sanitizer.sanitize_text(text)


def _estimated_selected_bytes(files: list[PlannedFile], scope: str) -> int:
    """Estimate written payload before formatting and anonymization rewrites."""
    total = 0
    for planned in files:
        source = planned.source
        sampled_jsonl = (
            scope == "github"
            and source.suffix.lower() == ".jsonl"
            and (
                source.name == "responses.jsonl"
                or source.name.startswith("responses_")
                or source.name.startswith("exposed_attributes_")
            )
        )
        if not sampled_jsonl:
            total += source.stat().st_size
            continue
        for line in _decode_text(source).splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if _github_sample_row(payload):
                total += len(line.encode("utf-8")) + 1
    return total


def _walk_json(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def audit_anonymized_directory(root: Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    checked_files = 0
    json_files = 0
    patterns = {
        "absolute_home_path": ABSOLUTE_HOME_RE,
        "letta_operational_id": LETTA_ID_RE,
        "uuid": UUID_RE,
        "openai_response_id": OPENAI_RESPONSE_ID_RE,
        "together_response_id": TOGETHER_RESPONSE_ID_RE,
        "run_timestamp": RUN_TIMESTAMP_RE,
        **{f"secret:{name}": pattern for name, pattern in SECRET_PATTERNS.items()},
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "anonymization_report.json":
            continue
        checked_files += 1
        text = _decode_text(path)
        for label, pattern in patterns.items():
            if pattern.search(text):
                issues.append({"file": path.relative_to(root).as_posix(), "type": label})
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        payloads: list[Any]
        if path.suffix.lower() == ".json":
            payloads = [json.loads(text)]
        else:
            payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
        json_files += 1
        for payload in payloads:
            for key, item in _walk_json(payload):
                lowered = key.lower()
                if _is_removed_key(key) or (
                    lowered == "id" and Sanitizer._is_operational_id(item)
                ):
                    issues.append(
                        {
                            "file": path.relative_to(root).as_posix(),
                            "type": f"forbidden_json_key:{key}",
                        }
                    )
    return {
        "status": "passed" if not issues else "failed",
        "checked_files": checked_files,
        "checked_json_jsonl_files": json_files,
        "issue_count": len(issues),
        "issues": issues[:200],
        "issues_truncated": len(issues) > 200,
    }


def write_artifact_checksums(root: Path) -> Path:
    root = root.expanduser().resolve()
    checksum_path = root / "SHA256SUMS"
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checksum_path


def verify_anonymized_artifact(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    issues: list[str] = []
    required = {
        "artifact_manifest.json",
        "anonymization_report.json",
        "README.md",
        "SHA256SUMS",
    }
    for name in sorted(required):
        if not (root / name).is_file():
            issues.append(f"missing required file: {name}")
    if issues:
        return {"status": "failed", "issue_count": len(issues), "issues": issues}

    checksum_rows = 0
    checksummed_paths: set[str] = set()
    for line_number, line in enumerate(
        (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            issues.append(f"invalid SHA256SUMS line {line_number}")
            continue
        expected, relative_text = match.groups()
        if relative_text in checksummed_paths:
            issues.append(f"duplicate checksum path: {relative_text}")
            continue
        checksummed_paths.add(relative_text)
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            issues.append(f"unsafe checksum path: {relative_text}")
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            issues.append(f"missing checksummed file: {relative_text}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            issues.append(f"checksum mismatch: {relative_text}")
        checksum_rows += 1

    artifact_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    for relative_text in sorted(artifact_paths - checksummed_paths):
        issues.append(f"unchecksummed artifact file: {relative_text}")
    for relative_text in sorted(checksummed_paths - artifact_paths):
        issues.append(f"checksum lists absent artifact file: {relative_text}")

    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    report = json.loads(
        (root / "anonymization_report.json").read_text(encoding="utf-8")
    )
    if manifest.get("scope") != report.get("scope"):
        issues.append("scope differs between manifest and anonymization report")
    if report.get("status") != "passed" or (report.get("audit") or {}).get(
        "status"
    ) != "passed":
        issues.append("anonymization audit is not passed")

    pipelines = manifest.get("pipelines") or []
    keys = [
        (row.get("model"), row.get("architecture"), row.get("persona_idx"))
        for row in pipelines
        if isinstance(row, dict)
    ]
    if len(keys) != len(set(keys)):
        issues.append("duplicate model/architecture/persona pipeline key")
    if len(keys) != int(manifest.get("selected_pipeline_count") or -1):
        issues.append("selected_pipeline_count does not match pipeline rows")
    expected_keys = {
        (model, architecture, persona)
        for model in manifest.get("models") or []
        for architecture in manifest.get("architectures") or []
        for persona in manifest.get("personas") or []
    }
    if set(keys) != expected_keys:
        issues.append("pipeline rows do not form the declared complete matrix")
    for row in pipelines:
        if not isinstance(row, dict):
            continue
        pipeline = root / str(row.get("pipeline") or "")
        if not pipeline.is_dir():
            issues.append(f"missing pipeline directory: {row.get('pipeline')}")

    locked_inputs = root / "paper_report" / "integrated_artifact_inputs.json"
    if locked_inputs.is_file():
        locked = json.loads(locked_inputs.read_text(encoding="utf-8"))
        locked_keys: set[tuple[Any, Any, Any]] = set()
        for row in locked.get("selected_runs") or []:
            if not isinstance(row, dict):
                continue
            locked_keys.add(
                (row.get("model"), row.get("architecture"), row.get("persona_idx"))
            )
            references = [row.get("pipeline_manifest"), row.get("pre_manifest")]
            references.extend(
                item
                for item in str(row.get("memory_stage_summaries") or "").split(";")
                if item
            )
            for reference in references:
                if not reference:
                    continue
                if not str(reference).startswith("artifact://"):
                    issues.append(f"non-portable locked reference: {reference}")
                    continue
                relative = Path(str(reference).removeprefix("artifact://"))
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not (root / relative).is_file()
                ):
                    issues.append(f"missing locked reference: {reference}")
        if locked_keys != set(keys):
            issues.append("locked report inputs differ from artifact pipeline matrix")

    sample_rows = 0
    if manifest.get("scope") == "github":
        for path in sorted((root / "pipelines").rglob("*.jsonl")):
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != 1 or not _github_sample_row(rows[0]):
                issues.append(
                    "GitHub sample must contain exactly context 0, repeat 0: "
                    f"{path.relative_to(root)}"
                )
            sample_rows += len(rows)

    return {
        "status": "passed" if not issues else "failed",
        "scope": manifest.get("scope"),
        "pipeline_count": len(keys),
        "checksum_count": checksum_rows,
        "sample_row_count": sample_rows,
        "issue_count": len(issues),
        "issues": issues,
    }


def export_anonymized_artifacts(
    inputs: Path,
    output: Path,
    *,
    scope: str = "full",
    progress: Callable[[int, int, PlannedFile], None] | None = None,
) -> Path:
    inputs_path, report_root = _resolve_inputs(inputs)
    selected_runs = _load_selected_runs(inputs_path)
    repository_root = Path.cwd().resolve()
    plans, public_runs = _copy_plans(selected_runs, report_root)
    token_aliases = _token_aliases(plans)
    files = _plan_files(plans, token_aliases, scope)

    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    output.mkdir(parents=True)

    replacements = _path_replacements(plans, repository_root)
    sanitizer = Sanitizer(
        PathRewriter(replacements, repository_root, token_aliases)
    )
    total = len(files)
    for index, planned in enumerate(files, 1):
        destination = output / planned.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            _sanitize_file(planned.source, sanitizer, scope), encoding="utf-8"
        )
        if progress is not None:
            progress(index, total, planned)

    artifact_manifest = {
        "schema_version": 1,
        "scope": scope,
        "software_release": ARTIFACT_SOFTWARE_RELEASE,
        "dataset": "CIMemories synthetic personas",
        "synthetic_persona_notice": (
            "Names, addresses, health facts, financial facts, and relationships in "
            "the pipeline evidence are synthetic dataset content, not participant data."
        ),
        "selected_pipeline_count": len(public_runs),
        "models": sorted({str(run["model"]) for run in public_runs}),
        "architectures": sorted({str(run["architecture"]) for run in public_runs}),
        "personas": sorted({int(run["persona_idx"]) for run in public_runs}),
        "pipelines": public_runs,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Anonymous CIMemories evaluation artifact\n\n"
        f"This directory is a non-destructive, sanitized `{scope}`-scope copy "
        "of the locked paper pipelines and paper-report bundle. Source directories "
        "were not modified.\n\n"
        "Operational identifiers, provider response identifiers, credentials and "
        "credential metadata, absolute user paths, execution fingerprints, exact run "
        "timestamps, and agent/conversation identifiers were removed or replaced. "
        "Paths inside copied files use portable `artifact://` or `repository://` "
        "references.\n\n"
        "The apparent personal information retained in prompts, memories, and responses "
        "belongs to synthetic CIMemories personas. It is retained because it is the "
        "scientific object measured by the privacy evaluation; it is not human-subject "
        "or author information.\n\n"
        "The exported figures and tables are provided as analysis and visualization "
        "aids. They do not necessarily match the final presentation, layout, or "
        "emphasis conventions used in the paper. The accompanying CSV and JSON files "
        "contain the authoritative numerical results.\n\n"
        "See `artifact_manifest.json` for the stable model/architecture/persona layout "
        "and `anonymization_report.json` for the automated release audit. Verify file "
        "integrity from the repository root with "
        "`python3 scripts/verify_anonymized_cimemories_artifact.py <this-directory>`.\n",
        encoding="utf-8",
    )

    audit = audit_anonymized_directory(output)
    report = {
        "schema_version": 1,
        "scope": scope,
        "status": audit["status"],
        "source_files_processed": len(files),
        "selected_pipeline_count": len(public_runs),
        "transform_counts": dict(sorted(sanitizer.counts.items())),
        "audit": audit,
    }
    (output / "anonymization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if audit["status"] != "passed":
        raise RuntimeError(
            f"Anonymization audit failed with {audit['issue_count']} issue(s); "
            f"inspect {output / 'anonymization_report.json'}"
        )
    write_artifact_checksums(output)
    return output


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-destructive anonymous release copy of every pipeline locked "
            "by a CIMemories integrated paper-artifact export."
        )
    )
    parser.add_argument(
        "inputs",
        type=Path,
        help=(
            "integrated_artifact_inputs.json or the report directory containing it"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_outputs/cimemories-anonymous-artifact"),
        help=(
            "new output directory; it must not already exist "
            "(default: research_outputs/cimemories-anonymous-artifact)"
        ),
    )
    parser.add_argument(
        "--scope",
        choices=EXPORT_SCOPES,
        default="full",
        help=(
            "github keeps aggregates, provenance, and representative context-0 "
            "samples; paper keeps normalized evidence for every evaluated cell; "
            "full preserves the prior lossless export (default: full)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and size the export without writing files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        inputs_path, report_root = _resolve_inputs(args.inputs)
        selected_runs = _load_selected_runs(inputs_path)
        plans, _ = _copy_plans(selected_runs, report_root)
        token_aliases = _token_aliases(plans)
        files = _plan_files(plans, token_aliases, args.scope)
        byte_count = _estimated_selected_bytes(files, args.scope)
        if args.dry_run:
            print(
                f"Validated {len(selected_runs)} pipelines and {len(files)} files "
                f"({byte_count / (1024 ** 2):.1f} MiB)."
            )
            print(f"Scope: {args.scope}")
            print(f"Would create: {args.output.expanduser().resolve()}")
            return 0

        last_percent = -1

        def show_progress(index: int, total: int, planned: PlannedFile) -> None:
            nonlocal last_percent
            percent = int(index * 100 / total) if total else 100
            if percent != last_percent and (percent % 2 == 0 or index == total):
                print(
                    f"[anonymize] {index}/{total} ({percent:3d}%) "
                    f"{planned.label}",
                    flush=True,
                )
                last_percent = percent

        output = export_anonymized_artifacts(
            args.inputs, args.output, scope=args.scope, progress=show_progress
        )
        print(f"Anonymous artifact: {output}")
        print(f"Audit: {output / 'anonymization_report.json'}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
