from __future__ import annotations

import math
import re
from typing import Any, Iterable


EFFICIENCY_SCHEMA_VERSION = 1
USAGE_LEDGER_SCHEMA_VERSION = 1
VISIBLE_TOKEN_ESTIMATOR = "unicode_word_or_punctuation_v1"


def elapsed_ms(start_ns: int, end_ns: int) -> float:
    """Convert a monotonic perf-counter interval to milliseconds."""
    return round((end_ns - start_ns) / 1_000_000, 3)


def estimate_visible_tokens(text: str) -> int:
    """Stable provider-independent proxy for visible prompt/response size.

    This deliberately is not presented as provider billing usage. It counts
    Unicode word runs and non-whitespace punctuation using a versioned rule.
    """
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _number(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def normalize_token_usage(value: Any, *, source: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    input_tokens = _number(value.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _number(value.get("prompt_tokens"))
    output_tokens = _number(value.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _number(value.get("completion_tokens"))
    total_tokens = _number(value.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    input_details = value.get("input_tokens_details") or value.get("prompt_tokens_details")
    output_details = value.get("output_tokens_details") or value.get("completion_tokens_details")
    cached_tokens = _number(input_details.get("cached_tokens")) if isinstance(input_details, dict) else None
    reasoning_tokens = (
        _number(output_details.get("reasoning_tokens")) if isinstance(output_details, dict) else None
    )
    return {
        "exact": True,
        "source": source,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_tokens,
        "reasoning_output_tokens": reasoning_tokens,
    }


def extract_token_usage(payload: Any) -> dict[str, Any] | None:
    """Extract one authoritative usage object without double-counting nested totals."""
    if not isinstance(payload, dict):
        return None
    for key in ("usage", "usage_statistics"):
        normalized = normalize_token_usage(payload.get(key), source=f"response.{key}")
        if normalized is not None:
            return normalized

    candidates: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if key in ("usage", "usage_statistics"):
                    normalized = normalize_token_usage(child, source=child_path)
                    if normalized is not None:
                        candidates.append(normalized)
                else:
                    visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "response")
    if not candidates:
        return None
    # A response may repeat an aggregate alongside step usage. Selecting the
    # largest authoritative object is deterministic and avoids summing both.
    return max(candidates, key=lambda item: item.get("total_tokens") or -1)


def sum_token_usage(usages: Iterable[dict[str, Any] | None]) -> dict[str, Any] | None:
    valid = [usage for usage in usages if isinstance(usage, dict)]
    if not valid:
        return None

    def summed(key: str) -> int | None:
        values = [_number(usage.get(key)) for usage in valid]
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    return {
        "exact": all(usage.get("exact") is True for usage in valid),
        "usage_count": len(valid),
        "input_tokens": summed("input_tokens"),
        "output_tokens": summed("output_tokens"),
        "total_tokens": summed("total_tokens"),
        "cached_input_tokens": summed("cached_input_tokens"),
        "reasoning_output_tokens": summed("reasoning_output_tokens"),
    }


def usage_component(
    component: str,
    *,
    stage: str,
    modality: str,
    operation: str,
    model: str | None = None,
    provider: str | None = None,
    usage: dict[str, Any] | None = None,
    expected_calls: int | None = None,
    observed_calls: int | None = None,
    availability: str | None = None,
    additive: bool | None = None,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one explicit, additive pipeline-usage component.

    Missing backend telemetry is represented as unavailable, never as zero.  A
    component is additive only when exact provider usage was observed for all
    expected calls (or when the backend supplied an authoritative aggregate).
    """
    normalized = normalize_token_usage(usage, source=f"usage_ledger.{component}") if usage else None
    if normalized is None and isinstance(usage, dict) and usage.get("exact") is True:
        normalized = dict(usage)
    observed = observed_calls if observed_calls is not None else (1 if normalized else 0)
    expected = expected_calls
    if availability is None:
        if normalized is None:
            availability = "unavailable"
        elif expected is None or observed >= expected:
            availability = "exact"
        else:
            availability = "partial"
    coverage = None
    if expected is not None:
        coverage = min(1.0, observed / expected) if expected > 0 else 1.0
    return {
        "component": component,
        "stage": stage,
        "modality": modality,
        "operation": operation,
        "provider": provider,
        "model": model,
        "availability": availability,
        "additive": (availability == "exact" and normalized is not None) if additive is None else additive,
        "expected_calls": expected,
        "observed_calls": observed,
        "coverage": round(coverage, 6) if coverage is not None else None,
        "tokens": normalized,
        "reason": reason,
        "metadata": metadata or {},
    }


def aggregate_usage_components(components: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = [dict(item) for item in components if isinstance(item, dict)]
    additive = [item for item in items if item.get("additive") is True]
    totals = sum_token_usage(
        item.get("tokens") for item in additive if isinstance(item.get("tokens"), dict)
    )
    unavailable = [item.get("component") for item in items if item.get("availability") == "unavailable"]
    partial = [item.get("component") for item in items if item.get("availability") == "partial"]
    return {
        "schema_version": USAGE_LEDGER_SCHEMA_VERSION,
        "components": items,
        "additive_exact_total": totals,
        "complete": not unavailable and not partial,
        "unavailable_components": unavailable,
        "partial_components": partial,
        "note": (
            "additive_exact_total sums only exact fully covered components; "
            "unavailable and partial components are never treated as zero"
        ),
    }


def token_usage_delta(before: Any, after: Any, *, source: str) -> dict[str, Any] | None:
    """Return a non-negative delta when a backend exposes cumulative token usage."""
    before_usage = extract_token_usage(before) or normalize_token_usage(before, source=f"{source}.before")
    after_usage = extract_token_usage(after) or normalize_token_usage(after, source=f"{source}.after")
    if after_usage is None:
        return None
    result: dict[str, Any] = {"exact": True, "source": source}
    found = False
    for key in (
        "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_output_tokens"
    ):
        after_value = _number(after_usage.get(key))
        before_value = _number(before_usage.get(key)) if before_usage else 0
        if after_value is None or before_value is None:
            result[key] = None
            continue
        delta = after_value - before_value
        if delta < 0:
            return None
        result[key] = delta
        found = True
    return result if found else None


def summarize_samples(values: Iterable[float]) -> dict[str, Any]:
    samples = sorted(float(value) for value in values)
    if not samples:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    count = len(samples)
    middle = count // 2
    median = samples[middle] if count % 2 else (samples[middle - 1] + samples[middle]) / 2
    p95 = samples[max(0, math.ceil(0.95 * count) - 1)]
    return {
        "count": count,
        "mean": round(sum(samples) / count, 3),
        "median": round(median, 3),
        "p95": round(p95, 3),
        "min": round(samples[0], 3),
        "max": round(samples[-1], 3),
    }


def summarize_query_efficiency(
    records: list[dict[str, Any]],
    context_preparation: dict[int, dict[str, Any]],
    *,
    initialization_ms: float,
    agent_creation_ms: float,
    generation_batch_ms: float,
    experiment_elapsed_ms: float,
    repeats_per_context: int,
    concurrency: int,
    additional_usage_components: Iterable[dict[str, Any]] = (),
    generation_model: str | None = None,
    reranker_model: str | None = None,
) -> dict[str, Any]:
    generation_ms: list[float] = []
    observed_online_ms: list[float] = []
    deployed_online_ms: list[float] = []
    generation_usage: list[dict[str, Any] | None] = []
    visible_prompt_tokens = 0
    visible_response_tokens = 0

    for record in records:
        efficiency = record.get("efficiency")
        if not isinstance(efficiency, dict):
            continue
        timings = efficiency.get("timings_ms")
        tokens = efficiency.get("tokens")
        if isinstance(timings, dict):
            generation = timings.get("generation")
            online = timings.get("online_observed")
            if isinstance(generation, (int, float)):
                generation_ms.append(float(generation))
            if isinstance(online, (int, float)):
                observed_online_ms.append(float(online))
                context_idx = record.get("context_idx")
                preparation = context_preparation.get(context_idx, {}).get("duration_ms")
                if isinstance(preparation, (int, float)):
                    deployed_online_ms.append(float(online) + float(preparation))
        if isinstance(tokens, dict):
            generation_usage.append(tokens.get("generation_exact"))
            visible_prompt_tokens += int(tokens.get("visible_prompt_estimate", 0) or 0)
            visible_response_tokens += int(tokens.get("visible_response_estimate", 0) or 0)

    preparation_usages = [item.get("exact_model_tokens") for item in context_preparation.values()]
    exact_generation = sum_token_usage(generation_usage)
    exact_preparation = sum_token_usage(preparation_usages)
    record_count = len(records)
    context_count = len(context_preparation)
    expected_preparation_usage = sum(
        1 for item in context_preparation.values() if item.get("exact_model_tokens_expected") is True
    )
    reported_expected_preparation_usage = sum(
        1
        for item in context_preparation.values()
        if item.get("exact_model_tokens_expected") is True
        and isinstance(item.get("exact_model_tokens"), dict)
    )
    generation_coverage = (
        (exact_generation.get("usage_count", 0) / record_count) if exact_generation and record_count else 0.0
    )
    preparation_coverage = (
        reported_expected_preparation_usage / expected_preparation_usage
        if expected_preparation_usage
        else 1.0
    )

    exact_observed_total = None
    exact_deployed_per_query = None
    if exact_generation and generation_coverage == 1.0 and preparation_coverage == 1.0:
        generation_total = exact_generation.get("total_tokens")
        preparation_total = exact_preparation.get("total_tokens") if exact_preparation else 0
        if isinstance(generation_total, int) and isinstance(preparation_total, int):
            exact_observed_total = generation_total + preparation_total
            if record_count:
                exact_deployed_per_query = round(
                    (generation_total / record_count)
                    + ((preparation_total / context_count) if context_count else 0),
                    3,
                )

    components = list(additional_usage_components)
    components.extend(
        [
            usage_component(
                "response_generation",
                stage="online",
                modality="generation",
                operation="generate_recipient_response",
                model=generation_model,
                provider="letta",
                usage=exact_generation,
                expected_calls=record_count,
                observed_calls=exact_generation.get("usage_count", 0) if exact_generation else 0,
                availability=("exact" if generation_coverage == 1.0 else "partial" if exact_generation else "unavailable"),
                reason=None if exact_generation else "Letta response did not expose provider token usage",
            ),
            usage_component(
                "explicit_memory_reranking",
                stage="online_memory_preparation",
                modality="reranking",
                operation="rerank_memory_candidates",
                model=reranker_model,
                usage=exact_preparation,
                expected_calls=expected_preparation_usage,
                observed_calls=reported_expected_preparation_usage,
                availability=(
                    "not_applicable" if expected_preparation_usage == 0
                    else "exact" if preparation_coverage == 1.0
                    else "partial" if exact_preparation else "unavailable"
                ),
                reason=(
                    None if expected_preparation_usage
                    else "No explicit generative reranker was configured; this component has zero expected calls"
                ),
            ),
        ]
    )
    usage_ledger = aggregate_usage_components(components)

    return {
        "schema_version": EFFICIENCY_SCHEMA_VERSION,
        "measurement": {
            "clock": "time.perf_counter_ns",
            "duration_unit": "milliseconds",
            "percentile_method": "nearest_rank",
            "visible_token_estimator": VISIBLE_TOKEN_ESTIMATOR,
            "exact_tokens_are_provider_reported": True,
        },
        "counts": {
            "contexts": context_count,
            "responses": record_count,
            "repeats_per_context": repeats_per_context,
            "memory_preparation_executions": context_count,
            "generation_concurrency": concurrency,
        },
        "one_time": {
            "agent_creation_ms": agent_creation_ms,
            "memory_initialization_ms": initialization_ms,
            "memory_initialization_exact_tokens": None,
            "memory_initialization_token_coverage": 0.0,
        },
        "timings_ms": {
            "memory_preparation": summarize_samples(
                item["duration_ms"]
                for item in context_preparation.values()
                if isinstance(item.get("duration_ms"), (int, float))
            ),
            "generation": summarize_samples(generation_ms),
            "online_observed": summarize_samples(observed_online_ms),
            "online_deployed_estimate": summarize_samples(deployed_online_ms),
            "generation_batch_wall": generation_batch_ms,
            "experiment_wall": experiment_elapsed_ms,
        },
        "tokens": {
            "memory_preparation_exact": exact_preparation,
            "generation_exact": exact_generation,
            "generation_exact_coverage": round(generation_coverage, 6),
            "memory_preparation_exact_coverage": round(preparation_coverage, 6),
            "observed_exact_total": exact_observed_total,
            "deployed_exact_tokens_per_query": exact_deployed_per_query,
            "visible_prompt_estimate_total": visible_prompt_tokens,
            "visible_response_estimate_total": visible_response_tokens,
            "visible_prompt_estimate_per_response": (
                round(visible_prompt_tokens / record_count, 3) if record_count else None
            ),
            "visible_response_estimate_per_response": (
                round(visible_response_tokens / record_count, 3) if record_count else None
            ),
        },
        "usage_ledger": usage_ledger,
        "context_preparation": {str(key): value for key, value in sorted(context_preparation.items())},
        "notes": [
            "Memory preparation is executed once per context and shared by its repeated generations.",
            "online_observed excludes shared memory preparation; online_deployed_estimate adds it to each response.",
            "Post-response history fetching is recorded per response but excluded from online architecture latency.",
            "Visible token estimates are stable size proxies, not provider billing tokens.",
            "One-time backend ingestion token usage is unavailable unless a backend exposes it; its latency remains measured.",
            "Exact-token coverage covers instrumented online generative calls, not unreported embedding or backend-internal usage.",
            "Evaluator and privacy-judge overhead is excluded from architecture efficiency.",
        ],
    }
