from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .comparative_report import ARCHITECTURES, _read_json
from .study_artifacts import (
    ARCH_COLORS,
    MODEL_COLORS,
    _estimate,
    _short_model,
    export_cimemories_study_artifacts,
)


MEMORY_CATEGORIES = ("necessary", "private", "ambiguous")
RESPONSE_METRICS = {
    "necessary": "necessary_recall",
    "private": "inappropriate_leak_rate",
    "ambiguous": "ambiguous_exposure_rate",
}
CATEGORY_LABELS = {
    "necessary": "Necessary",
    "private": "Private",
    "ambiguous": "Ambiguous",
}


def _model_sort_key(model: str) -> tuple[int, str]:
    lowered = model.lower()
    if model == "gpt-5.6-sol":
        return (0, lowered)
    if "glm-5.3" in lowered:
        return (1, lowered)
    if "deepseek-v4" in lowered:
        return (2, lowered)
    return (3, lowered)


def _condition_sort_key(item: tuple[str, str]) -> tuple[int, str, int]:
    model, architecture = item
    model_rank, model_name = _model_sort_key(model)
    return (model_rank, model_name, ARCHITECTURES.index(architecture))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: list[Any]) -> float | None:
    usable = [value for value in (_float(item) for item in values) if value is not None]
    return mean(usable) if usable else None


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = sum((x - left_mean) ** 2 for x in left)
    right_scale = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _architecture_efficiency(
    selected_runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize deployed post-rerank efficiency by memory architecture.

    The summary macro-averages model-persona pipeline cells, rather than pooling
    requests, so a model with more repetitions cannot dominate an architecture.
    Detailed rows retain the exact source-manifest measurements and coverage.
    """
    details: list[dict[str, Any]] = []
    for run in selected_runs:
        manifest_path = Path(str(run["pipeline_manifest"])).expanduser().resolve()
        manifest = _read_json(manifest_path)
        efficiency = manifest.get("efficiency") or {}
        timings = efficiency.get("timings_ms") or {}
        one_time = efficiency.get("one_time") or {}
        tokens = efficiency.get("tokens") or {}
        usage_ledger = efficiency.get("usage_ledger") or {}

        def seconds(group: str, statistic: str = "mean") -> float | None:
            value = _float((timings.get(group) or {}).get(statistic))
            return value / 1000 if value is not None else None

        initialization_ms = _float(one_time.get("memory_initialization_ms"))
        deployed_tokens = _float(tokens.get("deployed_exact_tokens_per_query"))
        details.append(
            {
                "model": str(run["model"]),
                "model_label": _short_model(str(run["model"])),
                "architecture": str(run["architecture"]),
                "persona_idx": int(run["persona_idx"]),
                "memory_preparation_seconds": seconds("memory_preparation"),
                "generation_seconds": seconds("generation"),
                "deployed_seconds": seconds("online_deployed_estimate"),
                "deployed_p95_seconds": seconds("online_deployed_estimate", "p95"),
                "reported_exact_tokens_per_query": deployed_tokens,
                "initialization_seconds": (
                    initialization_ms / 1000 if initialization_ms is not None else None
                ),
                "generation_exact_coverage": _float(
                    tokens.get("generation_exact_coverage")
                ),
                "memory_preparation_exact_coverage": _float(
                    tokens.get("memory_preparation_exact_coverage")
                ),
                "usage_ledger_complete": bool(usage_ledger.get("complete")),
                "source_manifest": str(manifest_path),
            }
        )

    present_architectures = [
        architecture
        for architecture in ARCHITECTURES
        if any(row["architecture"] == architecture for row in details)
    ]
    token_keys = [
        {
            (row["model"], row["persona_idx"])
            for row in details
            if row["architecture"] == architecture
            and row["reported_exact_tokens_per_query"] is not None
        }
        for architecture in present_architectures
    ]
    shared_token_keys = set.intersection(*token_keys) if token_keys else set()

    summary: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        subset = [row for row in details if row["architecture"] == architecture]
        if not subset:
            continue
        matched_token_subset = [
            row
            for row in subset
            if (row["model"], row["persona_idx"]) in shared_token_keys
        ]
        summary.append(
            {
                "architecture": architecture,
                "pipeline_cells": len(subset),
                "model_count": len({row["model"] for row in subset}),
                "persona_count": len({row["persona_idx"] for row in subset}),
                "memory_preparation_seconds": _mean(
                    [row["memory_preparation_seconds"] for row in subset]
                ),
                "generation_seconds": _mean(
                    [row["generation_seconds"] for row in subset]
                ),
                "deployed_seconds": _mean(
                    [row["deployed_seconds"] for row in subset]
                ),
                "deployed_p95_seconds": _mean(
                    [row["deployed_p95_seconds"] for row in subset]
                ),
                "reported_exact_tokens_per_query": _mean(
                    [
                        row["reported_exact_tokens_per_query"]
                        for row in matched_token_subset
                    ]
                ),
                "initialization_seconds": _mean(
                    [row["initialization_seconds"] for row in subset]
                ),
                "available_token_cells": sum(
                    row["reported_exact_tokens_per_query"] is not None for row in subset
                ),
                "reported_token_cells": len(matched_token_subset),
                "complete_usage_ledger_cells": sum(
                    row["usage_ledger_complete"] for row in subset
                ),
            }
        )
    return summary, details


def _memory_summary_bases(pipeline_manifest: Path) -> list[Path]:
    bases = [pipeline_manifest.parent]
    if pipeline_manifest.parent.name == "pipeline":
        bases.append(pipeline_manifest.parent.parent)
    return bases


def _summary_matches(
    payload: dict[str, Any], *, architecture: str, judge_model: str
) -> bool:
    if int(payload.get("context_count") or 0) != 49:
        return False
    strategy = str(payload.get("strategy") or "")
    if architecture == "list":
        return strategy == "exact-match"
    return (
        strategy == "monolithic"
        and str(payload.get("judge_model") or "").lower() == judge_model.lower()
        and payload.get("pilot_contexts") in (None, 0)
    )


def _find_memory_summary(
    run: dict[str, Any], *, judge_model: str, locked_path: str | None = None
) -> Path | None:
    architecture = str(run["architecture"])
    if locked_path:
        path = Path(locked_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Locked memory-stage summary is missing: {path}")
        payload = _read_json(path)
        if not _summary_matches(payload, architecture=architecture, judge_model=judge_model):
            raise ValueError(
                f"Locked memory-stage summary no longer matches {architecture}/{judge_model}: {path}"
            )
        return path

    manifest = Path(str(run["pipeline_manifest"])).expanduser().resolve()
    candidates: list[Path] = []
    for base in _memory_summary_bases(manifest):
        candidates.extend(base.glob("memory_stage_metrics*_summary.json"))
    matches: list[Path] = []
    for path in sorted(set(candidates)):
        if "pilot_" in path.name or "attribute_batch" in path.name or "per_attribute" in path.name:
            continue
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if _summary_matches(payload, architecture=architecture, judge_model=judge_model):
            matches.append(path.resolve())
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Multiple production memory-stage summaries match {run['model']} "
            f"{architecture} persona {run['persona_idx']}: {matches}"
        )
    return matches[0]


def _context_paths(summary_path: Path, summary: dict[str, Any]) -> list[Path]:
    result: list[Path] = []
    for value in summary.get("context_metric_files") or []:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = summary_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Memory-stage context artifact is missing: {path}")
        result.append(path)
    if len(result) != 49:
        raise ValueError(f"Expected 49 memory-stage context artifacts for {summary_path}; found {len(result)}")
    return result


def _collect_memory_rows(
    selected_runs: list[dict[str, Any]],
    *,
    memory_persona: int,
    judge_model: str,
    locked_memory: dict[tuple[str, str, int], str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    targets = [run for run in selected_runs if int(run["persona_idx"]) == memory_persona]
    for run in sorted(
        targets,
        key=lambda item: _condition_sort_key((str(item["model"]), str(item["architecture"]))),
    ):
        key = (str(run["model"]), str(run["architecture"]), memory_persona)
        locked_path = (locked_memory or {}).get(key)
        summary_path = _find_memory_summary(run, judge_model=judge_model, locked_path=locked_path)
        coverage.append(
            {
                "model": run["model"],
                "model_label": _short_model(str(run["model"])),
                "architecture": run["architecture"],
                "persona_idx": memory_persona,
                "complete": summary_path is not None,
                "memory_stage_summary": str(summary_path) if summary_path else "",
            }
        )
        if summary_path is None:
            continue
        summary = _read_json(summary_path)
        configuration = summary.get("judge_configuration") or {}
        provenance.append(
            {
                "model": run["model"],
                "model_label": _short_model(str(run["model"])),
                "architecture": run["architecture"],
                "persona_idx": memory_persona,
                "summary_path": str(summary_path),
                "strategy": summary.get("strategy"),
                "judge_model": summary.get("judge_model") or "exact",
                "judge_fingerprint": summary.get("judge_configuration_fingerprint") or "",
                "prompt_version": summary.get("prompt_version") or "",
                "api_style": configuration.get("api_style") or "",
                "reasoning_effort": configuration.get("reasoning_effort") or "",
                "max_output_tokens": configuration.get("max_output_tokens"),
                "context_count": summary.get("context_count"),
            }
        )
        for context_path in _context_paths(summary_path, summary):
            payload = _read_json(context_path)
            represented = payload.get("represented") or {}
            row: dict[str, Any] = {
                "model": run["model"],
                "model_label": _short_model(str(run["model"])),
                "architecture": run["architecture"],
                "persona_idx": memory_persona,
                "persona_name": payload.get("persona_name"),
                "context_idx": payload.get("context_idx"),
                "recipient": payload.get("recipient"),
                "task": payload.get("task"),
                "context_discarded": bool(payload.get("context_discarded")),
                "pre_fact_count": payload.get("pre_rerank_fact_count"),
                "post_fact_count": payload.get("post_rerank_fact_count"),
                "pre_attribute_count": len(represented.get("pre_rerank_indices") or []),
                "post_attribute_count": len(represented.get("post_rerank_indices") or []),
                "matching_method": summary.get("strategy"),
                "judge_model": summary.get("judge_model") or "exact",
                "source_summary": str(summary_path),
                "source_context": str(context_path),
            }
            for category in MEMORY_CATEGORIES:
                pre = (payload.get("pre_rerank") or {}).get(category) or {}
                post = (payload.get("post_rerank") or {}).get(category) or {}
                reranker = (payload.get("reranker") or {}).get(category) or {}
                row[f"pre_{category}_availability"] = pre.get("availability_rate")
                row[f"post_{category}_availability"] = post.get("availability_rate")
                pre_value = _float(pre.get("availability_rate"))
                post_value = _float(post.get("availability_rate"))
                row[f"delta_{category}_availability"] = (
                    post_value - pre_value if pre_value is not None and post_value is not None else None
                )
                row[f"{category}_retention"] = reranker.get("retention_rate")
                row[f"{category}_removal"] = reranker.get("removal_rate")
                row[f"pre_{category}_represented"] = pre.get("represented_count")
                row[f"post_{category}_represented"] = post.get("represented_count")
                row[f"{category}_total"] = pre.get("total")
            rows.append(row)
    return rows, provenance, coverage


def _memory_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    conditions = sorted(
        {(row["model"], row["architecture"]) for row in rows},
        key=_condition_sort_key,
    )
    for model, architecture in conditions:
        subset = [row for row in rows if row["model"] == model and row["architecture"] == architecture]
        for scope, scoped in (
            ("all", subset),
            ("exclude_discarded", [row for row in subset if not row["context_discarded"]]),
        ):
            for category in MEMORY_CATEGORIES:
                pre = _mean([row[f"pre_{category}_availability"] for row in scoped])
                post = _mean([row[f"post_{category}_availability"] for row in scoped])
                retention = _mean([row[f"{category}_retention"] for row in scoped])
                output.append(
                    {
                        "model": model,
                        "model_label": _short_model(model),
                        "architecture": architecture,
                        "persona_idx": subset[0]["persona_idx"],
                        "scope": scope,
                        "category": category,
                        "context_count": len(scoped),
                        "pre_availability": pre,
                        "post_availability": post,
                        "availability_delta": post - pre if pre is not None and post is not None else None,
                        "retention": retention,
                        "removal": 1 - retention if retention is not None else None,
                        "mean_pre_facts": _mean([row["pre_fact_count"] for row in scoped]),
                        "mean_post_facts": _mean([row["post_fact_count"] for row in scoped]),
                        "mean_pre_attributes": _mean([row["pre_attribute_count"] for row in scoped]),
                        "mean_post_attributes": _mean([row["post_attribute_count"] for row in scoped]),
                        "matching_method": subset[0]["matching_method"],
                        "judge_model": subset[0]["judge_model"],
                    }
                )
    return output


def _memory_reranking_effects(
    rows: list[dict[str, Any]], iterations: int,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    conditions = sorted(
        {(row["model"], row["architecture"]) for row in rows},
        key=_condition_sort_key,
    )
    for model, architecture in conditions:
        subset = [row for row in rows if row["model"] == model and row["architecture"] == architecture]
        for category in MEMORY_CATEGORIES:
            values = {
                (int(row["persona_idx"]), int(row["context_idx"])): float(row[f"delta_{category}_availability"])
                for row in subset
                if _float(row.get(f"delta_{category}_availability")) is not None
            }
            output.append(
                {
                    "model": model,
                    "model_label": _short_model(model),
                    "architecture": architecture,
                    "category": category,
                    **_estimate(
                        values,
                        iterations=iterations,
                        label=f"memory-rerank:{model}:{architecture}:{category}",
                        progress_callback=progress_callback,
                    ),
                }
            )
    return output


def _memory_paired_contrasts(
    rows: list[dict[str, Any]], *, dimension: str, iterations: int,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    if dimension not in {"architecture", "model"}:
        raise ValueError("dimension must be architecture or model")
    fixed = "model" if dimension == "architecture" else "architecture"
    output: list[dict[str, Any]] = []
    for fixed_value in sorted({row[fixed] for row in rows}):
        candidates = sorted(
            {row[dimension] for row in rows if row[fixed] == fixed_value},
            key=(lambda value: ARCHITECTURES.index(value)) if dimension == "architecture" else _model_sort_key,
        )
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                left_rows = {
                    int(row["context_idx"]): row
                    for row in rows if row[fixed] == fixed_value and row[dimension] == left
                }
                right_rows = {
                    int(row["context_idx"]): row
                    for row in rows if row[fixed] == fixed_value and row[dimension] == right
                }
                shared = set(left_rows) & set(right_rows)
                for stage in ("pre", "post"):
                    for category in MEMORY_CATEGORIES:
                        field = f"{stage}_{category}_availability"
                        values = {
                            (int(left_rows[index]["persona_idx"]), index): float(right_rows[index][field]) - float(left_rows[index][field])
                            for index in shared
                            if _float(left_rows[index].get(field)) is not None
                            and _float(right_rows[index].get(field)) is not None
                        }
                        output.append(
                            {
                                "comparison_dimension": dimension,
                                fixed: fixed_value,
                                f"left_{dimension}": left,
                                f"right_{dimension}": right,
                                "stage": stage,
                                "category": category,
                                "contrast": "right_minus_left",
                                **_estimate(
                                    values,
                                    iterations=iterations,
                                    label=f"memory-pair:{dimension}:{fixed_value}:{left}:{right}:{stage}:{category}",
                                    progress_callback=progress_callback,
                                ),
                            }
                        )
    return output


def _memory_interactions(
    rows: list[dict[str, Any]], *, dimension: str, iterations: int,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    if dimension not in {"architecture", "model"}:
        raise ValueError("dimension must be architecture or model")
    fixed = "model" if dimension == "architecture" else "architecture"
    output: list[dict[str, Any]] = []
    for fixed_value in sorted({row[fixed] for row in rows}):
        candidates = sorted(
            {row[dimension] for row in rows if row[fixed] == fixed_value},
            key=(lambda value: ARCHITECTURES.index(value)) if dimension == "architecture" else _model_sort_key,
        )
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                left_rows = {
                    int(row["context_idx"]): row
                    for row in rows if row[fixed] == fixed_value and row[dimension] == left
                }
                right_rows = {
                    int(row["context_idx"]): row
                    for row in rows if row[fixed] == fixed_value and row[dimension] == right
                }
                shared = set(left_rows) & set(right_rows)
                for category in MEMORY_CATEGORIES:
                    field = f"delta_{category}_availability"
                    values = {
                        (int(left_rows[index]["persona_idx"]), index): float(right_rows[index][field]) - float(left_rows[index][field])
                        for index in shared
                        if _float(left_rows[index].get(field)) is not None
                        and _float(right_rows[index].get(field)) is not None
                    }
                    output.append(
                        {
                            "interaction_dimension": dimension,
                            fixed: fixed_value,
                            f"left_{dimension}": left,
                            f"right_{dimension}": right,
                            "category": category,
                            "contrast": "right_rerank_effect_minus_left_rerank_effect",
                            **_estimate(
                                values,
                                iterations=iterations,
                                label=f"memory-interaction:{dimension}:{fixed_value}:{left}:{right}:{category}",
                                progress_callback=progress_callback,
                            ),
                        }
                    )
    return output


def _memory_bootstrap_estimate_count(rows: list[dict[str, Any]]) -> int:
    metric_count = len(MEMORY_CATEGORIES)
    conditions = {(row["model"], row["architecture"]) for row in rows}
    total = len(conditions) * metric_count
    for dimension in ("architecture", "model"):
        fixed = "model" if dimension == "architecture" else "architecture"
        for fixed_value in {row[fixed] for row in rows}:
            candidate_count = len(
                {row[dimension] for row in rows if row[fixed] == fixed_value}
            )
            pairs = candidate_count * (candidate_count - 1) // 2
            total += pairs * (2 * metric_count + metric_count)
    return total


def _join_response_rows(
    memory_rows: list[dict[str, Any]], response_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    response = {
        (row["model"], row["architecture"], int(row["persona_idx"]), int(row["context_idx"])): row
        for row in response_rows
    }
    joined: list[dict[str, Any]] = []
    for memory in memory_rows:
        key = (
            str(memory["model"]),
            str(memory["architecture"]),
            int(memory["persona_idx"]),
            int(memory["context_idx"]),
        )
        response_row = response.get(key)
        if response_row is None:
            raise ValueError(f"No paired response-stage context for direct-memory row {key}")
        row = dict(memory)
        for category, metric in RESPONSE_METRICS.items():
            for stage in ("pre", "post"):
                value = _float(response_row.get(f"{stage}_{metric}"))
                row[f"{stage}_{category}_response_rate"] = value
                memory_value = _float(memory.get(f"{stage}_{category}_availability"))
                row[f"{stage}_{category}_response_minus_memory"] = (
                    value - memory_value if value is not None and memory_value is not None else None
                )
            pre_response = _float(row.get(f"pre_{category}_response_rate"))
            post_response = _float(row.get(f"post_{category}_response_rate"))
            row[f"delta_{category}_response"] = (
                post_response - pre_response
                if pre_response is not None and post_response is not None else None
            )
        joined.append(row)

    alignment: list[dict[str, Any]] = []
    for model, architecture in sorted(
        {(row["model"], row["architecture"]) for row in joined},
        key=_condition_sort_key,
    ):
        subset = [row for row in joined if row["model"] == model and row["architecture"] == architecture]
        for category in MEMORY_CATEGORIES:
            for stage in ("pre", "post"):
                pairs = [
                    (
                        _float(row.get(f"{stage}_{category}_availability")),
                        _float(row.get(f"{stage}_{category}_response_rate")),
                    )
                    for row in subset
                ]
                valid = [(left, right) for left, right in pairs if left is not None and right is not None]
                memory_values = [left for left, _ in valid]
                response_values = [right for _, right in valid]
                memory_mean = mean(memory_values) if memory_values else None
                response_mean = mean(response_values) if response_values else None
                alignment.append(
                    {
                        "model": model,
                        "model_label": _short_model(model),
                        "architecture": architecture,
                        "persona_idx": subset[0]["persona_idx"],
                        "category": category,
                        "stage": stage,
                        "context_pairs": len(valid),
                        "memory_availability": memory_mean,
                        "response_rate": response_mean,
                        "response_minus_memory": (
                            response_mean - memory_mean
                            if memory_mean is not None and response_mean is not None else None
                        ),
                        "response_to_memory_ratio": (
                            response_mean / memory_mean
                            if memory_mean not in (None, 0) and response_mean is not None else None
                        ),
                        "context_pearson": _pearson(memory_values, response_values),
                        "memory_rerank_delta": _mean(
                            [row[f"delta_{category}_availability"] for row in subset]
                        ),
                        "response_rerank_delta": _mean(
                            [row[f"delta_{category}_response"] for row in subset]
                        ),
                    }
                )
    return joined, alignment


def _svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}.title{font-size:23px;font-weight:700}.sub{font-size:11px;fill:#667085}.lab{font-size:11px}.tiny{font-size:9px;fill:#667085}.axis{stroke:#98a2b3}.grid{stroke:#e4e7ec}</style>',
        f'<text class="title" x="24" y="32">{html.escape(title)}</text>',
        f'<text class="sub" x="24" y="51">{html.escape(subtitle)}</text>',
    ]


def _memory_lookup(conditions: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], float]:
    result: dict[tuple[str, str, str, str], float] = {}
    for row in conditions:
        if row["scope"] != "all":
            continue
        for stage in ("pre", "post"):
            value = _float(row.get(f"{stage}_availability"))
            if value is not None:
                result[(row["model"], row["architecture"], row["category"], stage)] = value
    return result


def _write_memory_trajectories(
    path: Path, conditions: list[dict[str, Any]], models: list[str]
) -> None:
    lookup = _memory_lookup(conditions)
    width, height = 1260, 480
    panel_w, panel_h, left, top = 350, 310, 72, 92
    gap = 55
    colors = {model: MODEL_COLORS[index % len(MODEL_COLORS)] for index, model in enumerate(models)}
    svg = _svg_start(
        width,
        height,
        "Direct-memory privacy–utility trajectories",
        "Private availability on x; necessary availability on y; arrows show pre → post reranking",
    )
    for panel, architecture in enumerate(ARCHITECTURES):
        x0 = left + panel * (panel_w + gap)
        svg.append(f'<text class="lab" x="{x0 + panel_w/2}" y="78" text-anchor="middle" font-weight="700">{architecture.title()}</text>')
        for tick in range(0, 101, 20):
            x = x0 + panel_w * tick / 100
            y = top + panel_h - panel_h * tick / 100
            svg.append(f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top+panel_h}"/>')
            svg.append(f'<line class="grid" x1="{x0}" y1="{y}" x2="{x0+panel_w}" y2="{y}"/>')
            svg.append(f'<text class="tiny" x="{x}" y="{top+panel_h+16}" text-anchor="middle">{tick}</text>')
            if panel == 0:
                svg.append(f'<text class="tiny" x="{x0-8}" y="{y+3}" text-anchor="end">{tick}</text>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{top+panel_h}" x2="{x0+panel_w}" y2="{top+panel_h}"/>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{top}" x2="{x0}" y2="{top+panel_h}"/>')
        for model in models:
            keys = [(model, architecture, category, stage) for category in ("private", "necessary") for stage in ("pre", "post")]
            if not all(key in lookup for key in keys):
                continue
            pre_x = x0 + panel_w * lookup[(model, architecture, "private", "pre")]
            pre_y = top + panel_h * (1 - lookup[(model, architecture, "necessary", "pre")])
            post_x = x0 + panel_w * lookup[(model, architecture, "private", "post")]
            post_y = top + panel_h * (1 - lookup[(model, architecture, "necessary", "post")])
            color = colors[model]
            svg.append(f'<line x1="{pre_x}" y1="{pre_y}" x2="{post_x}" y2="{post_y}" stroke="{color}" stroke-width="2.5"/>')
            svg.append(f'<circle cx="{pre_x}" cy="{pre_y}" r="5" fill="white" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<circle cx="{post_x}" cy="{post_y}" r="5" fill="{color}"/>')
    svg.append(f'<text class="lab" x="{width/2}" y="{height-27}" text-anchor="middle">Private attributes available in returned memory (%)</text>')
    svg.append(f'<text class="lab" transform="translate(18 {top+panel_h/2}) rotate(-90)" text-anchor="middle">Necessary attributes available (%)</text>')
    for index, model in enumerate(models):
        x = 855 + index * 130
        svg.append(f'<circle cx="{x}" cy="35" r="5" fill="{colors[model]}"/><text class="tiny" x="{x+9}" y="39">{html.escape(_short_model(model))}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_density(path: Path, conditions: list[dict[str, Any]], models: list[str]) -> None:
    base = [row for row in conditions if row["scope"] == "all" and row["category"] == "necessary"]
    width, height = 1260, 480
    panel_w, panel_h, left, top, gap = 350, 310, 72, 92, 55
    colors = {model: MODEL_COLORS[index % len(MODEL_COLORS)] for index, model in enumerate(models)}
    svg = _svg_start(width, height, "Representation density", "Returned facts on x; distinct represented attributes on y; arrows show pre → post")
    for panel, architecture in enumerate(ARCHITECTURES):
        x0 = left + panel * (panel_w + gap)
        subset = [row for row in base if row["architecture"] == architecture]
        max_x = max([float(row["mean_pre_facts"] or 0) for row in subset] + [1]) * 1.1
        max_y = max([float(row["mean_pre_attributes"] or 0) for row in subset] + [1]) * 1.1
        svg.append(f'<text class="lab" x="{x0+panel_w/2}" y="78" text-anchor="middle" font-weight="700">{architecture.title()}</text>')
        for tick in range(6):
            x = x0 + panel_w * tick / 5
            y = top + panel_h - panel_h * tick / 5
            svg.append(f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top+panel_h}"/>')
            svg.append(f'<line class="grid" x1="{x0}" y1="{y}" x2="{x0+panel_w}" y2="{y}"/>')
            svg.append(f'<text class="tiny" x="{x}" y="{top+panel_h+16}" text-anchor="middle">{max_x*tick/5:.0f}</text>')
            if panel == 0:
                svg.append(f'<text class="tiny" x="{x0-8}" y="{y+3}" text-anchor="end">{max_y*tick/5:.0f}</text>')
        for row in subset:
            px = x0 + panel_w * float(row["mean_pre_facts"] or 0) / max_x
            py = top + panel_h * (1 - float(row["mean_pre_attributes"] or 0) / max_y)
            qx = x0 + panel_w * float(row["mean_post_facts"] or 0) / max_x
            qy = top + panel_h * (1 - float(row["mean_post_attributes"] or 0) / max_y)
            color = colors[row["model"]]
            svg.append(f'<line x1="{px}" y1="{py}" x2="{qx}" y2="{qy}" stroke="{color}" stroke-width="2.5"/>')
            svg.append(f'<circle cx="{px}" cy="{py}" r="5" fill="white" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<circle cx="{qx}" cy="{qy}" r="5" fill="{color}"/>')
    svg.append(f'<text class="lab" x="{width/2}" y="{height-27}" text-anchor="middle">Mean returned memory facts</text>')
    svg.append(f'<text class="lab" transform="translate(18 {top+panel_h/2}) rotate(-90)" text-anchor="middle">Mean represented attributes</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_alignment(path: Path, alignment: list[dict[str, Any]], models: list[str]) -> None:
    rows = [row for row in alignment if row["stage"] == "post"]
    conditions = [(model, architecture) for model in models for architecture in ARCHITECTURES]
    width, height = 1260, 410
    left, top, panel_w, gap = 185, 82, 300, 55
    colors = {"memory": "#3559a8", "response": "#d65f45"}
    svg = _svg_start(width, height, "Post-rerank memory-to-response alignment", "Dumbbells connect attribute availability in returned memory to attribute appearance in the generated response")
    for panel, category in enumerate(MEMORY_CATEGORIES):
        x0 = left + panel * (panel_w + gap)
        svg.append(f'<text class="lab" x="{x0+panel_w/2}" y="70" text-anchor="middle" font-weight="700">{CATEGORY_LABELS[category]}</text>')
        for tick in range(0, 101, 20):
            x = x0 + panel_w * tick / 100
            svg.append(f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top+250}"/>')
            svg.append(f'<text class="tiny" x="{x}" y="{top+269}" text-anchor="middle">{tick}</text>')
        for index, (model, architecture) in enumerate(conditions):
            y = top + 14 + index * 27
            row = next((item for item in rows if item["model"] == model and item["architecture"] == architecture and item["category"] == category), None)
            if row is None:
                continue
            memory = float(row["memory_availability"] or 0)
            response = float(row["response_rate"] or 0)
            mx, rx = x0 + panel_w * memory, x0 + panel_w * response
            svg.append(f'<line x1="{mx}" y1="{y}" x2="{rx}" y2="{y}" stroke="#98a2b3" stroke-width="2"/>')
            svg.append(f'<circle cx="{mx}" cy="{y}" r="4.5" fill="{colors["memory"]}"/>')
            svg.append(f'<rect x="{rx-4}" y="{y-4}" width="8" height="8" fill="{colors["response"]}"/>')
            if panel == 0:
                label = f"{_short_model(model)} · {architecture.title()}"
                svg.append(f'<text class="tiny" x="{x0-10}" y="{y+3}" text-anchor="end">{html.escape(label)}</text>')
    svg.append(f'<circle cx="980" cy="35" r="4.5" fill="{colors["memory"]}"/><text class="tiny" x="990" y="39">Returned memory</text>')
    svg.append(f'<rect x="1080" y="31" width="8" height="8" fill="{colors["response"]}"/><text class="tiny" x="1093" y="39">Generated response</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def _pct(value: Any) -> str:
    number = _float(value)
    return "--" if number is None else f"{100 * number:.1f}"


def _latex_artifacts(
    output: Path,
    conditions: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    efficiency: list[dict[str, Any]],
) -> list[str]:
    includes = output / "includes"
    includes.mkdir()
    all_rows = [row for row in conditions if row["scope"] == "all"]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Model & Memory & Facts & Attrs. & Necessary & Private & Ambiguous \\", r"\midrule",
    ]
    for model, architecture in sorted(
        {(row["model"], row["architecture"]) for row in all_rows},
        key=_condition_sort_key,
    ):
        values = {row["category"]: row for row in all_rows if row["model"] == model and row["architecture"] == architecture}
        base = values["necessary"]
        facts = f"{base['mean_pre_facts']:.1f}$\\rightarrow${base['mean_post_facts']:.1f}"
        attrs = f"{base['mean_pre_attributes']:.1f}$\\rightarrow${base['mean_post_attributes']:.1f}"
        cats = [f"{_pct(values[cat]['pre_availability'])}$\\rightarrow${_pct(values[cat]['post_availability'])}" for cat in MEMORY_CATEGORIES]
        lines.append(f"{_short_model(model)} & {architecture.title()} & {facts} & {attrs} & " + " & ".join(cats) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Direct-memory operating points for the selected case-study persona. Values are macro percentages before $\rightarrow$ after reranking.}",
        r"\label{tab:direct-memory-operating-points-export}", r"\end{table*}", "",
    ]
    (includes / "direct_memory_operating_points.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrr}",
        r"\toprule", r"Model & Memory & Necessary retained & Private retained & Ambiguous retained \\", r"\midrule",
    ]
    for model, architecture in sorted(
        {(row["model"], row["architecture"]) for row in all_rows},
        key=_condition_sort_key,
    ):
        values = {row["category"]: row for row in all_rows if row["model"] == model and row["architecture"] == architecture}
        lines.append(
            f"{_short_model(model)} & {architecture.title()} & "
            + " & ".join(_pct(values[category]["retention"]) for category in MEMORY_CATEGORIES)
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Attribute retention from pre- to post-rerank memory.}", r"\label{tab:direct-memory-retention}", r"\end{table*}", ""]
    (includes / "direct_memory_retention.tex").write_text("\n".join(lines), encoding="utf-8")

    post = [row for row in alignment if row["stage"] == "post"]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrrrrr}",
        r"\toprule", r"Model & Memory & \multicolumn{2}{c}{Necessary} & \multicolumn{2}{c}{Private} & \multicolumn{2}{c}{Ambiguous} \\",
        r" & & Memory & Response & Memory & Response & Memory & Response \\", r"\midrule",
    ]
    for model, architecture in sorted(
        {(row["model"], row["architecture"]) for row in post},
        key=_condition_sort_key,
    ):
        values = {row["category"]: row for row in post if row["model"] == model and row["architecture"] == architecture}
        cells: list[str] = []
        for category in MEMORY_CATEGORIES:
            cells.extend((_pct(values[category]["memory_availability"]), _pct(values[category]["response_rate"])))
        lines.append(f"{_short_model(model)} & {architecture.title()} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Post-rerank attribute availability in returned memory and appearance in generated responses.}", r"\label{tab:memory-response-alignment}", r"\end{table*}", ""]
    (includes / "memory_response_alignment.tex").write_text("\n".join(lines), encoding="utf-8")

    efficiency_fields = (
        "memory_preparation_seconds",
        "generation_seconds",
        "deployed_seconds",
        "reported_exact_tokens_per_query",
        "initialization_seconds",
    )
    minima = {
        field: min(
            value
            for value in (_float(row.get(field)) for row in efficiency)
            if value is not None
        )
        for field in efficiency_fields
        if any(_float(row.get(field)) is not None for row in efficiency)
    }

    def efficiency_cell(field: str, value: Any) -> str:
        number = _float(value)
        if number is None:
            return "--"
        displayed = number / 60 if field == "initialization_seconds" else number
        digits = 0 if field == "reported_exact_tokens_per_query" else 1
        rendered = f"{displayed:.{digits}f}"
        if field in minima and math.isclose(number, minima[field], rel_tol=1e-9, abs_tol=1e-9):
            return rf"\textbf{{{rendered}}}"
        return rendered

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Memory & Cells & Preparation (s) & Generation (s) & Deployed (s) & Reported tokens/query & Init. (min) \\",
        r"\midrule",
    ]
    for row in efficiency:
        token_cell = efficiency_cell(
            "reported_exact_tokens_per_query",
            row.get("reported_exact_tokens_per_query"),
        )
        token_cell += f" ({row['reported_token_cells']}/{row['pipeline_cells']})"
        cells = [
            efficiency_cell("memory_preparation_seconds", row.get("memory_preparation_seconds")),
            efficiency_cell("generation_seconds", row.get("generation_seconds")),
            efficiency_cell("deployed_seconds", row.get("deployed_seconds")),
            token_cell,
            efficiency_cell("initialization_seconds", row.get("initialization_seconds")),
        ]
        lines.append(
            f"{str(row['architecture']).title()} & {row['pipeline_cells']} & "
            + " & ".join(cells)
            + r" \\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Architecture-level efficiency of the deployed post-rerank pipeline. Values are macro-means over model--persona pipeline cells, so response repetitions do not change a cell's weight. Preparation covers retrieval and explicit reranking; deployed latency combines preparation and generation. Reported tokens include exact provider-reported generation and preparation usage but exclude backend operations without telemetry; token means use the cross-architecture matched telemetry cohort, shown in parentheses over all cells. Lower values are bold.}",
        r"\label{tab:cimemories-architecture-efficiency}", r"\end{table*}", "",
    ]
    (includes / "architecture_efficiency.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    figure_names = (
        "direct_memory_trajectories",
        "memory_representation_density",
        "memory_to_response_alignment",
    )
    for name in figure_names:
        caption = name.replace("_", " ").title()
        (includes / f"{name}.tex").write_text(
            "\\begin{figure*}[t]\n\\centering\n"
            f"\\includesvg[width=0.98\\textwidth]{{figures/{name}.svg}}\n"
            f"\\caption{{{caption}.}}\n\\label{{fig:{name.replace('_', '-')}}}\n\\end{{figure*}}\n",
            encoding="utf-8",
        )
    (includes / "paper_preamble.tex").write_text(
        "\\usepackage{booktabs}\n\\usepackage{graphicx}\n\\usepackage{svg}\n\\svgsetup{inkscapelatex=false}\n",
        encoding="utf-8",
    )
    (includes / "all_direct_memory_artifacts.tex").write_text(
        "\\input{includes/direct_memory_operating_points}\n"
        "\\input{includes/direct_memory_retention}\n"
        "\\input{includes/memory_response_alignment}\n"
        + "".join(f"\\input{{includes/{name}}}\n" for name in figure_names),
        encoding="utf-8",
    )
    (includes / "all_integrated_artifacts.tex").write_text(
        "\\input{includes/architecture_efficiency}\n"
        "\\input{includes/all_direct_memory_artifacts}\n",
        encoding="utf-8",
    )
    return [
        "includes/paper_preamble.tex",
        "includes/architecture_efficiency.tex",
        "includes/direct_memory_operating_points.tex",
        "includes/direct_memory_retention.tex",
        "includes/memory_response_alignment.tex",
        "includes/all_direct_memory_artifacts.tex",
        "includes/all_integrated_artifacts.tex",
        *[f"includes/{name}.tex" for name in figure_names],
    ]


def _html_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in fields)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key, _ in fields) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _report_html(
    *,
    coverage: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    memory_persona: int,
    judge_model: str,
    efficiency: list[dict[str, Any]],
    analysis_provenance: dict[str, Any] | None = None,
    first_post_repeat: bool = False,
) -> str:
    cards = "".join(
        f"<div class='card {'ok' if row['complete'] else 'missing'}'><b>{html.escape(row['model_label'])}</b><span>{html.escape(str(row['architecture']).title())}</span><strong>{'ready' if row['complete'] else 'missing'}</strong></div>"
        for row in coverage
    )
    memory_table: list[dict[str, Any]] = []
    all_rows = [row for row in conditions if row["scope"] == "all"]
    for model, architecture in sorted(
        {(row["model"], row["architecture"]) for row in all_rows},
        key=_condition_sort_key,
    ):
        values = {row["category"]: row for row in all_rows if row["model"] == model and row["architecture"] == architecture}
        memory_table.append(
            {
                "model": _short_model(model),
                "memory": architecture.title(),
                "facts": f"{values['necessary']['mean_pre_facts']:.1f} → {values['necessary']['mean_post_facts']:.1f}",
                "attributes": f"{values['necessary']['mean_pre_attributes']:.1f} → {values['necessary']['mean_post_attributes']:.1f}",
                **{category: f"{_pct(values[category]['pre_availability'])} → {_pct(values[category]['post_availability'])}" for category in MEMORY_CATEGORIES},
            }
        )
    post_alignment = [
        {
            "model": row["model_label"],
            "memory": str(row["architecture"]).title(),
            "category": CATEGORY_LABELS[row["category"]],
            "available": _pct(row["memory_availability"]),
            "response": _pct(row["response_rate"]),
            "correlation": "--" if row["context_pearson"] is None else f"{row['context_pearson']:.2f}",
        }
        for row in alignment if row["stage"] == "post"
    ]
    efficiency_table = [
        {
            "memory": str(row["architecture"]).title(),
            "cells": row["pipeline_cells"],
            "preparation": "--" if row["memory_preparation_seconds"] is None else f"{row['memory_preparation_seconds']:.1f}",
            "generation": "--" if row["generation_seconds"] is None else f"{row['generation_seconds']:.1f}",
            "deployed": "--" if row["deployed_seconds"] is None else f"{row['deployed_seconds']:.1f}",
            "tokens": (
                f"-- ({row['reported_token_cells']}/{row['pipeline_cells']})"
                if row["reported_exact_tokens_per_query"] is None
                else f"{row['reported_exact_tokens_per_query']:.0f} ({row['reported_token_cells']}/{row['pipeline_cells']})"
            ),
            "initialization": "--" if row["initialization_seconds"] is None else f"{row['initialization_seconds'] / 60:.1f}",
        }
        for row in efficiency
    ]
    provenance_notice = ""
    repeat_notice = ""
    if first_post_repeat:
        repeat_notice = (
            "<section><h2>Matched-repeat response analysis</h2><p>Post-rerank response "
            "metrics use only the first saved repetition per context (lowest repeat index). "
            "Pre-rerank metrics and direct-memory metrics are unchanged. This makes the "
            "response comparison one generation per context for every model.</p></section>"
        )
    if analysis_provenance:
        label_name = html.escape(str(analysis_provenance.get("ground_truth_name") or "supplied ground truth"))
        label_path = html.escape(str(analysis_provenance.get("ground_truth_file") or ""))
        label_hash = html.escape(str(analysis_provenance.get("ground_truth_sha256") or ""))
        provenance_notice = (
            "<section><h2>Alternate ground-truth sensitivity analysis</h2>"
            f"<p>All label-dependent response and direct-memory metrics were recomputed with "
            f"<strong>{label_name}</strong> from <code>{label_path}</code> "
            f"(SHA-256 <code>{label_hash}</code>). Generated responses, exposure judgments, "
            "retrieval results, reranking results, and direct-memory attribute matches were reused "
            "unchanged. See <code>ground_truth_analysis/</code> for label support and comparison tables.</p></section>"
        )
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CIMemories integrated paper artifacts</title><style>
    :root{{--ink:#172033;--muted:#667085;--line:#d0d5dd;--paper:#f4f6fa;--blue:#3559a8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,system-ui,sans-serif}}header{{padding:52px max(5vw,28px);background:linear-gradient(120deg,#172033,#314f86);color:white}}header h1{{margin:0 0 10px;font-size:35px}}header p{{max-width:980px;line-height:1.55;color:#dce4ff}}nav{{position:sticky;top:0;background:white;border-bottom:1px solid var(--line);padding:12px 5vw;z-index:2}}nav a{{margin-right:20px;color:#344b85;text-decoration:none;font-weight:650}}main{{max-width:1450px;margin:auto;padding:30px}}section{{background:white;border:1px solid #e3e7ee;border-radius:14px;padding:26px;margin-bottom:24px;box-shadow:0 8px 28px #1720330b}}h2{{margin-top:0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.card{{display:grid;grid-template-columns:1fr auto;gap:4px;padding:14px;border-radius:10px;border:1px solid #c8e8d8;background:#eef8f3}}.card.missing{{background:#fff1ee;border-color:#f0c1b7}}.card span{{color:var(--muted)}}.card strong{{grid-row:1/3;grid-column:2;align-self:center}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0;border:1px solid #e4e7ec;border-radius:10px;padding:10px}}figure img{{width:100%;height:auto}}figcaption{{font-size:13px;color:var(--muted);padding:8px}}.table-wrap{{overflow:auto;max-height:650px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{position:sticky;top:0;background:#eef2f8;text-align:left}}th,td{{padding:9px 11px;border-bottom:1px solid #e4e7ec;white-space:nowrap}}code{{background:#eef2f8;padding:2px 5px;border-radius:4px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
    </style></head><body><header><h1>CIMemories integrated paper artifacts</h1><p>One offline bundle connecting architecture comparisons, model comparisons, pre/post-reranking effects, direct returned-memory availability, and memory-to-response behavior.</p></header><nav><a href='#map'>Study map</a><a href='#response'>Response analysis</a><a href='#efficiency'>Efficiency</a><a href='#memory'>Direct memory</a><a href='#cross'>Cross-stage</a><a href='#provenance'>Provenance</a></nav><main>
    {provenance_notice}{repeat_notice}<section id='map'><h2>Analysis structure</h2><p>The response layer contains the complete ten-persona factorial analysis. The direct-memory layer is a mechanistic case study for persona {memory_persona}, evaluated with exact list matching and <code>{html.escape(judge_model)}</code> semantic judging for graph/profile outputs. Cross-stage tables align memory availability with generated-message exposure without treating either layer as interchangeable ground truth.</p><div class='cards'>{cards}</div></section>
    <section id='response'><h2>Response-level architecture, model, and reranking effects</h2><p>The preserved legacy factorial bundle contains operating points, paired architecture and model contrasts, difference-in-differences, hierarchical intervals, persona heterogeneity, recipient/task-domain heterogeneity, latency, cost, and exact source provenance.</p><p><a href='response_analysis/REPORT.html'><strong>Open the complete response-level report →</strong></a></p><div class='grid'><figure><img src='response_analysis/figures/privacy_utility_trajectories.svg'><figcaption>Architecture and model operating points before and after reranking.</figcaption></figure><figure><img src='response_analysis/figures/reranking_effect_forest.svg'><figcaption>Paired reranking effects and hierarchical intervals.</figcaption></figure><figure><img src='response_analysis/figures/reranking_interaction_heatmap.svg'><figcaption>Architecture × model × reranking interactions.</figcaption></figure><figure><img src='response_analysis/figures/persona_heterogeneity.svg'><figcaption>Variation across personas.</figcaption></figure></div></section>
    <section id='efficiency'><h2>Architecture-level efficiency</h2><p>This compact view macro-averages model–persona pipeline cells, giving every cell equal weight regardless of response repetitions. Preparation includes retrieval and explicit reranking; deployed latency combines preparation and generation. Token counts include exact reported generation and preparation usage, while backend operations without telemetry remain excluded. Token means use only model–persona cells reported for all architectures; parentheses show that matched cohort over all cells.</p>{_html_table(efficiency_table, [('memory','Memory'),('cells','Cells'),('preparation','Preparation s'),('generation','Generation s'),('deployed','Deployed s'),('tokens','Reported tokens/query (matched)'),('initialization','Initialization min')])}<p>Raw per-cell measurements and telemetry coverage are in <code>efficiency_analysis/pipeline_efficiency.csv</code>; architecture macro-means are in <code>efficiency_analysis/architecture_efficiency.csv</code>.</p></section>
    <section id='memory'><h2>Direct returned-memory analysis</h2><p>These artifacts measure information made available to the generator, before message generation. They separate initial representation breadth from query-dependent selection.</p><div class='grid'><figure><img src='figures/direct_memory_trajectories.svg'><figcaption>Privacy–utility movement in returned memory.</figcaption></figure><figure><img src='figures/memory_representation_density.svg'><figcaption>Facts versus represented attributes exposes bundling and redundancy.</figcaption></figure></div>{_html_table(memory_table, [('model','Model'),('memory','Memory'),('facts','Facts pre → post'),('attributes','Attributes pre → post'),('necessary','Necessary %'),('private','Private %'),('ambiguous','Ambiguous %')])}<p>Analysis-ready paired comparisons are saved in <code>memory_analysis/architecture_contrasts.csv</code>, <code>model_contrasts.csv</code>, <code>reranking_effects.csv</code>, and <code>interaction_contrasts.csv</code>. Each contrast records its exact shared context count and paired-bootstrap interval.</p></section>
    <section id='cross'><h2>Memory-to-response cross-effects</h2><p>This layer asks whether attributes available in returned memory subsequently appear in generated messages. Differences describe stage-wise filtering; they are not interpreted causally because response generation is stochastic and the two layers use separate semantic judges.</p><figure><img src='figures/memory_to_response_alignment.svg'><figcaption>Post-rerank availability in memory versus appearance in the final response.</figcaption></figure>{_html_table(post_alignment, [('model','Model'),('memory','Memory'),('category','Category'),('available','Memory %'),('response','Response %'),('correlation','Context r')])}</section>
    <section id='provenance'><h2>Reproducibility</h2><p><code>integrated_artifact_inputs.json</code> locks every response pipeline and memory-stage summary. <code>manifest.json</code> inventories all generated artifacts. The export is entirely offline and makes no model calls.</p><ul><li><code>response_analysis/</code>: unchanged legacy factorial export.</li><li><code>efficiency_analysis/</code>: architecture macro-means and source pipeline-cell measurements.</li><li><code>memory_analysis/</code>: context and condition-level direct-memory data.</li><li><code>cross_stage_analysis/</code>: joined memory/response rows and aggregate alignment.</li><li><code>includes/</code>: generated LaTeX tables and figure snippets for paper preparation.</li></ul></section>
    </main></body></html>"""


def export_cimemories_integrated_paper_artifacts(
    *,
    output_root: Path = Path("research_outputs"),
    dataset_filter: str | None = "cimemories_raw",
    output: Path | None = None,
    inputs: Path | None = None,
    bootstrap_iterations: int = 5000,
    expected_personas: int = 10,
    require_complete: bool = False,
    require_memory_complete: bool = False,
    model_filters: list[str] | None = None,
    memory_persona: int = 0,
    memory_judge_model: str = "gpt-6-sol",
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    response_row_transform: Callable[[list[dict[str, Any]], list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    memory_row_transform: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    analysis_provenance: dict[str, Any] | None = None,
    first_post_repeat: bool = False,
) -> Path:
    if memory_persona < 0:
        raise ValueError("memory_persona must be non-negative")
    if not memory_judge_model.strip():
        raise ValueError("memory_judge_model cannot be empty")
    if inputs is not None and model_filters:
        raise ValueError("--models cannot be combined with --inputs; the lockfile already fixes models")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output or output_root / f"cimemories-integrated-paper-artifacts_{timestamp}"
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")

    locked_payload = _read_json(inputs) if inputs is not None else None
    if locked_payload is not None:
        memory_persona = int(locked_payload.get("memory_persona", memory_persona))
        memory_judge_model = str(locked_payload.get("memory_judge_model") or memory_judge_model)
        if (locked_payload.get("response_repeat_policy") or {}).get("post_repeat_limit") == 1:
            first_post_repeat = True

    response_output = output / "response_analysis"
    export_cimemories_study_artifacts(
        output_root=output_root,
        dataset_filter=dataset_filter,
        output=response_output,
        inputs=inputs,
        bootstrap_iterations=bootstrap_iterations,
        expected_personas=expected_personas,
        require_complete=require_complete,
        model_filters=model_filters,
        progress_callback=progress_callback,
        row_transform=response_row_transform,
        analysis_provenance=analysis_provenance,
        post_repeat_limit=1 if first_post_repeat else None,
    )
    response_lock = _read_json(response_output / "study_artifact_inputs.json")
    selected_runs = list(response_lock.get("selected_runs") or [])
    efficiency_summary, efficiency_details = _architecture_efficiency(selected_runs)

    locked_memory: dict[tuple[str, str, int], str] = {}
    if locked_payload is not None:
        for row in locked_payload.get("memory_stage_runs") or []:
            locked_memory[(str(row["model"]), str(row["architecture"]), int(row["persona_idx"]))] = str(row["summary_path"])

    if progress_callback is not None:
        progress_callback(
            "memory-inputs", 0, 1,
            f"selecting persona {memory_persona} direct-memory summaries",
        )
    memory_rows, memory_provenance, memory_coverage = _collect_memory_rows(
        selected_runs,
        memory_persona=memory_persona,
        judge_model=memory_judge_model,
        locked_memory=locked_memory or None,
    )
    if memory_row_transform is not None:
        memory_rows = memory_row_transform(memory_rows)
    missing = [row for row in memory_coverage if not row["complete"]]
    if require_memory_complete and missing:
        descriptions = ", ".join(f"{row['model_label']}/{row['architecture']}" for row in missing)
        raise ValueError(f"Direct-memory case study is incomplete for persona {memory_persona}: {descriptions}")
    if not memory_rows:
        raise ValueError(
            f"No direct-memory summaries found for persona {memory_persona} and judge {memory_judge_model}"
        )
    if progress_callback is not None:
        progress_callback(
            "memory-inputs", 1, 1,
            f"loaded {len(memory_provenance)} conditions and {len(memory_rows)} context rows",
        )

    memory_conditions = _memory_conditions(memory_rows)
    memory_bootstrap_total = _memory_bootstrap_estimate_count(memory_rows)
    memory_bootstrap_done = 0

    def memory_bootstrap_progress(label: str) -> None:
        nonlocal memory_bootstrap_done
        memory_bootstrap_done += 1
        if progress_callback is not None:
            progress_callback(
                "memory-bootstrap",
                memory_bootstrap_done,
                memory_bootstrap_total,
                label,
            )

    if progress_callback is not None:
        progress_callback(
            "memory-bootstrap", 0, memory_bootstrap_total,
            f"starting {bootstrap_iterations:,}-resample intervals",
        )
    memory_effects = _memory_reranking_effects(
        memory_rows,
        bootstrap_iterations,
        progress_callback=memory_bootstrap_progress,
    )
    memory_architecture_contrasts = _memory_paired_contrasts(
        memory_rows, dimension="architecture", iterations=bootstrap_iterations,
        progress_callback=memory_bootstrap_progress,
    )
    memory_model_contrasts = _memory_paired_contrasts(
        memory_rows, dimension="model", iterations=bootstrap_iterations,
        progress_callback=memory_bootstrap_progress,
    )
    memory_interactions = _memory_interactions(
        memory_rows, dimension="architecture", iterations=bootstrap_iterations,
        progress_callback=memory_bootstrap_progress,
    ) + _memory_interactions(
        memory_rows, dimension="model", iterations=bootstrap_iterations,
        progress_callback=memory_bootstrap_progress,
    )
    if progress_callback is not None:
        progress_callback("cross-stage", 0, 1, "joining memory and response contexts")
    response_rows = _read_csv(response_output / "context_metrics.csv")
    joined, alignment = _join_response_rows(memory_rows, response_rows)
    if progress_callback is not None:
        progress_callback(
            "cross-stage", 1, 1,
            f"joined {len(joined)} exact model/architecture/persona/context rows",
        )

    memory_dir = output / "memory_analysis"
    cross_dir = output / "cross_stage_analysis"
    efficiency_dir = output / "efficiency_analysis"
    figures = output / "figures"
    memory_dir.mkdir()
    cross_dir.mkdir()
    efficiency_dir.mkdir()
    figures.mkdir()
    _write_csv(efficiency_dir / "architecture_efficiency.csv", efficiency_summary)
    _write_csv(efficiency_dir / "pipeline_efficiency.csv", efficiency_details)
    _write_csv(memory_dir / "context_metrics.csv", memory_rows)
    _write_csv(memory_dir / "condition_metrics.csv", memory_conditions)
    _write_csv(memory_dir / "reranking_effects.csv", memory_effects)
    _write_csv(memory_dir / "architecture_contrasts.csv", memory_architecture_contrasts)
    _write_csv(memory_dir / "model_contrasts.csv", memory_model_contrasts)
    _write_csv(memory_dir / "interaction_contrasts.csv", memory_interactions)
    _write_csv(memory_dir / "coverage.csv", memory_coverage)
    _write_csv(memory_dir / "provenance.csv", memory_provenance)
    _write_csv(cross_dir / "memory_response_contexts.csv", joined)
    _write_csv(cross_dir / "memory_response_alignment.csv", alignment)

    models = sorted({str(row["model"]) for row in memory_rows}, key=_model_sort_key)
    if progress_callback is not None:
        progress_callback("integrated-figures", 0, 3, "rendering direct-memory figures")
    _write_memory_trajectories(figures / "direct_memory_trajectories.svg", memory_conditions, models)
    if progress_callback is not None:
        progress_callback("integrated-figures", 1, 3, "direct-memory trajectories")
    _write_density(figures / "memory_representation_density.svg", memory_conditions, models)
    if progress_callback is not None:
        progress_callback("integrated-figures", 2, 3, "memory representation density")
    _write_alignment(figures / "memory_to_response_alignment.svg", alignment, models)
    if progress_callback is not None:
        progress_callback("integrated-figures", 3, 3, "memory-to-response alignment")
        progress_callback("integrated-artifacts", 0, 1, "writing LaTeX and provenance")
    latex = _latex_artifacts(
        output, memory_conditions, alignment, efficiency_summary
    )

    lock = {
        "schema_version": 1,
        "command": "export_cimemories_integrated_paper_artifacts",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_filter": dataset_filter,
        "model_filters": model_filters,
        "expected_personas": expected_personas,
        "memory_persona": memory_persona,
        "memory_judge_model": memory_judge_model,
        "response_repeat_policy": {
            "post_repeat_limit": 1 if first_post_repeat else None,
            "selection": "lowest_repeat_idx_then_record_index",
            "pre_rerank_unchanged": True,
        },
        "selected_runs": selected_runs,
        "memory_stage_runs": memory_provenance,
    }
    if analysis_provenance:
        lock["analysis_provenance"] = analysis_provenance
    (output / "integrated_artifact_inputs.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    artifacts = [
        "REPORT.html",
        "integrated_artifact_inputs.json",
        "response_analysis/",
        "efficiency_analysis/architecture_efficiency.csv",
        "efficiency_analysis/pipeline_efficiency.csv",
        "memory_analysis/context_metrics.csv",
        "memory_analysis/condition_metrics.csv",
        "memory_analysis/reranking_effects.csv",
        "memory_analysis/architecture_contrasts.csv",
        "memory_analysis/model_contrasts.csv",
        "memory_analysis/interaction_contrasts.csv",
        "memory_analysis/coverage.csv",
        "memory_analysis/provenance.csv",
        "cross_stage_analysis/memory_response_contexts.csv",
        "cross_stage_analysis/memory_response_alignment.csv",
        "figures/direct_memory_trajectories.svg",
        "figures/memory_representation_density.svg",
        "figures/memory_to_response_alignment.svg",
        *latex,
    ]
    manifest = {
        "schema_version": 1,
        "command": "export_cimemories_integrated_paper_artifacts",
        "created_at": lock["created_at"],
        "dataset_filter": dataset_filter,
        "models": models,
        "architectures": list(ARCHITECTURES),
        "response_persona_scope": expected_personas,
        "memory_persona": memory_persona,
        "memory_judge_model": memory_judge_model,
        "response_repeat_policy": lock["response_repeat_policy"],
        "response_bundle": "response_analysis/manifest.json",
        "efficiency": {
            "aggregation": "macro_mean_over_model_persona_pipeline_cells",
            "stage": "post_rerank_deployed_pipeline",
            "preparation_scope": "retrieval_and_explicit_reranking",
            "token_scope": "exact_provider_reported_generation_and_preparation",
            "unreported_backend_operations_excluded": True,
        },
        "memory_complete": not missing,
        "missing_memory_conditions": missing,
        "artifacts": artifacts,
    }
    if analysis_provenance:
        manifest["analysis_provenance"] = analysis_provenance
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.html").write_text(
        _report_html(
            coverage=memory_coverage,
            conditions=memory_conditions,
            alignment=alignment,
            memory_persona=memory_persona,
            judge_model=memory_judge_model,
            efficiency=efficiency_summary,
            analysis_provenance=analysis_provenance,
            first_post_repeat=first_post_repeat,
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# CIMemories integrated paper artifacts\n\n"
        "Open `REPORT.html` for the integrated visual report. This bundle is offline and makes no model calls.\n\n"
        "- `response_analysis/` preserves the complete legacy factorial response export.\n"
        "- `efficiency_analysis/` contains architecture-level macro-means and the underlying model-persona pipeline measurements.\n"
        "- `memory_analysis/` contains direct pre/post-rerank memory availability, paired architecture/model contrasts, reranking effects, and difference-in-differences.\n"
        "- `cross_stage_analysis/` joins returned-memory availability to response exposure.\n"
        "- `figures/` and `includes/` contain generated SVG and LaTeX aids for paper preparation.\n"
        "- `integrated_artifact_inputs.json` locks every selected input for exact regeneration.\n\n"
        + ("Response metrics use one post-rerank generation per context (the lowest saved repeat index); pre-rerank and direct-memory metrics are unchanged.\n" if first_post_repeat else "Response metrics use all saved post-rerank repetitions.\n"),
        encoding="utf-8",
    )
    if progress_callback is not None:
        progress_callback("integrated-artifacts", 1, 1, "integrated bundle complete")
    return output.resolve()
