from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .comparative_report import (
    ARCHITECTURES,
    METRICS,
    _finite,
    _locked_entries,
    _pipeline_metrics,
    _pre_metrics,
    _read_json,
    _resolve,
    _select_entries,
)


MODEL_COLORS = ("#3559a8", "#d65f45", "#2b8a6e", "#8b5fbf", "#b28a24")
ARCH_COLORS = {"list": "#e76f51", "graph": "#2a9d8f", "profile": "#4568dc"}
METRIC_LABELS = {
    "necessary_recall": "Necessary recall",
    "inappropriate_leak_rate": "Inappropriate leakage",
    "ambiguous_exposure_rate": "Ambiguous exposure",
    "average_exposed_total": "Exposed attributes",
}


def _short_model(model: str) -> str:
    if model == "gpt-5.6-sol":
        return "GPT-5.6-sol"
    if "GLM-5.3" in model:
        return "GLM-5.3-Flash"
    if "DeepSeek-V4" in model:
        return "DeepSeek-V4-Flash"
    return model.rsplit("/", 1)[-1]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:12], 16)


def _hierarchical_interval(
    values: dict[tuple[int, int], float], *, iterations: int, label: str
) -> tuple[float, float]:
    by_persona: dict[int, list[float]] = {}
    for (persona, _), value in values.items():
        by_persona.setdefault(persona, []).append(value)
    personas = sorted(by_persona)
    if not personas:
        return float("nan"), float("nan")
    rng = random.Random(_seed(label))
    estimates: list[float] = []
    for _ in range(iterations):
        draw: list[float] = []
        for persona in rng.choices(personas, k=len(personas)):
            contexts = by_persona[persona]
            draw.extend(rng.choices(contexts, k=len(contexts)))
        estimates.append(mean(draw))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _estimate(
    values: dict[tuple[int, int], float], *, iterations: int, label: str,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    low, high = _hierarchical_interval(values, iterations=iterations, label=label)
    point = mean(values.values()) if values else None
    result = {
        "persona_count": len({key[0] for key in values}),
        "context_pairs": len(values),
        "estimate": point,
        "ci95_low": low if values else None,
        "ci95_high": high if values else None,
        "ci_excludes_zero": bool(values and (low > 0 or high < 0)),
    }
    if progress_callback is not None:
        progress_callback(label)
    return result


def _limited_post_metrics(
    manifest_path: Path, manifest: dict[str, Any], *, limit: int
) -> dict[int, dict[str, Any]]:
    if limit < 1:
        raise ValueError("post repeat limit must be at least 1")
    output: dict[int, dict[str, Any]] = {}
    for context in manifest.get("contexts") or []:
        context_idx = context.get("context_idx")
        metrics_path = _resolve(context.get("privacy_metrics_cimemories_json"), manifest_path)
        if not isinstance(context_idx, int) or metrics_path is None or not metrics_path.is_file():
            raise FileNotFoundError(
                f"Missing post-rerank privacy metrics for context {context_idx} in {manifest_path}"
            )
        payload = _read_json(metrics_path)
        records = list(payload.get("repeated_exposed_records") or [])
        records = sorted(
            enumerate(records),
            key=lambda item: (
                item[1].get("repeat_idx")
                if isinstance(item[1].get("repeat_idx"), int)
                else item[1].get("record_index")
                if isinstance(item[1].get("record_index"), int)
                else item[0]
            ),
        )
        selected = [record for _, record in records[:limit]]
        if len(selected) < limit:
            raise ValueError(
                f"Context {context_idx} in {metrics_path} has only {len(selected)} "
                f"saved post-rerank repetitions; requested {limit}"
            )
        labels = {
            "necessary": set(payload.get("necessary_attributes") or []),
            "private": set(payload.get("inappropriate_attributes") or []),
            "ambiguous": set(payload.get("ambiguous_attributes") or []),
        }
        counts = {category: 0 for category in labels}
        exposed_counts: list[int] = []
        for record in selected:
            exposed = record.get("exposed_attributes")
            if not isinstance(exposed, dict):
                raise ValueError(f"Malformed repeated exposure record in {metrics_path}")
            exposed_keys = set(exposed)
            exposed_counts.append(len(exposed_keys))
            for category, attributes in labels.items():
                counts[category] += len(exposed_keys & attributes)

        def rate(category: str) -> float | None:
            denominator = len(labels[category]) * len(selected)
            return counts[category] / denominator if denominator else None

        output[context_idx] = {
            **payload,
            "necessary_recall": rate("necessary"),
            "inappropriate_leak_rate": rate("private"),
            "ambiguous_exposure_rate": rate("ambiguous"),
            "average_exposed_total": mean(exposed_counts),
        }
    return output


def _collect_rows(
    entries: list[dict[str, Any]], *, post_repeat_limit: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: (item["model"], item["architecture"], item["persona_idx"])):
        post, pre = _pipeline_metrics(entry), _pre_metrics(entry)
        shared = sorted(set(post) & set(pre))
        if len(shared) != 49:
            raise ValueError(
                f"Expected 49 paired contexts for {entry['model']} {entry['architecture']} "
                f"persona {entry['persona_idx']}; found {len(shared)}"
            )
        manifest_path = Path(entry["manifest_path"])
        manifest = _read_json(manifest_path)
        if post_repeat_limit is not None:
            post = _limited_post_metrics(
                manifest_path, manifest, limit=post_repeat_limit
            )
        memory_summary_paths = set(manifest_path.parent.glob("memory_stage_metrics*_summary.json"))
        # Dataset runs place the persona-level summary next to the nested
        # ``pipeline`` directory, whereas standalone runs place it next to the
        # pipeline manifest itself.
        if manifest_path.parent.name == "pipeline":
            memory_summary_paths.update(
                manifest_path.parent.parent.glob("memory_stage_metrics*_summary.json")
            )
        memory_summaries = sorted(str(path.resolve()) for path in memory_summary_paths)
        query_dirs = [
            value for value in (
                (manifest.get("artifacts") or {}).get("query_output_dir"),
                manifest.get("query_output_dir"),
            )
            if isinstance(value, str)
        ]
        inventory.append(
            {
                "model": entry["model"],
                "model_label": _short_model(entry["model"]),
                "architecture": entry["architecture"],
                "persona_idx": entry["persona_idx"],
                "persona_name": entry["persona_name"],
                "post_records": entry.get("post_records", 0),
                "pre_records": entry.get("pre_records", 0),
                "post_repeats": entry.get("post_repeats"),
                "pre_repeats": entry.get("pre_repeats"),
                "post_generation_seconds": (entry.get("post_generation_mean_ms") or 0) / 1000,
                "pre_generation_seconds": (entry.get("pre_generation_mean_ms") or 0) / 1000,
                "estimated_cost_usd": float(entry.get("post_cost_usd") or 0) + float(entry.get("pre_cost_usd") or 0),
                "cost_incomplete": bool(entry.get("post_cost_incomplete") or entry.get("pre_cost_incomplete")),
                "pipeline_manifest": str(manifest_path.resolve()),
                "pre_manifest": str(Path(entry["pre_manifest_path"]).resolve()),
                "memory_stage_summary_count": len(memory_summaries),
                "memory_stage_summaries": ";".join(memory_summaries),
                "retrieval_snapshot_recorded": bool(query_dirs or manifest.get("contexts")),
            }
        )
        for context_idx in shared:
            row = {
                "model": entry["model"],
                "model_label": _short_model(entry["model"]),
                "architecture": entry["architecture"],
                "persona_idx": entry["persona_idx"],
                "persona_name": post[context_idx].get("persona_name") or entry["persona_name"],
                "context_idx": context_idx,
                "recipient": post[context_idx].get("recipient"),
                "task": post[context_idx].get("task"),
                "context_discarded": bool(post[context_idx].get("context_discarded")),
            }
            for metric in METRICS:
                before, after = pre[context_idx].get(metric), post[context_idx].get(metric)
                row[f"pre_{metric}"] = before
                row[f"post_{metric}"] = after
                row[f"delta_{metric}"] = (
                    float(after) - float(before) if _finite(before) and _finite(after) else None
                )
            rows.append(row)
    return rows, inventory


def _condition_metrics(
    rows: list[dict[str, Any]], iterations: int,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    conditions = sorted({(r["model"], r["architecture"]) for r in rows})
    for model, architecture in conditions:
        subset = [r for r in rows if r["model"] == model and r["architecture"] == architecture]
        for stage in ("pre", "post"):
            for metric in METRICS:
                values = {
                    (r["persona_idx"], r["context_idx"]): float(r[f"{stage}_{metric}"])
                    for r in subset if _finite(r.get(f"{stage}_{metric}"))
                }
                estimate = _estimate(
                    values,
                    iterations=iterations,
                    label=f"absolute:{model}:{architecture}:{stage}:{metric}",
                    progress_callback=progress_callback,
                )
                output.append({"model": model, "model_label": _short_model(model), "architecture": architecture, "stage": stage, "metric": metric, **estimate})
    return output


def _reranking_effects(
    rows: list[dict[str, Any]], iterations: int,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    conditions = sorted({(r["model"], r["architecture"]) for r in rows})
    for model, architecture in conditions:
        subset = [r for r in rows if r["model"] == model and r["architecture"] == architecture]
        for metric in METRICS:
            values = {
                (r["persona_idx"], r["context_idx"]): float(r[f"delta_{metric}"])
                for r in subset if _finite(r.get(f"delta_{metric}"))
            }
            output.append({
                "model": model,
                "model_label": _short_model(model),
                "architecture": architecture,
                "metric": metric,
                **_estimate(
                    values,
                    iterations=iterations,
                    label=f"rerank:{model}:{architecture}:{metric}",
                    progress_callback=progress_callback,
                ),
            })
    return output


def _paired_contrasts(
    rows: list[dict[str, Any]], *, dimension: str, iterations: int,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    if dimension not in {"architecture", "model"}:
        raise ValueError("dimension must be architecture or model")
    fixed = "model" if dimension == "architecture" else "architecture"
    output: list[dict[str, Any]] = []
    for fixed_value in sorted({r[fixed] for r in rows}):
        candidates = sorted({r[dimension] for r in rows if r[fixed] == fixed_value})
        for left, right in combinations(candidates, 2):
            left_rows = {
                (r["persona_idx"], r["context_idx"]): r
                for r in rows if r[fixed] == fixed_value and r[dimension] == left
            }
            right_rows = {
                (r["persona_idx"], r["context_idx"]): r
                for r in rows if r[fixed] == fixed_value and r[dimension] == right
            }
            shared = set(left_rows) & set(right_rows)
            for stage in ("pre", "post"):
                for metric in METRICS:
                    values = {
                        key: float(right_rows[key][f"{stage}_{metric}"]) - float(left_rows[key][f"{stage}_{metric}"])
                        for key in shared
                        if _finite(left_rows[key].get(f"{stage}_{metric}")) and _finite(right_rows[key].get(f"{stage}_{metric}"))
                    }
                    output.append({
                        "comparison_dimension": dimension,
                        fixed: fixed_value,
                        f"left_{dimension}": left,
                        f"right_{dimension}": right,
                        "stage": stage,
                        "metric": metric,
                        "contrast": "right_minus_left",
                        **_estimate(
                            values,
                            iterations=iterations,
                            label=f"pair:{dimension}:{fixed_value}:{left}:{right}:{stage}:{metric}",
                            progress_callback=progress_callback,
                        ),
                    })
    return output


def _interaction_contrasts(
    rows: list[dict[str, Any]], *, dimension: str, iterations: int,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Difference-in-differences between reranking effects."""
    fixed = "model" if dimension == "architecture" else "architecture"
    output: list[dict[str, Any]] = []
    for fixed_value in sorted({r[fixed] for r in rows}):
        candidates = sorted({r[dimension] for r in rows if r[fixed] == fixed_value})
        for left, right in combinations(candidates, 2):
            left_rows = {(r["persona_idx"], r["context_idx"]): r for r in rows if r[fixed] == fixed_value and r[dimension] == left}
            right_rows = {(r["persona_idx"], r["context_idx"]): r for r in rows if r[fixed] == fixed_value and r[dimension] == right}
            shared = set(left_rows) & set(right_rows)
            for metric in METRICS:
                values = {
                    key: float(right_rows[key][f"delta_{metric}"]) - float(left_rows[key][f"delta_{metric}"])
                    for key in shared
                    if _finite(left_rows[key].get(f"delta_{metric}")) and _finite(right_rows[key].get(f"delta_{metric}"))
                }
                output.append({
                    "interaction_dimension": dimension,
                    fixed: fixed_value,
                    f"left_{dimension}": left,
                    f"right_{dimension}": right,
                    "metric": metric,
                    "contrast": "right_rerank_effect_minus_left_rerank_effect",
                    **_estimate(
                        values,
                        iterations=iterations,
                        label=f"interaction:{dimension}:{fixed_value}:{left}:{right}:{metric}",
                        progress_callback=progress_callback,
                    ),
                })
    return output


def _bootstrap_estimate_count(rows: list[dict[str, Any]], metric_count: int) -> int:
    """Count absolute, reranking, paired, and interaction estimates."""
    conditions = {(row["model"], row["architecture"]) for row in rows}
    total = len(conditions) * (2 * metric_count + metric_count)
    for dimension in ("architecture", "model"):
        fixed = "model" if dimension == "architecture" else "architecture"
        for fixed_value in {row[fixed] for row in rows}:
            candidate_count = len(
                {row[dimension] for row in rows if row[fixed] == fixed_value}
            )
            pairs = candidate_count * (candidate_count - 1) // 2
            total += pairs * (2 * metric_count + metric_count)
    return total


def _sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, architecture in sorted({(r["model"], r["architecture"]) for r in rows}):
        base = [r for r in rows if r["model"] == model and r["architecture"] == architecture]
        for scope, subset in (("all", base), ("exclude_discarded", [r for r in base if not r["context_discarded"]])):
            for metric in METRICS:
                values = [float(r[f"delta_{metric}"]) for r in subset if _finite(r.get(f"delta_{metric}"))]
                output.append({
                    "model": model, "model_label": _short_model(model), "architecture": architecture,
                    "scope": scope, "metric": metric,
                    "persona_count": len({r["persona_idx"] for r in subset}),
                    "context_pairs": len(values), "mean_delta": mean(values) if values else None,
                })
    return output


def _domain_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate reranking effects by the shared recipient/task scenario."""
    output: list[dict[str, Any]] = []
    groups = sorted(
        {
            (r["model"], r["architecture"], r["context_idx"], r["recipient"], r["task"])
            for r in rows
        },
        key=lambda item: (item[0], item[1], item[2]),
    )
    for model, architecture, context_idx, recipient, task in groups:
        subset = [
            r for r in rows
            if r["model"] == model and r["architecture"] == architecture
            and r["context_idx"] == context_idx
        ]
        for metric in METRICS:
            values = [
                float(r[f"delta_{metric}"])
                for r in subset if _finite(r.get(f"delta_{metric}"))
            ]
            output.append(
                {
                    "model": model,
                    "model_label": _short_model(model),
                    "architecture": architecture,
                    "context_idx": context_idx,
                    "recipient": recipient,
                    "task": task,
                    "metric": metric,
                    "persona_count": len({r["persona_idx"] for r in subset}),
                    "mean_delta": mean(values) if values else None,
                }
            )
    return output


def _coverage(inventory: list[dict[str, Any]], expected_personas: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, architecture in (
        (model, architecture)
        for model in sorted({r["model"] for r in inventory})
        for architecture in ARCHITECTURES
    ):
        subset = [r for r in inventory if r["model"] == model and r["architecture"] == architecture]
        present = sorted({int(r["persona_idx"]) for r in subset})
        output.append({
            "model": model, "model_label": _short_model(model), "architecture": architecture,
            "complete_personas": len(present), "expected_personas": expected_personas,
            "coverage_percent": 100 * len(present) / expected_personas,
            "persona_indices": ",".join(map(str, present)),
            "missing_persona_indices": ",".join(str(i) for i in range(expected_personas) if i not in present),
            "complete": len(present) == expected_personas,
        })
    return output


def _svg_header(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.sub{font-size:11px;fill:#667085}.lab{font-size:11px}.tiny{font-size:9px}.axis{stroke:#98a2b3}.grid{stroke:#e4e7ec}.zero{stroke:#344054;stroke-dasharray:4 4}</style>',
        f'<text class="title" x="24" y="32">{html.escape(title)}</text>',
        f'<text class="sub" x="24" y="51">{html.escape(subtitle)}</text>' if subtitle else "",
    ]


def _write_effect_forest(path: Path, effects: list[dict[str, Any]], models: list[str]) -> None:
    metrics = ("necessary_recall", "average_exposed_total")
    panels = {metric: [r for r in effects if r["metric"] == metric] for metric in metrics}
    rows = max(len(v) for v in panels.values())
    width, height = 1120, 95 + rows * 28 + 60
    panel_w, left = 485, 185
    svg = _svg_header(width, height, "Reranking effects with hierarchical 95% intervals", "Post minus pre; intervals account for personas and contexts")
    for panel, metric in enumerate(metrics):
        data = panels[metric]
        maximum = max(abs(float(r["ci95_low"] or 0)) for r in data) if data else 1
        maximum = max(maximum, max(abs(float(r["ci95_high"] or 0)) for r in data)) or 1
        x0 = left + panel * 535
        svg.append(f'<text class="lab" x="{x0 + panel_w/2}" y="76" text-anchor="middle" font-weight="700">{METRIC_LABELS[metric]}</text>')
        svg.append(f'<line class="zero" x1="{x0 + panel_w/2}" y1="88" x2="{x0 + panel_w/2}" y2="{height-38}"/>')
        for index, row in enumerate(data):
            y = 102 + index * 28
            scale = panel_w / (2 * maximum)
            center = x0 + panel_w / 2
            low = center + float(row["ci95_low"]) * scale
            high = center + float(row["ci95_high"]) * scale
            point = center + float(row["estimate"]) * scale
            color = ARCH_COLORS[row["architecture"]]
            svg.append(f'<line x1="{low}" y1="{y}" x2="{high}" y2="{y}" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<circle cx="{point}" cy="{y}" r="4.5" fill="{color}"/>')
            if panel == 0:
                label = f"{row['model_label']} · {row['architecture'].title()}"
                svg.append(f'<text class="lab" x="{x0-10}" y="{y+4}" text-anchor="end">{html.escape(label)}</text>')
        svg.append(f'<text class="tiny" x="{x0}" y="{height-16}">{-maximum:+.2f}</text>')
        svg.append(f'<text class="tiny" x="{x0+panel_w}" y="{height-16}" text-anchor="end">{maximum:+.2f}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_privacy_utility(path: Path, conditions: list[dict[str, Any]], models: list[str]) -> None:
    lookup = {(r["model"], r["architecture"], r["stage"], r["metric"]): r["estimate"] for r in conditions}
    points: list[tuple[str, str, float, float, float, float]] = []
    for model in models:
        for architecture in ARCHITECTURES:
            keys = [(model, architecture, stage, metric) for stage in ("pre", "post") for metric in ("average_exposed_total", "necessary_recall")]
            if all(key in lookup and lookup[key] is not None for key in keys):
                points.append((model, architecture, lookup[keys[0]], lookup[keys[1]], lookup[keys[2]], lookup[keys[3]]))
    width, height, left, top, plot_w, plot_h = 920, 610, 85, 85, 735, 420
    max_x = max(max(p[2], p[4]) for p in points) * 1.12 if points else 1
    max_y = max(max(p[3], p[5]) for p in points) * 1.15 if points else 1
    svg = _svg_header(width, height, "Privacy–utility trajectories", "Each arrow shows how reranking moves one model × architecture condition")
    for tick in range(6):
        x = left + plot_w * tick / 5; y = top + plot_h * tick / 5
        svg.append(f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top+plot_h}"/>')
        svg.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}"/>')
        svg.append(f'<text class="tiny" x="{x}" y="{top+plot_h+19}" text-anchor="middle">{max_x*tick/5:.1f}</text>')
        svg.append(f'<text class="tiny" x="{left-10}" y="{top+plot_h-plot_h*tick/5+4}" text-anchor="end">{100*max_y*tick/5:.0f}%</text>')
    for model_index, (model, architecture, pre_x, pre_y, post_x, post_y) in enumerate(points):
        color = MODEL_COLORS[models.index(model) % len(MODEL_COLORS)]
        x1 = left + plot_w * pre_x / max_x; y1 = top + plot_h * (1-pre_y/max_y)
        x2 = left + plot_w * post_x / max_x; y2 = top + plot_h * (1-post_y/max_y)
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.5" marker-end="url(#arrow{models.index(model)})"/>')
        svg.append(f'<circle cx="{x1}" cy="{y1}" r="4" fill="white" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<circle cx="{x2}" cy="{y2}" r="5" fill="{color}"/>')
        svg.append(f'<text class="tiny" x="{x2+6}" y="{y2-5}">{html.escape(architecture.title())}</text>')
    defs = ["<defs>"]
    for i, color in enumerate(MODEL_COLORS[:len(models)]):
        defs.append(f'<marker id="arrow{i}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>')
    defs.append("</defs>"); svg[1:1] = defs
    svg.append(f'<text class="lab" x="{left+plot_w/2}" y="{top+plot_h+48}" text-anchor="middle">Mean exposed attributes (lower is more selective)</text>')
    svg.append(f'<text class="lab" transform="translate(22 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Necessary recall (higher is better)</text>')
    legend_y = height - 40
    for i, model in enumerate(models):
        x = 90 + i * 250
        svg.append(f'<circle cx="{x}" cy="{legend_y}" r="5" fill="{MODEL_COLORS[i]}"/><text class="lab" x="{x+11}" y="{legend_y+4}">{html.escape(_short_model(model))}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_interaction_heatmap(path: Path, effects: list[dict[str, Any]], models: list[str]) -> None:
    metrics = list(METRICS)
    rows = [(model, arch) for model in models for arch in ARCHITECTURES if any(r["model"] == model and r["architecture"] == arch for r in effects)]
    left, top, cell_w, cell_h = 235, 92, 170, 34
    width, height = left + len(metrics)*cell_w + 28, top + len(rows)*cell_h + 65
    scales = {m: max((abs(float(r["estimate"])) for r in effects if r["metric"] == m), default=1) or 1 for m in metrics}
    svg = _svg_header(width, height, "Reranking interaction map", "Green denotes a desirable movement: recall up or disclosure down")
    for col, metric in enumerate(metrics):
        svg.append(f'<text class="tiny" x="{left+col*cell_w+cell_w/2}" y="{top-18}" text-anchor="middle">{html.escape(METRIC_LABELS[metric])}</text>')
    for row_index, (model, arch) in enumerate(rows):
        y = top + row_index*cell_h
        svg.append(f'<text class="lab" x="{left-10}" y="{y+22}" text-anchor="end">{html.escape(_short_model(model))} · {arch.title()}</text>')
        for col, metric in enumerate(metrics):
            record = next(r for r in effects if r["model"] == model and r["architecture"] == arch and r["metric"] == metric)
            raw = float(record["estimate"]); desirable = raw if metric == "necessary_recall" else -raw
            ratio = min(1, abs(desirable)/scales[metric]); target = (39, 145, 95) if desirable >= 0 else (204, 69, 79)
            rgb = tuple(round(247+(channel-247)*ratio) for channel in target)
            x = left+col*cell_w
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" rx="4" fill="rgb{rgb}"/>')
            suffix = "" if metric == "average_exposed_total" else " pp"
            value = raw if metric == "average_exposed_total" else raw*100
            star = "*" if record["ci_excludes_zero"] else ""
            svg.append(f'<text class="lab" x="{x+(cell_w-3)/2}" y="{y+21}" text-anchor="middle">{value:+.2f}{suffix}{star}</text>')
    svg.append(f'<text class="tiny" x="24" y="{height-20}">* hierarchical 95% interval excludes zero. Color direction is metric-aware; printed signs are raw post-minus-pre effects.</text>')
    svg.append("</svg>"); path.write_text("\n".join(svg), encoding="utf-8")


def _write_persona_heterogeneity(
    path: Path, persona_effects: list[dict[str, Any]], models: list[str]
) -> None:
    """Show whether aggregate reranking effects are consistent across personas."""
    metrics = ("necessary_recall", "average_exposed_total")
    columns = [
        (model, architecture)
        for model in models for architecture in ARCHITECTURES
        if any(
            row["model"] == model and row["architecture"] == architecture
            for row in persona_effects
        )
    ]
    personas = sorted({int(row["persona_idx"]) for row in persona_effects})
    left, top, cell_w, cell_h, panel_gap = 90, 78, 108, 32, 42
    panel_w = len(columns) * cell_w
    panel_height = 76 + len(personas) * cell_h
    width = left + panel_w + 30
    height = top + 2 * panel_height + panel_gap + 54
    svg = _svg_header(
        width,
        height,
        "Persona-level reranking heterogeneity",
        "Raw post-minus-pre effects; neutral gray marks unavailable persona-condition cells",
    )
    svg.append('<style>.persona-label{font-size:15px}.persona-value{font-size:14px}.persona-head{font-size:13px;font-weight:600}</style>')
    lookup = {
        (row["model"], row["architecture"], row["persona_idx"], row["metric"]): row["mean_delta"]
        for row in persona_effects
    }
    scales = {
        metric: max(
            (abs(float(row["mean_delta"])) for row in persona_effects if row["metric"] == metric and _finite(row["mean_delta"])),
            default=1.0,
        ) or 1.0
        for metric in metrics
    }
    for panel, metric in enumerate(metrics):
        panel_top = top + panel * (panel_height + panel_gap)
        grid_top = panel_top + 76
        svg.append(
            f'<text class="persona-label" x="{left + panel_w/2}" y="{panel_top}" text-anchor="middle" '
            f'font-weight="700">{html.escape(METRIC_LABELS[metric])}</text>'
        )
        for column, (model, architecture) in enumerate(columns):
            x = left + column * cell_w + cell_w / 2
            model_label = _short_model(model).replace("-Flash", "")
            svg.append(f'<text class="persona-head" x="{x}" y="{panel_top+29}" text-anchor="middle">{html.escape(model_label)}</text>')
            svg.append(f'<text class="persona-head" x="{x}" y="{panel_top+49}" text-anchor="middle">{architecture.title()}</text>')
        for row_index, persona in enumerate(personas):
            y = grid_top + row_index * cell_h
            svg.append(
                f'<text class="persona-label" x="{left-10}" y="{y+22}" '
                f'text-anchor="end">P{persona}</text>'
            )
            for column, (model, architecture) in enumerate(columns):
                value = lookup.get((model, architecture, persona, metric))
                x = left + column * cell_w
                if not _finite(value):
                    fill, rendered = "#eaecf0", "—"
                else:
                    raw = float(value)
                    desirable = raw if metric == "necessary_recall" else -raw
                    ratio = min(1.0, abs(desirable) / scales[metric])
                    target = (39, 145, 95) if desirable >= 0 else (204, 69, 79)
                    rgb = tuple(round(247 + (channel - 247) * ratio) for channel in target)
                    fill = f"rgb{rgb}"
                    rendered = f"{raw*100:+.1f}" if metric != "average_exposed_total" else f"{raw:+.1f}"
                svg.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" '
                    f'rx="3" fill="{fill}"/>'
                )
                svg.append(
                    f'<text class="persona-value" x="{x+(cell_w-3)/2}" y="{y+21}" '
                    f'text-anchor="middle">{rendered}</text>'
                )
    svg.append(
        f'<text class="persona-label" x="24" y="{height-18}">Green is desirable (recall up or exposure down); red is undesirable. Recall values are percentage points; exposure values are attributes per response.</text>'
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_domain_heterogeneity(
    path: Path,
    domain_effects: list[dict[str, Any]],
    models: list[str],
    metric: str,
) -> None:
    if metric not in {"necessary_recall", "average_exposed_total"}:
        raise ValueError("Domain heterogeneity supports necessary recall or exposed attributes")
    columns = [
        (model, architecture)
        for model in models for architecture in ARCHITECTURES
        if any(r["model"] == model and r["architecture"] == architecture for r in domain_effects)
    ]
    contexts = sorted({int(r["context_idx"]) for r in domain_effects})
    labels: dict[int, str] = {}
    for context in contexts:
        row = next(r for r in domain_effects if r["context_idx"] == context)
        label = f"{context}: {row['recipient']} — {row['task']}"
        labels[context] = label if len(label) <= 46 else label[:45] + "…"
    left, top, cell_w, cell_h = 310, 138, 92, 24
    panel_w = len(columns) * cell_w
    width = left + panel_w + 30
    height = top + len(contexts) * cell_h + 62
    svg = _svg_header(
        width,
        height,
        f"Scenario-domain reranking effects: {METRIC_LABELS[metric]}",
        "Rows are recipient/task domains; cells average paired effects across available personas",
    )
    svg.append('<style>.domain-label{font-size:14px}.domain-value{font-size:13px}.domain-head{font-size:13px;font-weight:600}</style>')
    lookup = {
        (r["model"], r["architecture"], r["context_idx"], r["metric"]): r["mean_delta"]
        for r in domain_effects
    }
    scale = max(
        (abs(float(r["mean_delta"])) for r in domain_effects if r["metric"] == metric and _finite(r["mean_delta"])),
        default=1.0,
    ) or 1.0
    for column, (model, architecture) in enumerate(columns):
        x = left + column*cell_w + cell_w/2
        model_label = _short_model(model).replace("-Flash", "")
        svg.append(f'<text class="domain-head" x="{x}" y="{top-46}" text-anchor="middle">{html.escape(model_label)}</text>')
        svg.append(f'<text class="domain-head" x="{x}" y="{top-27}" text-anchor="middle">{architecture.title()}</text>')
    for row_index, context in enumerate(contexts):
        y = top + row_index*cell_h
        svg.append(f'<text class="domain-label" x="{left-10}" y="{y+17}" text-anchor="end">{html.escape(labels[context])}</text>')
        for column, (model, architecture) in enumerate(columns):
            raw_value = lookup.get((model, architecture, context, metric))
            x = left + column*cell_w
            if not _finite(raw_value):
                fill, rendered = "#eaecf0", "—"
            else:
                raw = float(raw_value); desirable = raw if metric == "necessary_recall" else -raw
                ratio = min(1.0, abs(desirable)/scale); target = (39,145,95) if desirable >= 0 else (204,69,79)
                fill = f"rgb{tuple(round(247+(channel-247)*ratio) for channel in target)}"
                rendered = f"{raw*100:+.0f}" if metric != "average_exposed_total" else f"{raw:+.1f}"
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-2}" rx="2" fill="{fill}"/>')
            svg.append(f'<text class="domain-value" x="{x+(cell_w-3)/2}" y="{y+16}" text-anchor="middle">{rendered}</text>')
    unit = "Recall values are percentage points." if metric == "necessary_recall" else "Values are attributes per response."
    svg.append(f'<text class="domain-label" x="24" y="{height-18}">Green is desirable (recall up or exposure down); red is undesirable. {unit}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_study_flow(path: Path) -> None:
    width, height = 1180, 410
    svg = _svg_header(width, height, "From stored memory to disclosed attributes", "Analysis layers in the current and planned study")
    boxes = [
        (35, 105, 185, 88, "Initialized memory", "List facts · graph topology · profile"),
        (255, 105, 185, 88, "Retrieved candidates", "Architecture-native search result"),
        (475, 105, 185, 88, "Contextual reranking", "Pre → post candidate selection"),
        (695, 105, 185, 88, "Message generation", "Model × architecture × stage"),
        (915, 105, 225, 88, "Observed disclosure", "Necessary · inappropriate · ambiguous"),
    ]
    for index, (x,y,w,h,title,sub) in enumerate(boxes):
        fill = "#eef4ff" if index < 3 else "#f2f8f5"
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#98a2b3"/>')
        svg.append(f'<text class="lab" x="{x+w/2}" y="{y+34}" text-anchor="middle" font-weight="700">{html.escape(title)}</text>')
        svg.append(f'<text class="tiny" x="{x+w/2}" y="{y+57}" text-anchor="middle">{html.escape(sub)}</text>')
        if index < len(boxes)-1:
            svg.append(f'<line x1="{x+w}" y1="{y+h/2}" x2="{boxes[index+1][0]-10}" y2="{y+h/2}" stroke="#667085" stroke-width="2"/>')
            svg.append(f'<path d="M {boxes[index+1][0]-10} {y+h/2-5} L {boxes[index+1][0]} {y+h/2} L {boxes[index+1][0]-10} {y+h/2+5} z" fill="#667085"/>')
    svg.append('<rect x="35" y="245" width="625" height="88" rx="12" fill="#fff7e8" stroke="#d99b28" stroke-dasharray="5 4"/>')
    svg.append('<text class="lab" x="347" y="275" text-anchor="middle" font-weight="700">Planned attribute-flow analysis</text>')
    svg.append('<text class="tiny" x="347" y="299" text-anchor="middle">Which ground-truth attributes exist in initialized memory, survive retrieval and reranking, and become exposed?</text>')
    svg.append('<text class="tiny" x="347" y="316" text-anchor="middle">Join keys: model · architecture · persona · context · attribute · stage</text>')
    svg.append('<rect x="695" y="245" width="445" height="88" rx="12" fill="#f5f3ff" stroke="#8065b5"/>')
    svg.append('<text class="lab" x="918" y="275" text-anchor="middle" font-weight="700">Current factorial analysis</text>')
    svg.append('<text class="tiny" x="918" y="299" text-anchor="middle">Operating points · reranking effects · architecture/model contrasts</text>')
    svg.append('<text class="tiny" x="918" y="316" text-anchor="middle">Difference-in-differences · heterogeneity · robustness</text>')
    svg.append("</svg>"); path.write_text("\n".join(svg), encoding="utf-8")


def _fmt(value: Any, metric: str) -> str:
    if not _finite(value):
        return "—"
    return f"{100*float(value):+.2f} pp" if metric != "average_exposed_total" else f"{float(value):+.2f}"


def _latex_tables(output: Path, conditions: list[dict[str, Any]], effects: list[dict[str, Any]]) -> list[str]:
    includes = output / "includes"; includes.mkdir()
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{lllrrrr}", r"\toprule", "Model & Memory & Stage & Necessary & Inappropriate & Ambiguous & Exposed \\\\", r"\midrule"]
    for model, architecture, stage in sorted({(r["model"],r["architecture"],r["stage"]) for r in conditions}):
        lookup = {r["metric"]: r for r in conditions if (r["model"],r["architecture"],r["stage"]) == (model,architecture,stage)}
        cells = []
        for metric in METRICS:
            value = lookup[metric]["estimate"]
            cells.append(f"{value:.2f}" if metric == "average_exposed_total" else f"{100*value:.1f}\%")
        lines.append(f"{_short_model(model)} & {architecture.title()} & {stage.title()} & " + " & ".join(cells) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Absolute privacy--utility operating points by response model, memory architecture, and retrieval stage.}", r"\label{tab:cimemories-operating-points}", r"\end{table*}", ""]
    (includes / "operating_points.tex").write_text("\n".join(lines), encoding="utf-8")
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrrr}", r"\toprule", "Model & Memory & $\\Delta$ Necessary & $\\Delta$ Inappropriate & $\\Delta$ Ambiguous & $\\Delta$ Exposed \\\\", r"\midrule"]
    for model, architecture in sorted({(r["model"],r["architecture"]) for r in effects}):
        lookup = {r["metric"]: r for r in effects if (r["model"],r["architecture"]) == (model,architecture)}
        cells=[]
        for metric in METRICS:
            record=lookup[metric]; rendered=_fmt(record["estimate"],metric)
            cells.append(rf"\textbf{{{rendered}}}" if record["ci_excludes_zero"] else rendered)
        lines.append(f"{_short_model(model)} & {architecture.title()} & " + " & ".join(cells) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Paired post-minus-pre reranking effects. Bold values have hierarchical 95\% intervals excluding zero.}", r"\label{tab:cimemories-reranking-effects}", r"\end{table*}", ""]
    (includes / "reranking_effects.tex").write_text("\n".join(lines), encoding="utf-8")
    (includes / "paper_preamble.tex").write_text("\\usepackage{booktabs}\n\\usepackage{svg}\n\\svgsetup{inkscapelatex=false}\n", encoding="utf-8")
    figure_names = ("study_flow", "privacy_utility_trajectories", "reranking_effect_forest", "reranking_interaction_heatmap", "persona_heterogeneity", "scenario_domain_necessary_recall", "scenario_domain_exposed_attributes")
    for name in figure_names:
        caption = name.replace("_", " ").title()
        (includes / f"{name}.tex").write_text(
            "\\begin{figure*}[t]\n\\centering\n"
            f"\\includesvg[width=0.98\\textwidth]{{figures/{name}.svg}}\n"
            f"\\caption{{{caption}.}}\n\\label{{fig:{name.replace('_','-')}}}\n\\end{{figure*}}\n",
            encoding="utf-8",
        )
    return ["includes/paper_preamble.tex", "includes/operating_points.tex", "includes/reranking_effects.tex", *[f"includes/{name}.tex" for name in figure_names]]


def _html_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]], limit: int | None = None) -> str:
    body=[]
    for row in rows[:limit]:
        body.append("<tr>"+"".join(f"<td>{html.escape(str(row.get(key,'')))}</td>" for key,_ in fields)+"</tr>")
    return "<div class='table-wrap'><table><thead><tr>"+"".join(f"<th>{html.escape(label)}</th>" for _,label in fields)+"</tr></thead><tbody>"+"".join(body)+"</tbody></table></div>"


def _report_html(
    *, coverage: list[dict[str, Any]], conditions: list[dict[str, Any]], effects: list[dict[str, Any]],
    architecture_contrasts: list[dict[str, Any]], model_contrasts: list[dict[str, Any]],
    interactions: list[dict[str, Any]], inventory: list[dict[str, Any]], expected_personas: int,
    iterations: int, partial: bool,
    analysis_provenance: dict[str, Any] | None = None,
    post_repeat_limit: int | None = None,
) -> str:
    coverage_cards="".join(
        f"<div class='coverage-card {'partial' if not row['complete'] else ''}'><b>{html.escape(row['model_label'])}</b><span>{row['architecture'].title()}</span><strong>{row['complete_personas']}/{row['expected_personas']}</strong></div>"
        for row in coverage
    )
    effect_rows=[]
    for model, architecture in sorted({(r["model"],r["architecture"]) for r in effects}):
        lookup={r["metric"]:r for r in effects if r["model"]==model and r["architecture"]==architecture}
        effect_rows.append({"model":_short_model(model),"architecture":architecture.title(),"personas":max(r["persona_count"] for r in lookup.values()), **{metric:_fmt(lookup[metric]["estimate"],metric)+(" *" if lookup[metric]["ci_excludes_zero"] else "") for metric in METRICS}})
    notice = "This is an interim export: incomplete cells are shown explicitly and pairwise contrasts use only their shared persona/context cells." if partial else "All expected model × architecture cells have complete matched pre/post coverage."
    provenance_notice = ""
    repeat_notice = ""
    if post_repeat_limit is not None:
        repeat_notice = (
            "<div class='notice'><strong>Matched-repeat analysis:</strong> "
            f"post-rerank metrics use only the first {post_repeat_limit} saved repetition(s), "
            "selected by the lowest repeat index. Pre-rerank metrics are unchanged.</div>"
        )
    if analysis_provenance:
        label_name = html.escape(str(analysis_provenance.get("ground_truth_name") or "supplied ground truth"))
        label_path = html.escape(str(analysis_provenance.get("ground_truth_file") or ""))
        label_hash = html.escape(str(analysis_provenance.get("ground_truth_sha256") or ""))
        provenance_notice = (
            "<div class='notice'><strong>Alternate ground truth:</strong> "
            f"response metrics were rescored offline with <code>{label_name}</code> "
            f"from <code>{label_path}</code> (SHA-256 <code>{label_hash}</code>). "
            "Saved generations and exposure judgments were reused unchanged.</div>"
        )
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CIMemories factorial study artifacts</title><style>
    :root{{--ink:#172033;--muted:#667085;--line:#d0d5dd;--paper:#f5f7fb;--blue:#3559a8;--warn:#fff4e5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,system-ui,sans-serif}}header{{padding:54px max(5vw,30px);background:linear-gradient(120deg,#172033,#2d4275);color:white}}header h1{{margin:0 0 10px;font-size:34px}}header p{{max-width:920px;line-height:1.55;color:#dce4ff}}nav{{position:sticky;top:0;background:white;border-bottom:1px solid var(--line);padding:12px 5vw;z-index:2}}nav a{{margin-right:20px;color:#344b85;text-decoration:none;font-weight:650}}main{{max-width:1380px;margin:auto;padding:30px}}section{{background:white;border:1px solid #e3e7ee;border-radius:14px;padding:26px;margin:0 0 24px;box-shadow:0 8px 28px #1720330b}}h2{{margin-top:0}}.notice{{padding:14px 16px;border-left:4px solid #d99b28;background:var(--warn);margin:16px 0;line-height:1.5}}.coverage{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.coverage-card{{display:grid;grid-template-columns:1fr auto;gap:4px;padding:14px;border-radius:10px;background:#eef8f3;border:1px solid #c8e8d8}}.coverage-card.partial{{background:#fff5e8;border-color:#f2d39b}}.coverage-card span{{color:var(--muted)}}.coverage-card strong{{grid-row:1/3;grid-column:2;align-self:center;font-size:22px}}.fig-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0;border:1px solid #e4e7ec;border-radius:10px;padding:10px}}figure img{{width:100%;height:auto}}figcaption{{color:var(--muted);font-size:13px;padding:8px}}.table-wrap{{overflow:auto;max-height:620px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{position:sticky;top:0;background:#eef2f8;text-align:left}}th,td{{padding:9px 11px;border-bottom:1px solid #e4e7ec;white-space:nowrap}}code{{background:#eef2f8;padding:2px 5px;border-radius:4px}}@media(max-width:900px){{.fig-grid{{grid-template-columns:1fr}}}}
    </style></head><body><header><h1>CIMemories factorial study</h1><p>Models × memory architectures × pre/post reranking, with paired contrasts, interaction tests, heterogeneity, provenance, and a forward-compatible bridge to attribute-level memory-flow analysis.</p></header><nav><a href='#coverage'>Coverage</a><a href='#headline'>Headline</a><a href='#contrasts'>Contrasts</a><a href='#memory'>Memory flow</a><a href='#provenance'>Provenance</a></nav><main>
    <section id='coverage'><h2>Coverage and analysis status</h2><div class='notice'>{html.escape(notice)}</div>{repeat_notice}{provenance_notice}<div class='coverage'>{coverage_cards}</div></section>
    <section id='headline'><h2>Headline figures</h2><div class='fig-grid'><figure><img src='figures/privacy_utility_trajectories.svg'><figcaption>Absolute operating points and pre-to-post movement.</figcaption></figure><figure><img src='figures/reranking_effect_forest.svg'><figcaption>Magnitude and uncertainty of necessary-recall and total-disclosure effects.</figcaption></figure><figure><img src='figures/reranking_interaction_heatmap.svg'><figcaption>Metric-aware direction and significance across all conditions.</figcaption></figure><figure><img src='figures/persona_heterogeneity.svg'><figcaption>Whether aggregate reranking effects are consistent across personas.</figcaption></figure><figure><img src='figures/scenario_domain_necessary_recall.svg'><figcaption>Necessary-recall movement across recipient/task domains.</figcaption></figure><figure><img src='figures/scenario_domain_exposed_attributes.svg'><figcaption>Total-disclosure movement across recipient/task domains.</figcaption></figure><figure><img src='figures/study_flow.svg'><figcaption>Current factorial analysis and planned attribute-flow layer.</figcaption></figure></div><h3>Reranking effects</h3>{_html_table(effect_rows,[("model","Model"),("architecture","Memory"),("personas","N"),("necessary_recall","Necessary"),("inappropriate_leak_rate","Inappropriate"),("ambiguous_exposure_rate","Ambiguous"),("average_exposed_total","Exposed")])}<p><small>* hierarchical 95% interval excludes zero.</small></p></section>
    <section id='contrasts'><h2>Paired comparisons and interactions</h2><p>Architecture and model comparisons use the largest pairwise matched cohort available for that exact contrast. Difference-in-differences tests whether reranking effects change across architectures or models.</p><ul><li><code>architecture_contrasts.csv</code>: architecture differences within model and stage.</li><li><code>model_contrasts.csv</code>: model differences within architecture and stage.</li><li><code>interaction_contrasts.csv</code>: reranking-effect differences across models and architectures.</li><li><code>discarded_context_sensitivity.csv</code>: all-context versus retained-context directions.</li><li><code>persona_reranking_effects.csv</code>: heterogeneity behind aggregate means.</li></ul>{_html_table(interactions,[("interaction_dimension","Interaction"),("model","Fixed model"),("architecture","Fixed memory"),("left_model","Left model"),("right_model","Right model"),("left_architecture","Left memory"),("right_architecture","Right memory"),("metric","Metric"),("persona_count","N"),("estimate","Difference-in-differences"),("ci95_low","Low"),("ci95_high","High")],limit=80)}</section>
    <section id='memory'><h2>Attribute-level memory-flow extension</h2><p>The exporter already inventories retrieval snapshots and memory-stage summaries per selected persona pipeline in <code>memory_stage_inventory.csv</code>. Future snapshot analysis should use the stable join key <code>model × architecture × persona × context × attribute × stage</code> and report transitions from initialized memory to retrieved candidates, post-rerank memory, and final exposure. This avoids conflating retrieval availability with model disclosure.</p><img src='figures/study_flow.svg' style='width:100%'></section>
    <section id='provenance'><h2>Reproducibility and provenance</h2><p>The exact source manifests are locked in <code>study_artifact_inputs.json</code>. Re-run this exporter with <code>--inputs</code> to reproduce the same cohort even after newer experiments appear. Bootstrap iterations: {iterations:,}. Expected personas per condition: {expected_personas}.</p>{_html_table(inventory,[("model_label","Model"),("architecture","Memory"),("persona_idx","Persona"),("post_records","Post"),("pre_records","Pre"),("post_repeats","Post reps"),("pre_repeats","Pre reps"),("memory_stage_summary_count","Memory-stage summaries"),("pipeline_manifest","Source manifest")])}</section>
    </main></body></html>"""


def export_cimemories_study_artifacts(
    *, output_root: Path = Path("research_outputs"), dataset_filter: str | None = "cimemories_raw",
    output: Path | None = None, inputs: Path | None = None, bootstrap_iterations: int = 5000,
    expected_personas: int = 10, require_complete: bool = False,
    model_filters: list[str] | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    row_transform: Callable[[list[dict[str, Any]], list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    analysis_provenance: dict[str, Any] | None = None,
    post_repeat_limit: int | None = None,
) -> Path:
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100")
    if expected_personas < 1:
        raise ValueError("expected_personas must be at least 1")
    if inputs is not None and model_filters:
        raise ValueError("--models cannot be combined with --inputs; the lockfile already fixes models")
    if progress_callback is not None:
        progress_callback("response-inputs", 0, 1, "selecting source pipelines")
    if inputs is not None:
        entries = _locked_entries(inputs)
        locked_payload = _read_json(inputs)
        locked_limit = (locked_payload.get("response_repeat_policy") or {}).get(
            "post_repeat_limit"
        )
        if locked_limit is not None:
            post_repeat_limit = int(locked_limit)
    else:
        from .cli import _cimemories_progress_entries
        entries = _select_entries(_cimemories_progress_entries(output_root=output_root, dataset_filter=dataset_filter, model_filter=None))
    entries = [e for e in entries if e.get("post_complete") and e.get("pre_complete")]
    if model_filters:
        available = sorted({str(e["model"]) for e in entries})
        selected_models: set[str] = set()
        for requested in model_filters:
            matches = [model for model in available if requested.lower() in model.lower()]
            if len(matches) != 1:
                raise ValueError(
                    f"Model filter {requested!r} matched {len(matches)} models: {matches or available}"
                )
            selected_models.add(matches[0])
        entries = [e for e in entries if e["model"] in selected_models]
    if not entries:
        raise ValueError("No complete matched pre/post persona cells were found")
    rows, inventory = _collect_rows(entries, post_repeat_limit=post_repeat_limit)
    if row_transform is not None:
        rows = row_transform(entries, rows)
    coverage = _coverage(inventory, expected_personas)
    partial = any(not row["complete"] for row in coverage)
    if require_complete and partial:
        missing = "; ".join(f"{r['model_label']}/{r['architecture']}: {r['missing_persona_indices'] or 'coverage mismatch'}" for r in coverage if not r["complete"])
        raise ValueError(f"Study is incomplete for expected_personas={expected_personas}: {missing}")
    if progress_callback is not None:
        progress_callback(
            "response-inputs", 1, 1,
            f"loaded {len(inventory)} persona cells and {len(rows)} context rows",
        )

    bootstrap_total = _bootstrap_estimate_count(rows, len(METRICS))
    bootstrap_done = 0

    def bootstrap_progress(label: str) -> None:
        nonlocal bootstrap_done
        bootstrap_done += 1
        if progress_callback is not None:
            progress_callback("response-bootstrap", bootstrap_done, bootstrap_total, label)

    if progress_callback is not None:
        progress_callback(
            "response-bootstrap", 0, bootstrap_total,
            f"starting {bootstrap_iterations:,}-resample intervals",
        )
    conditions = _condition_metrics(
        rows, bootstrap_iterations, progress_callback=bootstrap_progress
    )
    effects = _reranking_effects(
        rows, bootstrap_iterations, progress_callback=bootstrap_progress
    )
    architecture_contrasts = _paired_contrasts(
        rows, dimension="architecture", iterations=bootstrap_iterations,
        progress_callback=bootstrap_progress,
    )
    model_contrasts = _paired_contrasts(
        rows, dimension="model", iterations=bootstrap_iterations,
        progress_callback=bootstrap_progress,
    )
    interactions = _interaction_contrasts(
        rows, dimension="architecture", iterations=bootstrap_iterations,
        progress_callback=bootstrap_progress,
    ) + _interaction_contrasts(
        rows, dimension="model", iterations=bootstrap_iterations,
        progress_callback=bootstrap_progress,
    )
    sensitivity = _sensitivity(rows)
    domain_effects = _domain_effects(rows)
    persona_effects: list[dict[str, Any]] = []
    for model, architecture, persona in sorted({(r["model"],r["architecture"],r["persona_idx"]) for r in rows}):
        subset=[r for r in rows if r["model"]==model and r["architecture"]==architecture and r["persona_idx"]==persona]
        for metric in METRICS:
            values=[float(r[f"delta_{metric}"]) for r in subset if _finite(r.get(f"delta_{metric}"))]
            persona_effects.append({"model":model,"model_label":_short_model(model),"architecture":architecture,"persona_idx":persona,"persona_name":subset[0]["persona_name"],"metric":metric,"context_pairs":len(values),"mean_delta":mean(values) if values else None})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output or output_root / f"cimemories-study-artifacts_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"; figures.mkdir()
    models = sorted({r["model"] for r in rows})
    figure_done = 0
    figure_total = 7
    if progress_callback is not None:
        progress_callback("response-figures", 0, figure_total, "rendering SVG figures")
    _write_effect_forest(figures / "reranking_effect_forest.svg", effects, models)
    figure_done += 1
    if progress_callback is not None:
        progress_callback("response-figures", figure_done, figure_total, "reranking effect forest")
    _write_privacy_utility(figures / "privacy_utility_trajectories.svg", conditions, models)
    figure_done += 1
    if progress_callback is not None:
        progress_callback("response-figures", figure_done, figure_total, "privacy-utility trajectories")
    _write_interaction_heatmap(figures / "reranking_interaction_heatmap.svg", effects, models)
    figure_done += 1
    if progress_callback is not None:
        progress_callback("response-figures", figure_done, figure_total, "interaction heatmap")
    _write_persona_heterogeneity(figures / "persona_heterogeneity.svg", persona_effects, models)
    figure_done += 1
    if progress_callback is not None:
        progress_callback("response-figures", figure_done, figure_total, "persona heterogeneity")
    _write_domain_heterogeneity(
        figures / "scenario_domain_necessary_recall.svg",
        domain_effects,
        models,
        "necessary_recall",
    )
    figure_done += 1
    if progress_callback is not None:
        progress_callback("response-figures", figure_done, figure_total, "domain necessary recall")
    _write_domain_heterogeneity(
        figures / "scenario_domain_exposed_attributes.svg",
        domain_effects,
        models,
        "average_exposed_total",
    )
    figure_done += 1
    if progress_callback is not None:
        progress_callback("response-figures", figure_done, figure_total, "domain exposed attributes")
    _write_study_flow(figures / "study_flow.svg")
    figure_done += 1
    if progress_callback is not None:
        progress_callback("response-figures", figure_done, figure_total, "study flow")
        progress_callback("response-artifacts", 0, 1, "writing tables and provenance")
    latex_artifacts = _latex_tables(output, conditions, effects)

    _write_csv(output / "coverage.csv", coverage)
    _write_csv(output / "condition_metrics.csv", conditions)
    _write_csv(output / "reranking_effects.csv", effects)
    _write_csv(output / "architecture_contrasts.csv", architecture_contrasts)
    _write_csv(output / "model_contrasts.csv", model_contrasts)
    _write_csv(output / "interaction_contrasts.csv", interactions)
    _write_csv(output / "discarded_context_sensitivity.csv", sensitivity)
    _write_csv(output / "persona_reranking_effects.csv", persona_effects)
    _write_csv(output / "scenario_domain_reranking_effects.csv", domain_effects)
    _write_csv(output / "context_metrics.csv", rows)
    _write_csv(output / "experiment_inventory.csv", inventory)
    _write_csv(output / "memory_stage_inventory.csv", inventory)
    attribute_flow_schema = {
        "schema_version": 1,
        "status": "planned-extension-contract",
        "unit": "one ground-truth attribute in one model/architecture/persona/context",
        "join_key": ["model", "architecture", "persona_idx", "context_idx", "attribute_id"],
        "required_fields": [
            "model", "architecture", "persona_idx", "context_idx", "attribute_id",
            "attribute_text", "privacy_class", "present_initialized_memory",
            "present_pre_rerank", "present_post_rerank", "exposed_pre_response",
            "exposed_post_response", "matching_method", "source_artifact",
        ],
        "privacy_classes": ["necessary", "inappropriate", "ambiguous"],
        "transition_analyses": [
            "initialized_to_pre_rerank_retrieval_recall",
            "pre_to_post_rerank_retention",
            "post_rerank_to_response_exposure",
            "end_to_end_initialized_to_response_exposure",
        ],
        "note": "The current exporter inventories source readiness but does not infer missing attribute-level matches.",
    }
    (output / "attribute_flow_schema.json").write_text(
        json.dumps(attribute_flow_schema, indent=2) + "\n", encoding="utf-8"
    )

    selected_runs = []
    for entry in entries:
        record = next(r for r in inventory if r["model"]==entry["model"] and r["architecture"]==entry["architecture"] and r["persona_idx"]==entry["persona_idx"])
        selected_runs.append({**record, "created_at": entry.get("created_at", "")})
    repeat_policy = {
        "post_repeat_limit": post_repeat_limit,
        "selection": "lowest_repeat_idx_then_record_index",
        "pre_rerank_unchanged": True,
    }
    lock = {"schema_version": 1, "command": "export_cimemories_study_artifacts", "created_at": datetime.now().astimezone().isoformat(), "dataset_filter": dataset_filter, "model_filters": model_filters, "expected_personas": expected_personas, "selection_rule": "strongest complete post scope, then pre scope, then newest creation time", "response_repeat_policy": repeat_policy, "selected_runs": selected_runs}
    if analysis_provenance:
        lock["analysis_provenance"] = analysis_provenance
    (output / "study_artifact_inputs.json").write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "command": "export_cimemories_study_artifacts", "created_at": lock["created_at"],
        "dataset_filter": dataset_filter, "expected_personas": expected_personas, "partial": partial,
        "bootstrap": {"method":"paired hierarchical persona/context","iterations":bootstrap_iterations},
        "models": models, "architectures": list(ARCHITECTURES), "stages":["pre","post"],
        "metrics": METRIC_LABELS, "coverage": coverage,
        "response_repeat_policy": repeat_policy,
        "artifacts": ["REPORT.html","study_artifact_inputs.json","coverage.csv","condition_metrics.csv","reranking_effects.csv","architecture_contrasts.csv","model_contrasts.csv","interaction_contrasts.csv","discarded_context_sensitivity.csv","persona_reranking_effects.csv","scenario_domain_reranking_effects.csv","context_metrics.csv","experiment_inventory.csv","memory_stage_inventory.csv","attribute_flow_schema.json","figures/*.svg",*latex_artifacts],
    }
    if analysis_provenance:
        manifest["analysis_provenance"] = analysis_provenance
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (output / "REPORT.html").write_text(_report_html(coverage=coverage,conditions=conditions,effects=effects,architecture_contrasts=architecture_contrasts,model_contrasts=model_contrasts,interactions=interactions,inventory=inventory,expected_personas=expected_personas,iterations=bootstrap_iterations,partial=partial,analysis_provenance=analysis_provenance,post_repeat_limit=post_repeat_limit),encoding="utf-8")
    (output / "README.md").write_text(
        "# CIMemories factorial study artifacts\n\nOpen `REPORT.html` for the visual report. The exact selected inputs are locked in `study_artifact_inputs.json`.\n\n"
        "The bundle separates absolute operating points, within-condition reranking effects, architecture contrasts, model contrasts, and difference-in-differences. Pairwise contrasts use exact shared persona/context cells and record their own sample sizes. `memory_stage_inventory.csv` is the bridge to the planned attribute-level retrieval and exposure analysis.\n\n"
        + ("Post-rerank response metrics use only the first saved repetition per context; pre-rerank metrics are unchanged.\n" if post_repeat_limit == 1 else "Post-rerank response metrics use all saved repetitions.\n"),
        encoding="utf-8",
    )
    if progress_callback is not None:
        progress_callback("response-artifacts", 1, 1, "response bundle complete")
    return output.resolve()
