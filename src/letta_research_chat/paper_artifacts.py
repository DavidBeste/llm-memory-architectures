from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


METRICS = (
    ("completion", "necessary_recall", True),
    ("leakage", "inappropriate_leak_rate", False),
    ("ambiguous", "ambiguous_exposure_rate", False),
)
COLORS = ("#2474b5", "#e07a24", "#3a9855", "#8e63b5", "#b64b59")


def _architecture_label(name: str) -> str:
    """Return a compact, paper-friendly label for a memory architecture."""
    lowered = name.lower()
    if lowered.startswith("list"):
        return "List"
    if lowered.startswith("graph"):
        return "Graph"
    if lowered.startswith("profile"):
        return "Profile"
    return name


@dataclass(frozen=True)
class Observation:
    persona_idx: int
    persona_name: str
    context_idx: int
    recipient: str
    task: str
    completion: float
    leakage: float
    ambiguous: float
    context_discarded: bool


@dataclass(frozen=True)
class PaperRun:
    path: Path
    mode: int
    name: str
    dataset: str
    labels_file: str
    created_at: str
    observations: tuple[Observation, ...]
    pipeline_summaries: tuple[dict[str, Any], ...]


def _artifact_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw).expanduser()
    candidates = (path, manifest_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Referenced metrics file does not exist: {raw}")


def _load_dataset_run(value: str) -> PaperRun:
    path = Path(value).expanduser().resolve()
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset-run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("command") != "run_privacy_pipeline_cimemories_dataset":
        raise ValueError(f"Expected a dataset-level privacy pipeline run: {path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Dataset run is not complete: {path} ({manifest.get('status')})")
    expected_personas = int(manifest.get("persona_count") or 0)
    personas = manifest.get("personas") or []
    if expected_personas < 1 or len(personas) != expected_personas:
        raise ValueError(f"Dataset run has incomplete persona coverage: {len(personas)}/{expected_personas}")

    observations: list[Observation] = []
    pipeline_summaries: list[dict[str, Any]] = []
    for persona in personas:
        pipeline_value = persona.get("local_pipeline")
        if not isinstance(pipeline_value, str):
            raise ValueError(f"Persona entry has no local_pipeline in {manifest_path}")
        pipeline_path = Path(pipeline_value)
        if not pipeline_path.is_absolute():
            pipeline_path = Path.cwd() / pipeline_path
        pipeline_manifest = pipeline_path / "manifest.json"
        summary = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
        pipeline_summaries.append(summary)
        contexts = summary.get("contexts") or []
        if len(contexts) != int(summary.get("context_count") or 0):
            raise ValueError(f"Incomplete contexts in {pipeline_manifest}")
        for context in contexts:
            metrics_path = _artifact_path(str(context["privacy_metrics_cimemories_json"]), pipeline_manifest)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics") or {}
            observations.append(
                Observation(
                    persona_idx=int(payload["persona_idx"]),
                    persona_name=str(payload["persona_name"]),
                    context_idx=int(payload["context_idx"]),
                    recipient=str(payload.get("recipient") or context.get("recipient") or ""),
                    task=str(payload.get("task") or context.get("task") or ""),
                    completion=float(metrics.get("necessary_recall") or 0.0),
                    leakage=float(metrics.get("inappropriate_leak_rate") or 0.0),
                    ambiguous=float(metrics.get("ambiguous_exposure_rate") or 0.0),
                    context_discarded=bool(payload.get("context_discarded", False)),
                )
            )
    return PaperRun(
        path=path,
        mode=int(manifest["memory_mode"]),
        name=str(manifest.get("memory_mode_name") or f"mode-{manifest['memory_mode']}"),
        dataset=str(manifest.get("dataset_name") or ""),
        labels_file=str(manifest.get("context_labels_file") or "model-generated"),
        created_at=str(manifest.get("created_at") or ""),
        observations=tuple(observations),
        pipeline_summaries=tuple(pipeline_summaries),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _bootstrap_intervals(
    runs: list[PaperRun], *, samples: int = 2000, seed: int = 20260907
) -> dict[str, dict[str, tuple[float, float]]]:
    """Paired hierarchical bootstrap over personas and contexts."""
    maps = [{(o.persona_idx, o.context_idx): o for o in run.observations} for run in runs]
    keys = set(maps[0])
    if any(set(mapping) != keys for mapping in maps[1:]):
        raise ValueError("Runs do not contain identical persona/context cells")
    personas = sorted({persona for persona, _ in keys})
    contexts = {persona: sorted(context for p, context in keys if p == persona) for persona in personas}
    rng = random.Random(seed)
    draws = {run.name: {metric[0]: [] for metric in METRICS} for run in runs}
    for _ in range(samples):
        sampled_keys: list[tuple[int, int]] = []
        for persona in rng.choices(personas, k=len(personas)):
            sampled_keys.extend(
                (persona, context)
                for context in rng.choices(contexts[persona], k=len(contexts[persona]))
            )
        for run, mapping in zip(runs, maps):
            for metric_name, _, _ in METRICS:
                draws[run.name][metric_name].append(
                    _mean([getattr(mapping[key], metric_name) for key in sampled_keys])
                )
    return {
        run_name: {
            metric_name: (_percentile(values, 0.025), _percentile(values, 0.975))
            for metric_name, values in metrics.items()
        }
        for run_name, metrics in draws.items()
    }


def _paired_bootstrap_differences(
    runs: list[PaperRun], baseline: PaperRun, *, samples: int = 2000, seed: int = 20260907
) -> dict[str, dict[str, tuple[float, float, float]]]:
    maps = {run.name: {(o.persona_idx, o.context_idx): o for o in run.observations} for run in runs}
    keys = set(maps[baseline.name])
    personas = sorted({persona for persona, _ in keys})
    contexts = {persona: sorted(context for p, context in keys if p == persona) for persona in personas}
    rng = random.Random(seed)
    draws = {
        run.name: {metric[0]: [] for metric in METRICS}
        for run in runs if run.name != baseline.name
    }
    for _ in range(samples):
        sampled_keys: list[tuple[int, int]] = []
        for persona in rng.choices(personas, k=len(personas)):
            sampled_keys.extend(
                (persona, context)
                for context in rng.choices(contexts[persona], k=len(contexts[persona]))
            )
        for run in runs:
            if run.name == baseline.name:
                continue
            for metric_name, _, _ in METRICS:
                draws[run.name][metric_name].append(
                    _mean([
                        getattr(maps[run.name][key], metric_name)
                        - getattr(maps[baseline.name][key], metric_name)
                        for key in sampled_keys
                    ])
                )
    output: dict[str, dict[str, tuple[float, float, float]]] = {}
    for run in runs:
        if run.name == baseline.name:
            continue
        output[run.name] = {}
        for metric_name, _, _ in METRICS:
            point = _mean([
                getattr(maps[run.name][key], metric_name)
                - getattr(maps[baseline.name][key], metric_name)
                for key in keys
            ])
            values = draws[run.name][metric_name]
            output[run.name][metric_name] = (
                point,
                _percentile(values, 0.025),
                _percentile(values, 0.975),
            )
    return output


def _csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _aggregate(run: PaperRun) -> dict[str, float]:
    return {name: _mean([getattr(item, name) for item in run.observations]) for name, _, _ in METRICS}


def _group(run: PaperRun, field: str) -> dict[Any, dict[str, Any]]:
    grouped: dict[Any, list[Observation]] = {}
    for item in run.observations:
        grouped.setdefault(getattr(item, field), []).append(item)
    return {
        key: {
            "label": values[0].persona_name if field == "persona_idx" else values[0].recipient,
            "task": "" if field == "persona_idx" else values[0].task,
            **{name: _mean([getattr(item, name) for item in values]) for name, _, _ in METRICS},
        }
        for key, values in grouped.items()
    }


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#20242a}.title{font-size:20px;font-weight:700}.label{font-size:11px}.small{font-size:9px}.axis{stroke:#6c737c;stroke-width:1}.grid{stroke:#dfe3e8;stroke-width:1}</style>',
        f'<text class="title" x="20" y="30">{html.escape(title)}</text>',
    ]


def _heat_color(value: float, scale: float) -> str:
    ratio = min(1.0, abs(value) / scale) if scale else 0.0
    target = (49, 151, 91) if value >= 0 else (210, 67, 62)
    rgb = tuple(round(248 + (channel - 248) * ratio) for channel in target)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _write_heatmap(path: Path, runs: list[PaperRun], baseline: PaperRun) -> None:
    grouped = {run.name: _group(run, "context_idx") for run in runs}
    others = [run for run in runs if run.name != baseline.name]
    contexts = sorted(grouped[baseline.name])
    columns = [(run, metric[0], metric[2]) for metric in METRICS for run in others]
    desirable: dict[tuple[int, str, str], float] = {}
    for context in contexts:
        for run, metric, higher in columns:
            delta = grouped[run.name][context][metric] - grouped[baseline.name][context][metric]
            desirable[(context, run.name, metric)] = delta if higher else -delta
    scales = {
        metric: max(abs(value) for (ctx, run, met), value in desirable.items() if met == metric) or 1.0
        for metric, _, _ in METRICS
    }
    left, top, row_h, cell_w = 245, 92, 24, 112
    width, height = left + len(columns) * cell_w + 25, top + len(contexts) * row_h + 55
    svg = _svg_start(
        width,
        height,
        f"Context-level differences relative to {_architecture_label(baseline.name)}",
    )
    for col, (run, metric, _) in enumerate(columns):
        x = left + col * cell_w
        svg.append(f'<text class="small" x="{x + cell_w/2}" y="58" text-anchor="middle">{html.escape(metric.title())}</text>')
        svg.append(
            f'<text class="small" x="{x + cell_w/2}" y="74" text-anchor="middle">'
            f'{html.escape(_architecture_label(run.name))}</text>'
        )
    for row, context in enumerate(contexts):
        y = top + row * row_h
        label = f"{context}: {grouped[baseline.name][context]['label']}"
        if len(label) > 37:
            label = label[:36] + "…"
        svg.append(f'<text class="label" x="{left - 8}" y="{y + 16}" text-anchor="end">{html.escape(label)}</text>')
        for col, (run, metric, _) in enumerate(columns):
            value = desirable[(context, run.name, metric)]
            x = left + col * cell_w
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{row_h-2}" fill="{_heat_color(value, scales[metric])}"/>')
            svg.append(f'<text class="small" x="{x + (cell_w-2)/2}" y="{y + 15}" text-anchor="middle">{value*100:+.1f}</text>')
    svg.append(f'<text class="small" x="20" y="{height-18}">Green is better; red is worse. Leakage and ambiguous-exposure differences are direction-adjusted. Values are percentage points.</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_persona_plot(path: Path, runs: list[PaperRun]) -> None:
    grouped = {run.name: _group(run, "persona_idx") for run in runs}
    personas = sorted(grouped[runs[0].name])
    left, top, plot_w, row_h = 180, 70, 720, 42
    width, height = 940, top + len(personas) * row_h + 70
    svg = _svg_start(width, height, "Completion by persona")
    for tick in range(0, 101, 10):
        x = left + plot_w * tick / 100
        svg.append(f'<line class="grid" x1="{x}" y1="{top-10}" x2="{x}" y2="{height-48}"/>')
        svg.append(f'<text class="small" x="{x}" y="{height-28}" text-anchor="middle">{tick}%</text>')
    for row, persona in enumerate(personas):
        y = top + row * row_h
        values = [grouped[run.name][persona]["completion"] * 100 for run in runs]
        svg.append(f'<text class="label" x="{left-12}" y="{y+5}" text-anchor="end">{html.escape(grouped[runs[0].name][persona]["label"])}</text>')
        svg.append(f'<line x1="{left+plot_w*min(values)/100}" y1="{y}" x2="{left+plot_w*max(values)/100}" y2="{y}" stroke="#abb1b8" stroke-width="2"/>')
        for index, (run, value) in enumerate(zip(runs, values)):
            x = left + plot_w * value / 100
            offset = (index - (len(runs)-1)/2) * 7
            svg.append(f'<circle cx="{x}" cy="{y+offset}" r="5" fill="{COLORS[index]}"/>')
    for index, run in enumerate(runs):
        x = left + index * 190
        svg.append(f'<circle cx="{x}" cy="{height-8}" r="5" fill="{COLORS[index]}"/><text class="label" x="{x+10}" y="{height-4}">{html.escape(run.name)}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_tradeoff(path: Path, runs: list[PaperRun], aggregates: dict[str, dict[str, float]]) -> None:
    width, height, left, top, plot_w, plot_h = 760, 550, 85, 65, 620, 365
    max_x = max(aggregates[run.name]["leakage"] for run in runs) * 1.25 or 0.01
    max_y = max(aggregates[run.name]["completion"] for run in runs) * 1.2 or 1.0
    svg = _svg_start(width, height, "Privacy–utility trade-off")
    for tick in range(6):
        x = left + plot_w * tick / 5
        y = top + plot_h * tick / 5
        svg.append(f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top+plot_h}"/>')
        svg.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}"/>')
        svg.append(f'<text class="small" x="{x}" y="{top+plot_h+20}" text-anchor="middle">{max_x*100*tick/5:.2f}%</text>')
        svg.append(f'<text class="small" x="{left-10}" y="{top+plot_h-y+top+4}" text-anchor="end">{max_y*100*(5-tick)/5:.0f}%</text>')
    svg.append(f'<text class="label" x="{left+plot_w/2}" y="{top+plot_h+50}" text-anchor="middle">Inappropriate leakage (lower is better)</text>')
    svg.append(f'<text class="label" transform="translate(22 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Necessary completion (higher is better)</text>')
    for index, run in enumerate(runs):
        values = aggregates[run.name]
        x = left + plot_w * values["leakage"] / max_x
        y = top + plot_h * (1 - values["completion"] / max_y)
        svg.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{COLORS[index]}"/>')
    legend_y = height - 18
    legend_width = min(170, (width - 80) / len(runs))
    legend_left = (width - legend_width * len(runs)) / 2
    for index, run in enumerate(runs):
        x = legend_left + index * legend_width
        svg.append(f'<circle cx="{x}" cy="{legend_y-4}" r="5" fill="{COLORS[index]}"/>')
        svg.append(
            f'<text class="label" x="{x+10}" y="{legend_y}">'
            f'{html.escape(_architecture_label(run.name))}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _weighted(values: list[tuple[float | None, int]]) -> float | None:
    present = [(value, count) for value, count in values if value is not None and count > 0]
    if not present:
        return None
    return sum(value * count for value, count in present) / sum(count for _, count in present)


def _nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def _efficiency_summary(run: PaperRun) -> dict[str, Any]:
    deployed_samples: list[float] = []
    responses = 0
    contexts = 0
    tokens_per_query: list[tuple[float | None, int]] = []
    online_coverage: list[tuple[float | None, int]] = []
    generation_mean: list[tuple[float | None, int]] = []
    preparation_mean: list[tuple[float | None, int]] = []
    initialization_ms = 0.0
    experiment_wall_ms = 0.0
    evaluation_ms = 0.0
    tracked_tokens = 0.0
    exact_components = 0
    applicable_components = 0
    unavailable: set[str] = set()

    for summary in run.pipeline_summaries:
        efficiency = summary.get("efficiency") if isinstance(summary.get("efficiency"), dict) else {}
        counts = efficiency.get("counts") if isinstance(efficiency.get("counts"), dict) else {}
        timings = efficiency.get("timings_ms") if isinstance(efficiency.get("timings_ms"), dict) else {}
        tokens = efficiency.get("tokens") if isinstance(efficiency.get("tokens"), dict) else {}
        one_time = efficiency.get("one_time") if isinstance(efficiency.get("one_time"), dict) else {}
        response_count = int(counts.get("responses") or 0)
        context_count = int(counts.get("contexts") or 0)
        responses += response_count
        contexts += context_count
        tokens_per_query.append((_number(tokens.get("deployed_exact_tokens_per_query")), response_count))
        generation_coverage = _number(tokens.get("generation_exact_coverage"))
        preparation_coverage = _number(tokens.get("memory_preparation_exact_coverage"))
        coverage_parts = [part for part in (generation_coverage, preparation_coverage) if part is not None]
        online_coverage.append((_mean(coverage_parts) if coverage_parts else None, response_count))
        generation = timings.get("generation") if isinstance(timings.get("generation"), dict) else {}
        preparation = timings.get("memory_preparation") if isinstance(timings.get("memory_preparation"), dict) else {}
        generation_mean.append((_number(generation.get("mean")), int(generation.get("count") or response_count)))
        preparation_mean.append((_number(preparation.get("mean")), int(preparation.get("count") or context_count)))
        initialization_ms += _number(one_time.get("memory_initialization_ms")) or 0.0
        experiment_wall_ms += _number(timings.get("experiment_wall")) or 0.0
        overhead = summary.get("evaluation_overhead") if isinstance(summary.get("evaluation_overhead"), dict) else {}
        evaluation_ms += sum(
            _number(overhead.get(key)) or 0.0
            for key in ("exposure_judging_ms", "context_labeling_and_privacy_metrics_ms")
        )

        ledger = summary.get("pipeline_usage_ledger") if isinstance(summary.get("pipeline_usage_ledger"), dict) else {}
        logical = ledger.get("logical_exact_total") if isinstance(ledger.get("logical_exact_total"), dict) else {}
        tracked_tokens += _number(logical.get("total_tokens")) or 0.0
        for component in ledger.get("components") or []:
            if not isinstance(component, dict) or component.get("availability") == "not_applicable":
                continue
            metadata_value = component.get("metadata")
            if isinstance(metadata_value, dict) and metadata_value.get("non_billing_view"):
                continue
            name = str(component.get("component") or "unknown")
            # External labels were imported rather than generated during these
            # experiments. Their older artifacts lack token telemetry, but
            # counting 49 no-call setup records per persona as uncovered paid
            # components would make architecture coverage meaningless.
            if run.labels_file != "model-generated" and name.startswith("context_") and name.endswith(".context_labeling"):
                continue
            applicable_components += 1
            if component.get("availability") == "exact":
                exact_components += 1
            else:
                unavailable.add(name)

        artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
        responses_value = artifacts.get("responses_jsonl")
        if isinstance(responses_value, str):
            responses_path = Path(responses_value).expanduser()
            if not responses_path.is_file():
                responses_path = run.path / responses_path
            preparation_by_context = efficiency.get("context_preparation")
            preparation_by_context = preparation_by_context if isinstance(preparation_by_context, dict) else {}
            if responses_path.is_file():
                for line in responses_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record_efficiency = record.get("efficiency") if isinstance(record.get("efficiency"), dict) else {}
                    record_timings = record_efficiency.get("timings_ms") if isinstance(record_efficiency.get("timings_ms"), dict) else {}
                    observed = _number(record_timings.get("online_observed"))
                    context_idx = record.get("context_idx")
                    context_prep = preparation_by_context.get(str(context_idx), {}) if isinstance(context_idx, int) else {}
                    prep_ms = _number(context_prep.get("duration_ms")) if isinstance(context_prep, dict) else None
                    if observed is not None:
                        deployed_samples.append(observed + (prep_ms or 0.0))

    deployed_mean = _mean(deployed_samples) if deployed_samples else None
    return {
        "responses": responses,
        "contexts": contexts,
        "deployed_exact_tokens_per_query": _weighted(tokens_per_query),
        "online_exact_token_coverage": _weighted(online_coverage),
        "generation_mean_ms": _weighted(generation_mean),
        "preparation_mean_ms": _weighted(preparation_mean),
        "deployed_mean_ms": deployed_mean,
        "deployed_p95_ms": _nearest_rank(deployed_samples, 0.95),
        "initialization_total_ms": initialization_ms,
        "initialization_mean_ms": initialization_ms / len(run.pipeline_summaries),
        "architecture_experiment_wall_ms": experiment_wall_ms,
        "evaluation_overhead_ms": evaluation_ms,
        "end_to_end_accounted_ms": experiment_wall_ms + evaluation_ms,
        "tracked_exact_pipeline_tokens": tracked_tokens,
        "pipeline_component_coverage": exact_components / applicable_components if applicable_components else None,
        "unavailable_components": sorted(unavailable),
    }


def _fmt(value: float | None, scale: float = 1.0, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value / scale:.{digits}f}"


def _write_efficiency_plot(
    path: Path,
    runs: list[PaperRun],
    aggregates: dict[str, dict[str, float]],
    efficiency: dict[str, dict[str, Any]],
) -> None:
    width, height = 980, 480
    svg = _svg_start(width, height, "Effectiveness–efficiency trade-offs")
    panels = (
        ("deployed_exact_tokens_per_query", "Deployed exact tokens/query", 70),
        ("deployed_mean_ms", "Deployed mean latency (s)", 550),
    )
    for key, label, left in panels:
        top, plot_w, plot_h = 90, 360, 285
        raw = [efficiency[run.name].get(key) for run in runs]
        values = [(value / 1000 if key.endswith("_ms") and value is not None else value) for value in raw]
        numeric = [value for value in values if value is not None]
        max_x = max(numeric) * 1.2 if numeric else 1.0
        max_y = max(aggregates[run.name]["completion"] for run in runs) * 1.2
        svg.append(f'<text class="label" x="{left+plot_w/2}" y="{height-28}" text-anchor="middle">{html.escape(label)}</text>')
        for tick in range(6):
            x = left + plot_w * tick / 5
            y = top + plot_h * tick / 5
            svg.append(f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top+plot_h}"/>')
            svg.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}"/>')
            svg.append(f'<text class="small" x="{x}" y="{top+plot_h+18}" text-anchor="middle">{max_x*tick/5:.1f}</text>')
        for index, (run, value) in enumerate(zip(runs, values)):
            if value is None:
                continue
            x = left + plot_w * value / max_x
            y = top + plot_h * (1 - aggregates[run.name]["completion"] / max_y)
            svg.append(f'<circle cx="{x}" cy="{y}" r="7" fill="{COLORS[index]}"/>')
    legend_width = 150
    legend_left = (width - legend_width * len(runs)) / 2
    for index, run in enumerate(runs):
        x = legend_left + index * legend_width
        svg.append(f'<circle cx="{x}" cy="55" r="5" fill="{COLORS[index]}"/>')
        svg.append(
            f'<text class="label" x="{x+10}" y="59">'
            f'{html.escape(_architecture_label(run.name))}</text>'
        )
    svg.append('<text class="label" transform="translate(18 235) rotate(-90)" text-anchor="middle">Necessary completion (higher is better)</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


FIGURE_TEMPLATES = {
    "context_difference_heatmap": {
        "width": r"\textwidth",
        "placement": "p",
        "caption": (
            "Context-level performance differences relative to the {baseline} baseline, "
            "averaged across {cell_scope}. Completion differences retain their original direction; "
            "leakage and ambiguous-exposure differences are direction-adjusted so that green always "
            "denotes an improvement and red a deterioration. Cell values are percentage points."
        ),
        "label": "fig:privacy-context-differences",
    },
    "persona_completion": {
        "width": r"0.95\textwidth",
        "placement": "t",
        "caption": (
            "Necessary-attribute completion by persona and memory architecture, macro-averaged over "
            "{cell_scope} and {repeats} repeated responses per context. Horizontal spans show the "
            "range across architectures for each persona."
        ),
        "label": "fig:privacy-persona-completion",
    },
    "privacy_utility": {
        "width": r"0.72\columnwidth",
        "placement": "t",
        "caption": (
            "Aggregate privacy--utility trade-off across memory architectures. Utility is necessary-"
            "attribute completion and privacy risk is inappropriate-attribute leakage; points toward "
            "the upper left are preferable. Values are macro-averaged over persona--context cells."
        ),
        "label": "fig:privacy-utility-tradeoff",
    },
    "efficiency_tradeoff": {
        "width": r"\textwidth",
        "placement": "t",
        "caption": (
            "Effectiveness--efficiency trade-offs. The left panel compares necessary-attribute "
            "completion with exact tracked deployed tokens per query; the right panel compares it "
            "with deployed mean latency, including contextual memory preparation. Backend-internal "
            "operations without telemetry are excluded from token totals rather than counted as zero."
        ),
        "label": "fig:privacy-efficiency-tradeoff",
    },
}


def _write_overleaf_templates(
    output: Path, *, baseline_name: str, cell_scope: str, repeats: int
) -> list[str]:
    includes = output / "includes"
    artifacts: list[str] = []
    for stem, spec in FIGURE_TEMPLATES.items():
        filename = f"{stem}.tex"
        lines = [
            rf"\begin{{figure*}}[{spec['placement']}]",
            r"  \centering",
            rf"  \includesvg[width={spec['width']}]{{figures/{stem}.svg}}",
            rf"  \caption{{{spec['caption'].format(baseline=baseline_name, cell_scope=cell_scope, repeats=repeats)}}}",
            rf"  \label{{{spec['label']}}}",
            r"\end{figure*}",
            "",
        ]
        # Column-width plots should use a normal single-column float.
        if spec["width"] == r"0.72\columnwidth":
            lines[0] = rf"\begin{{figure}}[{spec['placement']}]"
            lines[-2] = r"\end{figure}"
        (includes / filename).write_text("\n".join(lines), encoding="utf-8")
        artifacts.append(f"includes/{filename}")
    (includes / "overleaf_preamble.tex").write_text(
        "% Add this file to the document preamble with \\input{includes/overleaf_preamble.tex}.\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{svg}\n"
        "\\svgsetup{inkscapelatex=false}\n",
        encoding="utf-8",
    )
    artifacts.append("includes/overleaf_preamble.tex")
    (includes / "overleaf_figures.tex").write_text(
        "% Include selected figures where they should appear; floats may move.\n"
        + "".join(rf"\input{{includes/{stem}.tex}}" + "\n" for stem in FIGURE_TEMPLATES),
        encoding="utf-8",
    )
    artifacts.append("includes/overleaf_figures.tex")
    return artifacts


def export_privacy_paper_artifacts(
    run_paths: list[str],
    *,
    output_dir: str | None = None,
    baseline_name: str | None = None,
    exclude_discarded_contexts: bool = False,
) -> Path:
    if len(run_paths) < 2:
        raise ValueError("At least two complete dataset runs are required")
    runs = [_load_dataset_run(value) for value in run_paths]
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("Every run must have a distinct memory_mode_name")
    if len({run.dataset for run in runs}) != 1:
        raise ValueError("Runs use different datasets")
    if len({run.labels_file for run in runs}) != 1:
        raise ValueError("Runs use different ground-truth label sources")
    cell_sets = [{(o.persona_idx, o.context_idx) for o in run.observations} for run in runs]
    if any(cells != cell_sets[0] for cells in cell_sets[1:]):
        raise ValueError("Runs do not have identical persona/context coverage")
    original_cell_count = len(cell_sets[0])
    discarded_cell_count = 0
    if exclude_discarded_contexts:
        discarded_sets = [
            {(o.persona_idx, o.context_idx) for o in run.observations if o.context_discarded}
            for run in runs
        ]
        if any(cells != discarded_sets[0] for cells in discarded_sets[1:]):
            raise ValueError(
                "Runs do not have identical context_discarded masks; refusing an unpaired comparison"
            )
        discarded_cell_count = len(discarded_sets[0])
        if discarded_cell_count == original_cell_count:
            raise ValueError("All persona/context cells are marked context_discarded")
        runs = [
            PaperRun(
                path=run.path,
                mode=run.mode,
                name=run.name,
                dataset=run.dataset,
                labels_file=run.labels_file,
                created_at=run.created_at,
                observations=tuple(o for o in run.observations if not o.context_discarded),
                pipeline_summaries=run.pipeline_summaries,
            )
            for run in runs
        ]
    baseline = runs[0]
    if baseline_name:
        matches = [run for run in runs if baseline_name in (run.name, str(run.mode))]
        if len(matches) != 1:
            raise ValueError(f"Baseline {baseline_name!r} did not match exactly one run")
        baseline = matches[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(output_dir).expanduser() if output_dir else Path("research_outputs") / f"privacy-paper-artifacts_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    includes = output / "includes"
    includes.mkdir()
    aggregates = {run.name: _aggregate(run) for run in runs}
    intervals = _bootstrap_intervals(runs)
    analysis_scope = (
        f"{len(runs[0].observations)} ground-truth-eligible persona--context cells "
        f"({discarded_cell_count} discarded-label cells excluded)"
        if exclude_discarded_contexts
        else f"all {original_cell_count} persona--context cells"
    )

    aggregate_rows: list[list[Any]] = []
    for run in runs:
        row: list[Any] = [run.name, run.mode, len(run.observations)]
        for metric, _, _ in METRICS:
            low, high = intervals[run.name][metric]
            row.extend([aggregates[run.name][metric] * 100, low * 100, high * 100])
        aggregate_rows.append(row)
    _csv(
        output / "aggregate_metrics.csv",
        ["architecture", "mode", "persona_context_cells", *[item for metric, _, _ in METRICS for item in (f"{metric}_percent", f"{metric}_ci_low", f"{metric}_ci_high")]],
        aggregate_rows,
    )

    persona_groups = {run.name: _group(run, "persona_idx") for run in runs}
    persona_rows: list[list[Any]] = []
    for persona in sorted(persona_groups[runs[0].name]):
        for run in runs:
            values = persona_groups[run.name][persona]
            persona_rows.append([persona, values["label"], run.name, *[values[m[0]] * 100 for m in METRICS]])
    _csv(output / "persona_metrics.csv", ["persona_idx", "persona", "architecture", "completion_percent", "leakage_percent", "ambiguous_percent"], persona_rows)

    context_groups = {run.name: _group(run, "context_idx") for run in runs}
    context_rows: list[list[Any]] = []
    delta_rows: list[list[Any]] = []
    for context in sorted(context_groups[runs[0].name]):
        base = context_groups[baseline.name][context]
        for run in runs:
            values = context_groups[run.name][context]
            context_rows.append([context, values["label"], values["task"], run.name, *[values[m[0]] * 100 for m in METRICS]])
            if run.name != baseline.name:
                delta_rows.append([context, values["label"], values["task"], run.name, baseline.name, *[(values[m[0]] - base[m[0]]) * 100 for m in METRICS]])
    _csv(output / "context_metrics.csv", ["context_idx", "recipient", "task", "architecture", "completion_percent", "leakage_percent", "ambiguous_percent"], context_rows)
    _csv(output / "context_deltas.csv", ["context_idx", "recipient", "task", "architecture", "baseline", "completion_delta_pp", "leakage_delta_pp", "ambiguous_delta_pp"], delta_rows)

    paired = _paired_bootstrap_differences(runs, baseline)
    paired_rows: list[list[Any]] = []
    for run in runs:
        if run.name == baseline.name:
            continue
        for metric, _, higher_is_better in METRICS:
            point, low, high = paired[run.name][metric]
            paired_rows.append(
                [run.name, baseline.name, metric, point * 100, low * 100, high * 100,
                 "higher" if higher_is_better else "lower", low > 0 or high < 0]
            )
    _csv(
        output / "paired_comparisons.csv",
        ["architecture", "baseline", "metric", "difference_pp", "ci_low_pp", "ci_high_pp", "better_direction", "ci_excludes_zero"],
        paired_rows,
    )
    paired_latex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Architecture & Metric & Difference (pp) & 95\% CI \\", r"\midrule",
    ]
    for row in paired_rows:
        architecture, _, metric, point, low, high, _, excludes_zero = row
        rendered = f"{point:+.2f} & [{low:+.2f}, {high:+.2f}]"
        if excludes_zero:
            rendered = rf"\textbf{{{point:+.2f}}} & \textbf{{[{low:+.2f}, {high:+.2f}]}}"
        paired_latex.append(f"{_latex_escape(str(architecture))} & {_latex_escape(str(metric))} & {rendered} \\\\ ")
    paired_latex.extend([
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Paired differences relative to {_latex_escape(baseline.name)} over {analysis_scope}, with hierarchical-bootstrap intervals over personas and contexts. Positive differences indicate larger metric values; lower is preferable for leakage and ambiguous exposure. Bold intervals exclude zero.}}",
        r"\label{tab:privacy-memory-paired}", r"\end{table}", "",
    ])
    (includes / "paired_comparisons.tex").write_text("\n".join(paired_latex), encoding="utf-8")

    best = {
        metric: (max if higher else min)(aggregates[run.name][metric] for run in runs)
        for metric, _, higher in METRICS
    }
    latex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Architecture & Completion $\uparrow$ & Leakage $\downarrow$ & Ambiguous $\downarrow$ \\",
        r"\midrule",
    ]
    for run in runs:
        cells = []
        for metric, _, _ in METRICS:
            value = aggregates[run.name][metric]
            low, high = intervals[run.name][metric]
            rendered = f"{value*100:.1f} [{low*100:.1f}, {high*100:.1f}]"
            cells.append(rf"\textbf{{{rendered}}}" if math.isclose(value, best[metric]) else rendered)
        latex.append(f"{_latex_escape(run.name)} & " + " & ".join(cells) + r" \\")
    latex.extend([
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Macro performance over {analysis_scope}, with 95\% paired hierarchical bootstrap intervals over personas and contexts.}}",
        r"\label{tab:privacy-memory-architectures}", r"\end{table}", "",
    ])
    (includes / "aggregate_metrics.tex").write_text("\n".join(latex), encoding="utf-8")

    efficiency = {run.name: _efficiency_summary(run) for run in runs}
    efficiency_columns = [
        "responses", "contexts", "deployed_exact_tokens_per_query", "online_exact_token_coverage",
        "generation_mean_ms", "preparation_mean_ms", "deployed_mean_ms", "deployed_p95_ms",
        "initialization_total_ms", "initialization_mean_ms", "architecture_experiment_wall_ms",
        "evaluation_overhead_ms", "end_to_end_accounted_ms", "tracked_exact_pipeline_tokens",
        "pipeline_component_coverage", "unavailable_components",
    ]
    _csv(
        output / "efficiency_summary.csv",
        ["architecture", "mode", *efficiency_columns],
        [
            [
                run.name,
                run.mode,
                *[
                    ";".join(efficiency[run.name][column])
                    if column == "unavailable_components"
                    else efficiency[run.name][column]
                    for column in efficiency_columns
                ],
            ]
            for run in runs
        ],
    )
    efficiency_persona_rows: list[list[Any]] = []
    for run in runs:
        for summary in run.pipeline_summaries:
            single = PaperRun(
                path=run.path,
                mode=run.mode,
                name=run.name,
                dataset=run.dataset,
                labels_file=run.labels_file,
                created_at=run.created_at,
                observations=(),
                pipeline_summaries=(summary,),
            )
            values = _efficiency_summary(single)
            efficiency_persona_rows.append(
                [summary.get("persona_idx"), summary.get("persona_name"), run.name]
                + [
                    ";".join(values[column]) if column == "unavailable_components" else values[column]
                    for column in efficiency_columns
                ]
            )
    _csv(
        output / "efficiency_by_persona.csv",
        ["persona_idx", "persona", "architecture", *efficiency_columns],
        efficiency_persona_rows,
    )

    efficiency_latex = [
        r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Architecture & Tokens/query & Online coverage & Mean (s) & p95 (s) & Init. (min) & Run wall (h) & Eval. (h) & Tracked tokens (M) \\",
        r"\midrule",
    ]
    for run in runs:
        values = efficiency[run.name]
        efficiency_latex.append(
            f"{_latex_escape(run.name)} & "
            f"{_fmt(values['deployed_exact_tokens_per_query'], digits=0)} & "
            f"{_fmt(values['online_exact_token_coverage'], scale=0.01, digits=0)}\\% & "
            f"{_fmt(values['deployed_mean_ms'], scale=1000, digits=2)} & "
            f"{_fmt(values['deployed_p95_ms'], scale=1000, digits=2)} & "
            f"{_fmt(values['initialization_total_ms'], scale=60000, digits=1)} & "
            f"{_fmt(values['architecture_experiment_wall_ms'], scale=3600000, digits=2)} & "
            f"{_fmt(values['evaluation_overhead_ms'], scale=3600000, digits=2)} & "
            f"{_fmt(values['tracked_exact_pipeline_tokens'], scale=1000000, digits=2)} \\\\"
        )
    efficiency_latex.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Efficiency across all personas. Tokens/query and online latency estimate a standalone deployed query, including contextual memory preparation. Initialization and run wall time sum the ten sequential persona runs. Evaluation time is excluded from architecture latency. Tracked tokens are exact provider-reported tokens and are a lower bound when backend-internal usage is unavailable; coverage details are retained in the accompanying CSV.}",
        r"\label{tab:privacy-memory-efficiency}", r"\end{table*}", "",
    ])
    (includes / "efficiency_summary.tex").write_text("\n".join(efficiency_latex), encoding="utf-8")

    figures = output / "figures"
    figures.mkdir()
    _write_heatmap(figures / "context_difference_heatmap.svg", runs, baseline)
    _write_persona_plot(figures / "persona_completion.svg", runs)
    _write_tradeoff(figures / "privacy_utility.svg", runs, aggregates)
    _write_efficiency_plot(figures / "efficiency_tradeoff.svg", runs, aggregates, efficiency)
    first_summary = runs[0].pipeline_summaries[0]
    overleaf_artifacts = _write_overleaf_templates(
        output,
        baseline_name=baseline.name,
        cell_scope=(
            f"ground-truth-eligible persona--context cells ($n={len(runs[0].observations)}$; "
            f"{discarded_cell_count} discarded-label cells excluded)"
            if exclude_discarded_contexts
            else f"all {int(first_summary.get('context_count') or 0)} disclosure contexts"
        ),
        repeats=int(first_summary.get("scenario_repeats") or 0),
    )

    provenance = {
        "created_at": timestamp,
        "dataset": runs[0].dataset,
        "labels_file": runs[0].labels_file,
        "baseline": baseline.name,
        "evaluation_cells": {
            "policy": "exclude-context-discarded" if exclude_discarded_contexts else "all-cells-legacy",
            "original": original_cell_count,
            "excluded": discarded_cell_count,
            "included": len(runs[0].observations),
        },
        "bootstrap": {"method": "paired hierarchical persona/context", "samples": 2000, "seed": 20260907},
        "runs": [{"path": str(run.path), "mode": run.mode, "name": run.name, "created_at": run.created_at, "cells": len(run.observations)} for run in runs],
        "artifacts": ["aggregate_metrics.csv", "includes/aggregate_metrics.tex", "paired_comparisons.csv", "includes/paired_comparisons.tex", "persona_metrics.csv", "context_metrics.csv", "context_deltas.csv", "figures/context_difference_heatmap.svg", "figures/persona_completion.svg", "figures/privacy_utility.svg", "efficiency_summary.csv", "includes/efficiency_summary.tex", "efficiency_by_persona.csv", "figures/efficiency_tradeoff.svg", *overleaf_artifacts],
    }
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Privacy paper artifacts

Dataset: `{runs[0].dataset}`  
Ground-truth labels: `{runs[0].labels_file}`  
Heatmap baseline: `{baseline.name}`
Evaluation-cell policy: `{"exclude-context-discarded" if exclude_discarded_contexts else "all-cells-legacy"}`
Persona-context cells: `{len(runs[0].observations)}` included, `{discarded_cell_count}` excluded from `{original_cell_count}` original cells

The aggregate table reports macro means over included persona-context cells. Confidence intervals use a deterministic paired hierarchical bootstrap over personas and contexts; the ten response repetitions are already summarized inside each saved metric file and are not treated as independent experimental units.

{"Cells marked `context_discarded` by ground-truth labeling are excluded consistently from every architecture and metric. This complete-case policy requires at least one necessary and one inappropriate ground-truth attribute in each retained persona-context cell." if exclude_discarded_contexts else "This export retains the legacy all-cells behavior, under which undefined metric values are represented as zero. For a paper-oriented complete-case comparison, rerun with `--exclude-discarded-contexts`."}

`paired_comparisons.csv` and `.tex` report paired architecture-minus-baseline differences and are the appropriate artifacts for claims about whether architectures differ; marginal confidence-interval overlap in the aggregate table is not such a test.

The heatmap reports percentage-point differences relative to the baseline, direction-adjusted so green always means better. `context_metrics.csv` and `persona_metrics.csv` contain the underlying absolute values.

`efficiency_summary.csv` and `.tex` separate online deployed query cost, one-time memory initialization, architecture experiment wall time, and excluded evaluation overhead. Exact tracked tokens are lower bounds whenever a backend does not expose internal LLM or embedding usage; `pipeline_component_coverage` and `unavailable_components` make that limitation explicit. `efficiency_by_persona.csv` retains the underlying run-level values. SVG figures are vector graphics suitable for conversion to PDF by the paper toolchain.

For Overleaf, upload the complete directory without renaming files or flattening the sibling `includes/` and `figures/` directories. Add `\\input{{includes/overleaf_preamble.tex}}` in the document preamble, then either include individual snippets such as `\\input{{includes/context_difference_heatmap.tex}}` where desired or use `\\input{{includes/overleaf_figures.tex}}` to include all four. Figure snippets resolve their images as `figures/<name>.svg` from the Overleaf project root. Overleaf's `svg` package invokes Inkscape during compilation; if the selected compiler does not permit that conversion, convert the SVG files to PDF and replace `\\includesvg` with `\\includegraphics` in the snippets.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return output.resolve()


def discover_privacy_dataset_runs(
    root: str, *, dataset_filter: str, labels_filter: str, modes: list[int]
) -> list[str]:
    """Select the newest complete full-dataset run for each requested mode."""
    candidates: dict[int, list[tuple[str, Path]]] = {mode: [] for mode in modes}
    for manifest_path in Path(root).glob("privacy-pipeline-cimemories-dataset_*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mode = manifest.get("memory_mode")
        if (
            manifest.get("command") != "run_privacy_pipeline_cimemories_dataset"
            or manifest.get("status") != "complete"
            or mode not in candidates
            or dataset_filter.lower() not in str(manifest.get("dataset_name") or "").lower()
            or labels_filter.lower() not in str(manifest.get("context_labels_file") or "model-generated").lower()
        ):
            continue
        candidates[mode].append((str(manifest.get("created_at") or ""), manifest_path.parent))
    selected: list[str] = []
    missing: list[int] = []
    for mode in modes:
        choices = sorted(candidates[mode], reverse=True)
        if not choices:
            missing.append(mode)
        else:
            selected.append(str(choices[0][1]))
    if missing:
        raise FileNotFoundError(
            f"No matching complete dataset run for mode(s): {', '.join(map(str, missing))}"
        )
    return selected
