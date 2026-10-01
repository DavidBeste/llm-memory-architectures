from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import html
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any


METRICS = {
    "necessary_recall": "Necessary recall",
    "inappropriate_leak_rate": "Inappropriate leakage",
    "ambiguous_exposure_rate": "Ambiguous exposure",
    "average_exposed_total": "Mean exposed attributes",
}
ARCHITECTURES = ("list", "graph", "profile")
ARCH_COLORS = {"list": "#e76f51", "graph": "#2a9d8f", "profile": "#4568dc"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve(value: Any, manifest_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("artifact://"):
        relative = Path(value.removeprefix("artifact://"))
        if relative.is_absolute() or ".." in relative.parts:
            return None
        for parent in (manifest_path.parent, *manifest_path.parents):
            if (parent / "artifact_manifest.json").is_file():
                candidate = (parent / relative).resolve()
                return candidate if candidate.is_relative_to(parent.resolve()) else None
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    candidate = manifest_path.parent / path
    return candidate.resolve() if candidate.is_file() else path.resolve()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _metric_record(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    metrics = payload.get("metrics") or {}
    return {
        "persona_idx": payload.get("persona_idx"),
        "persona_name": payload.get("persona_name"),
        "context_idx": payload.get("context_idx"),
        "recipient": payload.get("recipient"),
        "task": payload.get("task"),
        "context_discarded": bool(payload.get("context_discarded")),
        **{metric: metrics.get(metric) for metric in METRICS},
    }


def _pipeline_metrics(entry: dict[str, Any]) -> dict[int, dict[str, Any]]:
    manifest_path = Path(entry["manifest_path"])
    manifest = _read_json(manifest_path)
    result: dict[int, dict[str, Any]] = {}
    for context in manifest.get("contexts") or []:
        if not isinstance(context, dict):
            continue
        context_idx = context.get("context_idx")
        metric_path = _resolve(context.get("privacy_metrics_cimemories_json"), manifest_path)
        if isinstance(context_idx, int) and metric_path is not None and metric_path.is_file():
            result[context_idx] = _metric_record(metric_path)
    return result


def _pre_metrics(entry: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw = entry.get("pre_manifest_path")
    if not isinstance(raw, Path) or not raw.is_file():
        return {}
    manifest = _read_json(raw)
    result: dict[int, dict[str, Any]] = {}
    for value in (manifest.get("artifacts") or {}).get("privacy_metrics_cimemories_json") or []:
        path = _resolve(value, raw)
        if path is None or not path.is_file():
            continue
        record = _metric_record(path)
        context_idx = record.get("context_idx")
        if isinstance(context_idx, int):
            result[context_idx] = record
    return result


def _select_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(
            (entry["model"], entry["architecture"], entry["persona_idx"]), []
        ).append(entry)
    selected: list[dict[str, Any]] = []
    for alternatives in grouped.values():
        alternatives.sort(
            key=lambda item: (
                item["post_complete"], item["post_records"],
                item["pre_complete"], item["pre_records"], item["created_at"],
            ),
            reverse=True,
        )
        selected.append(alternatives[0])
    return selected


def _locked_entries(inputs: Path) -> list[dict[str, Any]]:
    payload = _read_json(inputs)
    entries: list[dict[str, Any]] = []
    for item in payload.get("selected_runs") or []:
        if not isinstance(item, dict):
            continue
        pipeline_manifest = _resolve(item.get("pipeline_manifest"), inputs)
        if pipeline_manifest is None:
            raise FileNotFoundError(
                f"Locked pipeline manifest cannot be resolved: {item.get('pipeline_manifest')}"
            )
        if not pipeline_manifest.is_file():
            raise FileNotFoundError(f"Locked pipeline manifest is missing: {pipeline_manifest}")
        pre_value = item.get("pre_manifest")
        pre_manifest = _resolve(pre_value, inputs) if pre_value else None
        entries.append(
            {
                **item,
                "manifest_path": pipeline_manifest,
                "pre_manifest_path": pre_manifest,
                "post_complete": True,
                "pre_complete": bool(pre_manifest and pre_manifest.is_file()),
                "post_records": int(item.get("post_records") or 0),
                "pre_records": int(item.get("pre_records") or 0),
                "post_generation_mean_ms": float(item.get("post_generation_seconds") or 0) * 1000,
                "pre_generation_mean_ms": float(item.get("pre_generation_seconds") or 0) * 1000,
                "post_cost_usd": float(item.get("estimated_cost_usd") or 0),
                "pre_cost_usd": 0.0,
                "post_cost_incomplete": bool(item.get("cost_incomplete")),
                "pre_cost_incomplete": False,
            }
        )
    return entries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _hierarchical_ci(rows: list[dict[str, Any]], metric: str, iterations: int) -> tuple[float, float]:
    by_persona: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if _finite(row.get(f"delta_{metric}")):
            by_persona.setdefault(int(row["persona_idx"]), []).append(row)
    persona_ids = sorted(by_persona)
    if not persona_ids:
        return float("nan"), float("nan")
    rng = random.Random(20260923 + list(METRICS).index(metric))
    estimates: list[float] = []
    for _ in range(iterations):
        values: list[float] = []
        for persona_idx in rng.choices(persona_ids, k=len(persona_ids)):
            contexts = by_persona[persona_idx]
            values.extend(
                float(row[f"delta_{metric}"])
                for row in rng.choices(contexts, k=len(contexts))
            )
        estimates.append(mean(values))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _summary_rows(rows: list[dict[str, Any]], iterations: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups = sorted({(row["model"], row["architecture"]) for row in rows})
    for model, architecture in groups:
        subset = [
            row for row in rows
            if row["model"] == model and row["architecture"] == architecture
        ]
        for metric in METRICS:
            pre = [float(row[f"pre_{metric}"]) for row in subset if _finite(row.get(f"pre_{metric}"))]
            post = [float(row[f"post_{metric}"]) for row in subset if _finite(row.get(f"post_{metric}"))]
            delta = [float(row[f"delta_{metric}"]) for row in subset if _finite(row.get(f"delta_{metric}"))]
            low, high = _hierarchical_ci(subset, metric, iterations)
            result.append(
                {
                    "model": model,
                    "architecture": architecture,
                    "metric": metric,
                    "persona_count": len({row["persona_idx"] for row in subset}),
                    "context_pairs": len(delta),
                    "pre_mean": mean(pre),
                    "post_mean": mean(post),
                    "mean_delta": mean(delta),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return result


def _matched_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = sorted({row["model"] for row in rows})
    indexed = {
        (row["model"], row["architecture"], row["persona_idx"], row["context_idx"]): row
        for row in rows
    }
    result: list[dict[str, Any]] = []
    for left_index, left_model in enumerate(models):
        for right_model in models[left_index + 1 :]:
            for architecture in ARCHITECTURES:
                left_keys = {
                    (key[2], key[3]) for key in indexed
                    if key[0] == left_model and key[1] == architecture
                }
                right_keys = {
                    (key[2], key[3]) for key in indexed
                    if key[0] == right_model and key[1] == architecture
                }
                for persona_idx, context_idx in sorted(left_keys & right_keys):
                    left = indexed[(left_model, architecture, persona_idx, context_idx)]
                    right = indexed[(right_model, architecture, persona_idx, context_idx)]
                    base = {
                        "left_model": left_model,
                        "right_model": right_model,
                        "architecture": architecture,
                        "persona_idx": persona_idx,
                        "persona_name": left["persona_name"],
                        "context_idx": context_idx,
                    }
                    for stage in ("pre", "post"):
                        for metric in METRICS:
                            lv, rv = left.get(f"{stage}_{metric}"), right.get(f"{stage}_{metric}")
                            base[f"{stage}_left_{metric}"] = lv
                            base[f"{stage}_right_{metric}"] = rv
                            base[f"{stage}_right_minus_left_{metric}"] = (
                                float(rv) - float(lv) if _finite(lv) and _finite(rv) else None
                            )
                    result.append(base)
    return result


def _matched_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups = sorted({(r["left_model"], r["right_model"], r["architecture"]) for r in rows})
    for left, right, architecture in groups:
        subset = [r for r in rows if (r["left_model"], r["right_model"], r["architecture"]) == (left, right, architecture)]
        for stage in ("pre", "post"):
            for metric in METRICS:
                field = f"{stage}_right_minus_left_{metric}"
                values = [float(r[field]) for r in subset if _finite(r.get(field))]
                result.append(
                    {
                        "left_model": left,
                        "right_model": right,
                        "architecture": architecture,
                        "stage": stage,
                        "metric": metric,
                        "persona_count": len({r["persona_idx"] for r in subset}),
                        "context_pairs": len(values),
                        "right_minus_left_mean": mean(values) if values else None,
                    }
                )
    return result


def _cohort_robustness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Contrast each model's full cohort with personas shared by every model."""
    models = sorted({row["model"] for row in rows})
    result: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        persona_sets = [
            {row["persona_idx"] for row in rows if row["model"] == model and row["architecture"] == architecture}
            for model in models
        ]
        common = set.intersection(*persona_sets) if persona_sets and all(persona_sets) else set()
        for model in models:
            full = [row for row in rows if row["model"] == model and row["architecture"] == architecture]
            matched = [row for row in full if row["persona_idx"] in common]
            for metric in METRICS:
                full_values = [float(row[f"delta_{metric}"]) for row in full if _finite(row.get(f"delta_{metric}"))]
                matched_values = [float(row[f"delta_{metric}"]) for row in matched if _finite(row.get(f"delta_{metric}"))]
                full_mean = mean(full_values) if full_values else None
                matched_mean = mean(matched_values) if matched_values else None
                result.append({
                    "model": model,
                    "architecture": architecture,
                    "metric": metric,
                    "full_personas": len({row["persona_idx"] for row in full}),
                    "matched_personas": len(common),
                    "full_mean_delta": full_mean,
                    "matched_mean_delta": matched_mean,
                    "matched_minus_full": (
                        matched_mean - full_mean
                        if _finite(full_mean) and _finite(matched_mean) else None
                    ),
                })
    return result


def _fmt(metric: str, value: Any, signed: bool = False) -> str:
    if not _finite(value):
        return "—"
    if metric == "average_exposed_total":
        return f"{value:+.2f}" if signed else f"{value:.2f}"
    return f"{100 * value:+.1f} pp" if signed else f"{100 * value:.1f}%"


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _html_report(
    inventory: list[dict[str, Any]], summaries: list[dict[str, Any]],
    matched: list[dict[str, Any]], robustness: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    created_at: str, iterations: int,
) -> str:
    models = sorted({row["model"] for row in inventory})
    persona_counts = {
        model: len({row["persona_idx"] for row in inventory if row["model"] == model})
        for model in models
    }
    matched_persona_count = min(persona_counts.values()) if persona_counts else 0
    cohort_description = ", ".join(
        f"{model}: {persona_counts[model]} personas" for model in models
    )
    inventory_html = "".join(
        f"<tr><td>{_esc(row['model'])}</td><td><span class='tag {row['architecture']}'>{row['architecture']}</span></td>"
        f"<td>{row['persona_idx']} · {_esc(row['persona_name'])}</td><td>{row['post_records']}</td>"
        f"<td>{row['pre_records']}</td><td>{row['post_repeats']}</td><td>{row['pre_repeats']}</td>"
        f"<td>{row['post_generation_seconds']}</td><td>{row['pre_generation_seconds']}</td>"
        f"<td>${row['estimated_cost_usd']:.3f}{'*' if row['cost_incomplete'] else ''}</td></tr>"
        for row in inventory
    )
    summary_html = "".join(
        f"<tr><td>{_esc(row['model'])}</td><td><span class='tag {row['architecture']}'>{row['architecture']}</span></td>"
        f"<td>{METRICS[row['metric']]}</td><td>{row['persona_count']}</td>"
        f"<td>{_fmt(row['metric'], row['pre_mean'])}</td><td>{_fmt(row['metric'], row['post_mean'])}</td>"
        f"<td class='delta'>{_fmt(row['metric'], row['mean_delta'], True)}</td>"
        f"<td>{_fmt(row['metric'], row['ci95_low'], True)} to {_fmt(row['metric'], row['ci95_high'], True)}</td></tr>"
        for row in summaries
    )
    matched_html = "".join(
        f"<tr><td>{_esc(row['left_model'])}<br><span class='quiet'>vs</span><br>{_esc(row['right_model'])}</td>"
        f"<td><span class='tag {row['architecture']}'>{row['architecture']}</span></td><td>{row['stage']}</td>"
        f"<td>{METRICS[row['metric']]}</td><td>{row['persona_count']}</td><td>{row['context_pairs']}</td>"
        f"<td class='delta'>{_fmt(row['metric'], row['right_minus_left_mean'], True)}</td></tr>"
        for row in matched
    )
    robustness_html = "".join(
        f"<tr><td>{_esc(row['model'])}</td><td><span class='tag {row['architecture']}'>{row['architecture']}</span></td>"
        f"<td>{METRICS[row['metric']]}</td><td>{row['full_personas']}</td><td>{row['matched_personas']}</td>"
        f"<td>{_fmt(row['metric'], row['full_mean_delta'], True)}</td>"
        f"<td>{_fmt(row['metric'], row['matched_mean_delta'], True)}</td>"
        f"<td class='delta'>{_fmt(row['metric'], row['matched_minus_full'], True)}</td></tr>"
        for row in robustness
    )
    model_cards = "".join(
        f"<div class='card'><b>{_esc(model)}</b><span>{sum(1 for r in inventory if r['model']==model)} complete persona-architecture cells</span></div>"
        for model in models
    )
    context_html = "".join(
        f"<tr data-search='{_esc((str(r['model'])+' '+r['architecture']+' '+str(r['persona_name'])+' '+str(r['recipient'])+' '+str(r['task'])).lower())}'>"
        f"<td>{_esc(r['model'])}</td><td><span class='tag {r['architecture']}'>{r['architecture']}</span></td>"
        f"<td>{r['persona_idx']} · {_esc(r['persona_name'])}</td><td>{r['context_idx']}</td><td>{_esc(r['recipient'])}<br><span class='quiet'>{_esc(r['task'])}</span></td>"
        f"<td>{_fmt('necessary_recall',r['pre_necessary_recall'])}</td><td>{_fmt('necessary_recall',r['post_necessary_recall'])}</td>"
        f"<td>{_fmt('inappropriate_leak_rate',r['pre_inappropriate_leak_rate'])}</td><td>{_fmt('inappropriate_leak_rate',r['post_inappropriate_leak_rate'])}</td></tr>"
        for r in context_rows
    )
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CIMemories comparative report</title><style>
    :root{{--ink:#172033;--muted:#667085;--line:#dbe2ea;--paper:#f3f6fa;--navy:#17365d;--blue:#4568dc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}header{{padding:55px max(22px,calc((100vw - 1300px)/2));color:white;background:linear-gradient(125deg,#102a4c,#214d7d,#087e8b)}}header h1{{font-size:clamp(32px,5vw,54px);margin:0}}header p{{max-width:900px;color:#dceafa}}nav{{position:sticky;top:0;z-index:10;display:flex;gap:20px;padding:12px max(20px,calc((100vw - 1300px)/2));overflow:auto;background:#fffffff2;border-bottom:1px solid var(--line)}}nav a{{font-weight:750;color:var(--navy);text-decoration:none;white-space:nowrap}}main{{max-width:1300px;margin:auto;padding:24px 18px 80px}}section{{margin:35px 0 55px;scroll-margin-top:60px}}h2{{font-size:28px;margin-bottom:5px}}.quiet,.lead{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin-top:-48px;position:relative}}.card,.panel,.notice{{background:white;border:1px solid var(--line);border-radius:15px;box-shadow:0 8px 25px #17365d12}}.card{{padding:20px}}.card b{{display:block;font-size:22px;color:var(--navy)}}.panel{{overflow:auto;max-height:72vh}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 12px;border-bottom:1px solid #e8ecf1;text-align:left;white-space:nowrap;vertical-align:top}}th{{position:sticky;top:0;background:var(--navy);color:white;font-size:12px}}.tag{{display:inline-block;padding:3px 8px;border-radius:999px;color:white;font-weight:800}}.tag.list{{background:{ARCH_COLORS['list']}}}.tag.graph{{background:{ARCH_COLORS['graph']}}}.tag.profile{{background:{ARCH_COLORS['profile']}}}.delta{{font-weight:800}}.notice{{padding:16px;border-left:5px solid #c98112}}input{{width:min(100%,430px);padding:10px;border:1px solid #b8c3d0;border-radius:8px;margin:8px 0 14px}}@media print{{nav,input{{display:none}}.panel{{max-height:none}}th{{position:static}}}}
    </style></head><body><header><h1>CIMemories comparative report</h1><p>One red thread across generation models, memory architectures, and pre/post-reranking stages. Generated from locked saved artifacts only; no model calls.</p></header><nav><a href='#scope'>Scope</a><a href='#effects'>Reranking effects</a><a href='#matched'>Matched models</a><a href='#robustness'>Cohorts</a><a href='#efficiency'>Efficiency</a><a href='#contexts'>Contexts</a><a href='#methods'>Methods</a></nav><main>
    <section id='scope'><div class='cards'>{model_cards}<div class='card'><b>{len(context_rows):,}</b><span>paired model/architecture/persona/context observations</span></div><div class='card'><b>{created_at[:10]}</b><span>report snapshot</span></div></div><h2>Experiment scope</h2><p class='lead'>Coverage and provenance are explicit because the model cohorts have different persona and repetition counts.</p><div class='panel'><table><thead><tr><th>Model</th><th>Memory</th><th>Persona</th><th>Post responses</th><th>Pre responses</th><th>Post reps</th><th>Pre reps</th><th>Post sec</th><th>Pre sec</th><th>Cost</th></tr></thead><tbody>{inventory_html}</tbody></table></div></section>
    <section id='effects'><h2>Within-model reranking effects</h2><p class='lead'>Post minus pre is paired within persona and context. Positive values mean more disclosure after reranking.</p><div class='panel'><table><thead><tr><th>Model</th><th>Memory</th><th>Metric</th><th>Personas</th><th>Pre</th><th>Post</th><th>Post − pre</th><th>Hierarchical 95% interval</th></tr></thead><tbody>{summary_html}</tbody></table></div></section>
    <section id='matched'><h2>Cross-model matched subset</h2><div class='notice'><strong>Matched comparison.</strong> Comparisons use the {matched_persona_count}-persona context set shared by every model. Cohort coverage: {_esc(cohort_description)}.</div><div class='panel'><table><thead><tr><th>Models</th><th>Memory</th><th>Stage</th><th>Metric</th><th>Personas</th><th>Pairs</th><th>Right − left</th></tr></thead><tbody>{matched_html}</tbody></table></div></section>
    <section id='robustness'><h2>Full versus matched cohort</h2><p class='lead'>This checks whether conclusions from a model's complete persona cohort survive when restricted to the personas shared by every model.</p><div class='panel'><table><thead><tr><th>Model</th><th>Memory</th><th>Metric</th><th>Full N</th><th>Matched N</th><th>Full post − pre</th><th>Matched post − pre</th><th>Matched − full</th></tr></thead><tbody>{robustness_html}</tbody></table></div></section>
    <section id='efficiency'><h2>Efficiency and cost</h2><p class='lead'>The inventory reports measured mean generation latency and recorded-token cost. An asterisk marks incomplete backend or embedding telemetry.</p><div class='panel'><table><thead><tr><th>Model</th><th>Memory</th><th>Persona</th><th>Post sec</th><th>Pre sec</th><th>Estimated cost</th></tr></thead><tbody>{''.join(f"<tr><td>{_esc(r['model'])}</td><td>{r['architecture']}</td><td>{r['persona_idx']}</td><td>{r['post_generation_seconds']}</td><td>{r['pre_generation_seconds']}</td><td>${r['estimated_cost_usd']:.3f}{'*' if r['cost_incomplete'] else ''}</td></tr>" for r in inventory)}</tbody></table></div></section>
    <section id='contexts'><h2>Context browser</h2><p class='lead'>Use this for heterogeneity checks and drill-down selection. Exact source paths remain in the lockfile and CSV exports.</p><input id='search' placeholder='Search model, architecture, persona, recipient, or task'><div class='panel'><table id='context-table'><thead><tr><th>Model</th><th>Memory</th><th>Persona</th><th>Context</th><th>Scenario</th><th>Pre necessary</th><th>Post necessary</th><th>Pre leakage</th><th>Post leakage</th></tr></thead><tbody>{context_html}</tbody></table></div></section>
    <section id='methods'><h2>Methods and limitations</h2><p>Within-model intervals use a hierarchical bootstrap over personas and contexts ({iterations:,} iterations). Contexts are not treated as independent human subjects. Cross-model contrasts use the {matched_persona_count}-persona cohort shared by every model. Post-rerank repetition counts differ across model cohorts; therefore mean effects are comparable, but response-level variance precision is asymmetric.</p><p>Generation model, exposure judge, context-label source, and memory-internal models are separate roles. See <code>comparative_report_inputs.json</code> and <code>summary.json</code> for exact provenance.</p></section>
    </main><script>const q=document.getElementById('search'),rs=[...document.querySelectorAll('#context-table tbody tr')];q.addEventListener('input',()=>{{const v=q.value.toLowerCase();rs.forEach(r=>r.hidden=!r.dataset.search.includes(v))}});</script></body></html>"""


def build_comparative_report(
    *, output_root: Path = Path("research_outputs"), dataset_filter: str | None = "cimemories_raw",
    output: Path | None = None, inputs: Path | None = None, bootstrap_iterations: int = 5000,
) -> Path:
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100")
    if inputs is not None:
        selected = _locked_entries(inputs)
    else:
        # Lazy import avoids making the CLI/report modules cyclic at import time.
        from .cli import _cimemories_progress_entries

        selected = _select_entries(
            _cimemories_progress_entries(
                output_root=output_root, dataset_filter=dataset_filter, model_filter=None
            )
        )
    selected = [entry for entry in selected if entry.get("post_complete") and entry.get("pre_complete")]
    if not selected:
        raise ValueError("No complete pre/post model-architecture-persona cells were found")
    created_at = datetime.now().astimezone().isoformat()
    if output is None:
        output = output_root / f"cimemories-comparative-report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)

    inventory: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    lock_runs: list[dict[str, Any]] = []
    for entry in sorted(selected, key=lambda item: (item["model"], item["architecture"], item["persona_idx"])):
        post, pre = _pipeline_metrics(entry), _pre_metrics(entry)
        shared = sorted(set(post) & set(pre))
        if len(shared) != 49:
            raise ValueError(
                f"Expected 49 paired contexts for {entry['model']} {entry['architecture']} "
                f"persona {entry['persona_idx']}; found {len(shared)}"
            )
        pipeline_manifest = _read_json(Path(entry["manifest_path"]))
        components = pipeline_manifest.get("component_usage_ledger") or []
        judge_model = next(
            (component.get("model") for component in components
             if isinstance(component, dict) and component.get("component") == "exposure_judging"),
            pipeline_manifest.get("judge_model"),
        )
        inventory_row = {
            "model": entry["model"], "architecture": entry["architecture"],
            "persona_idx": entry["persona_idx"], "persona_name": entry["persona_name"],
            "post_records": entry.get("post_records", 0), "pre_records": entry.get("pre_records", 0),
            "post_repeats": entry.get("post_repeats", "-"), "pre_repeats": entry.get("pre_repeats", "-"),
            "post_generation_seconds": round(float(entry.get("post_generation_mean_ms") or 0) / 1000, 3),
            "pre_generation_seconds": round(float(entry.get("pre_generation_mean_ms") or 0) / 1000, 3),
            "estimated_cost_usd": float(entry.get("post_cost_usd") or 0) + float(entry.get("pre_cost_usd") or 0),
            "cost_incomplete": bool(entry.get("post_cost_incomplete") or entry.get("pre_cost_incomplete")),
            "exposure_judge_model": judge_model,
            "context_label_model": pipeline_manifest.get("context_label_model"),
            "context_labels_file": pipeline_manifest.get("context_labels_file"),
            "agent_reasoning_effort": pipeline_manifest.get("agent_reasoning_effort"),
            "pipeline_manifest": str(Path(entry["manifest_path"]).resolve()),
            "pre_manifest": str(Path(entry["pre_manifest_path"]).resolve()),
        }
        inventory.append(inventory_row)
        lock_runs.append(dict(inventory_row))
        for context_idx in shared:
            base = {
                "model": entry["model"], "architecture": entry["architecture"],
                "persona_idx": entry["persona_idx"], "persona_name": post[context_idx].get("persona_name") or entry["persona_name"],
                "context_idx": context_idx, "recipient": post[context_idx].get("recipient"),
                "task": post[context_idx].get("task"),
                "context_discarded": post[context_idx].get("context_discarded"),
            }
            for metric in METRICS:
                pv, qv = pre[context_idx].get(metric), post[context_idx].get(metric)
                base[f"pre_{metric}"] = pv
                base[f"post_{metric}"] = qv
                base[f"delta_{metric}"] = float(qv) - float(pv) if _finite(pv) and _finite(qv) else None
            paired.append(base)

    summaries = _summary_rows(paired, bootstrap_iterations)
    matched_contexts = _matched_rows(paired)
    matched_summaries = _matched_summary(matched_contexts)
    robustness = _cohort_robustness(paired)
    architecture_rows: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in paired}):
        for stage in ("pre", "post"):
            for architecture in ARCHITECTURES:
                subset = [r for r in paired if r["model"] == model and r["architecture"] == architecture]
                for metric in METRICS:
                    values = [float(r[f"{stage}_{metric}"]) for r in subset if _finite(r.get(f"{stage}_{metric}"))]
                    architecture_rows.append({"model": model, "stage": stage, "architecture": architecture, "metric": metric, "persona_count": len({r['persona_idx'] for r in subset}), "mean": mean(values) if values else None})

    lock = {
        "schema_version": 1, "created_at": created_at, "dataset_filter": dataset_filter,
        "selection_rule": "strongest complete post scope, then pre scope, then newest creation time",
        "selected_runs": lock_runs,
    }
    summary = {
        "schema_version": 1, "created_at": created_at,
        "bootstrap_iterations": bootstrap_iterations,
        "models": sorted({row["model"] for row in inventory}),
        "inventory": inventory, "within_model_pre_post": summaries,
        "matched_model_comparison": matched_summaries,
        "cohort_robustness": robustness,
        "limitations": [
            "Cross-model inference uses only persona/context cells shared by every model.",
            "Post-rerank repetition counts differ across model cohorts; response-level variance precision is asymmetric.",
            "Costs marked incomplete exclude backend operations without token telemetry.",
        ],
    }
    (output / "comparative_report_inputs.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(output / "experiment_inventory.csv", inventory)
    _write_csv(output / "paired_context_metrics.csv", paired)
    _write_csv(output / "matched_model_comparison.csv", matched_contexts)
    _write_csv(output / "cohort_robustness.csv", robustness)
    _write_csv(output / "architecture_effects.csv", architecture_rows)
    _write_csv(output / "efficiency_costs.csv", inventory)
    (output / "REPORT.html").write_text(
        _html_report(inventory, summaries, matched_summaries, robustness, paired, created_at, bootstrap_iterations),
        encoding="utf-8",
    )
    return output
