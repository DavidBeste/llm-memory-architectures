#!/usr/bin/env python3
"""Build an offline, paper-oriented pre/post-reranking analysis dashboard.

The script reads existing CIMemories artifacts only. It performs no network or
model calls and requires no third-party Python packages.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_RUNS = {
    "list": Path("research_outputs/privacy-pipeline-cimemories-dataset_cimemories-raw_mem3_20260912_054751"),
    "graph": Path("research_outputs/privacy-pipeline-cimemories-dataset_cimemories-raw_mem6_20260912_092917"),
    "profile": Path("research_outputs/privacy-pipeline-cimemories-dataset_cimemories-raw_mem19_20260912_143132"),
}
DEFAULT_OUTPUT = Path("research_outputs/cimemories_reranking_analysis_3persona")
METRICS = {
    "necessary_recall": "Necessary recall",
    "inappropriate_leak_rate": "Inappropriate leakage",
    "ambiguous_exposure_rate": "Ambiguous exposure",
    "average_exposed_total": "Mean exposed attributes",
}
ARCH_COLORS = {"list": "#e76f51", "graph": "#2a9d8f", "profile": "#4568dc"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def display(metric: str, value: Any, signed: bool = False) -> str:
    if not finite(value):
        return "—"
    if metric == "average_exposed_total":
        return f"{value:+.2f}" if signed else f"{value:.2f}"
    return f"{100 * value:+.2f} pp" if signed else f"{100 * value:.2f}%"


def metric_payload(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    metrics = payload.get("metrics") or {}
    return {
        "persona_idx": payload.get("persona_idx"),
        "persona_name": payload.get("persona_name"),
        "context_idx": payload.get("context_idx"),
        "recipient": payload.get("recipient"),
        "task": payload.get("task"),
        "context_discarded": bool(payload.get("context_discarded")),
        **{key: metrics.get(key) for key in METRICS},
    }


def load_pairs(architecture: str, root: Path, personas: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for persona_idx in range(personas):
        pre_dir = root / "pre_rerank_response_evaluation/personas" / f"persona_{persona_idx:03d}" / "privacy_metrics"
        post_dir = root / "personas" / f"persona_{persona_idx:03d}" / "privacy_metrics"
        pre = {
            item["context_idx"]: item
            for item in (metric_payload(path) for path in pre_dir.glob("context_*/privacy_metrics_cimemories.json"))
        }
        post = {
            item["context_idx"]: item
            for item in (metric_payload(path) for path in post_dir.glob("context_*/privacy_metrics_cimemories.json"))
        }
        if set(pre) != set(range(49)) or set(post) != set(range(49)):
            raise ValueError(f"{architecture} persona {persona_idx} is not complete for all 49 contexts")
        for context_idx in range(49):
            row = {
                "architecture": architecture,
                **{k: pre[context_idx][k] for k in ("persona_idx", "persona_name", "context_idx", "recipient", "task", "context_discarded")},
            }
            for metric in METRICS:
                row[f"pre_{metric}"] = pre[context_idx][metric]
                row[f"post_{metric}"] = post[context_idx][metric]
                row[f"delta_{metric}"] = (
                    post[context_idx][metric] - pre[context_idx][metric]
                    if finite(pre[context_idx][metric]) and finite(post[context_idx][metric])
                    else None
                )
            pairs.append(row)
    return pairs


def complete_persona_count(root: Path) -> int:
    """Return the contiguous number of complete 49-context pre/post persona pairs."""
    count = 0
    while True:
        persona = f"persona_{count:03d}"
        pre_dir = root / "pre_rerank_response_evaluation/personas" / persona / "privacy_metrics"
        post_dir = root / "personas" / persona / "privacy_metrics"
        pre = {path.parent.name for path in pre_dir.glob("context_*/privacy_metrics_cimemories.json")}
        post = {path.parent.name for path in post_dir.glob("context_*/privacy_metrics_cimemories.json")}
        expected = {f"context_{idx:03d}" for idx in range(49)}
        if pre != expected or post != expected:
            return count
        count += 1


def hierarchical_bootstrap(
    rows: list[dict[str, Any]], metric: str, *, iterations: int, seed: int
) -> tuple[float, float]:
    by_persona: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if finite(row[f"delta_{metric}"]):
            by_persona[int(row["persona_idx"])].append(row)
    persona_ids = sorted(by_persona)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled: list[float] = []
        for persona_idx in rng.choices(persona_ids, k=len(persona_ids)):
            persona_rows = by_persona[persona_idx]
            sampled.extend(
                row[f"delta_{metric}"]
                for row in rng.choices(persona_rows, k=len(persona_rows))
            )
        estimates.append(mean(sampled))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def summarize(rows: list[dict[str, Any]], iterations: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"context_pairs": len(rows), "metrics": {}}
    for index, metric in enumerate(METRICS):
        pre = [row[f"pre_{metric}"] for row in rows if finite(row[f"pre_{metric}"])]
        post = [row[f"post_{metric}"] for row in rows if finite(row[f"post_{metric}"])]
        deltas = [row[f"delta_{metric}"] for row in rows if finite(row[f"delta_{metric}"])]
        higher = sum(value > 1e-12 for value in deltas)
        lower = sum(value < -1e-12 for value in deltas)
        ci_low, ci_high = hierarchical_bootstrap(rows, metric, iterations=iterations, seed=20260920 + index)
        summary["metrics"][metric] = {
            "pre_mean": mean(pre),
            "post_mean": mean(post),
            "mean_delta": mean(deltas),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "post_higher": higher,
            "pre_higher": lower,
            "tied": len(deltas) - higher - lower,
            "valid_pairs": len(deltas),
        }
    return summary


def write_pairs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "architecture", "persona_idx", "persona_name", "context_idx", "recipient", "task", "context_discarded"
    ]
    for metric in METRICS:
        fields.extend((f"pre_{metric}", f"post_{metric}", f"delta_{metric}"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def exposure_html(exposed: Any) -> str:
    if not isinstance(exposed, dict) or not exposed:
        return '<p class="quiet">No exposed attributes identified.</p>'
    return "<ul>" + "".join(
        f"<li><strong>{esc(attribute)}</strong><blockquote>{esc(evidence)}</blockquote></li>"
        for attribute, evidence in exposed.items()
    ) + "</ul>"


def audit_sample(
    runs: dict[str, Path], persona_counts: dict[str, int], per_persona: int, output: Path
) -> tuple[list[dict[str, Any]], str]:
    rng = random.Random(24092026)
    audit_rows: list[dict[str, Any]] = []
    cards: list[str] = []
    for architecture, root in runs.items():
        for persona_idx in range(persona_counts[architecture]):
            contexts = rng.sample(range(49), per_persona)
            persona = f"persona_{persona_idx:03d}"
            pre_dir = root / "pre_rerank_response_evaluation/personas" / persona
            pre_responses = {(x["context_idx"], x["repeat_idx"]): x for x in read_jsonl(pre_dir / "responses.jsonl")}
            pre_exposed = {(x["context_idx"], x["repeat_idx"]): x for x in read_jsonl(pre_dir / "exposed_attributes.jsonl")}
            post_dir = root / "personas" / persona / "query"
            post_responses = {(x["context_idx"], x["repeat_idx"]): x for x in read_jsonl(post_dir / "responses.jsonl")}
            post_files = sorted(post_dir.glob("exposed_attributes_*.jsonl"))
            post_exposed = {(x["context_idx"], x["repeat_idx"]): x for x in read_jsonl(post_files[-1])}
            for context_idx in contexts:
                key = (context_idx, 0)
                pre = pre_responses[key]
                post = post_responses[key]
                pre_judge = pre_exposed[key]
                post_judge = post_exposed[key]
                sample_id = f"{architecture}-p{persona_idx:03d}-c{context_idx:03d}"
                audit_rows.append(
                    {
                        "sample_id": sample_id,
                        "architecture": architecture,
                        "persona_idx": persona_idx,
                        "persona_name": pre.get("persona_name"),
                        "context_idx": context_idx,
                        "recipient": pre.get("recipient"),
                        "task": pre.get("task"),
                        "pre_response": pre.get("assistant_response"),
                        "pre_exposed_attributes": json.dumps(pre_judge.get("exposed_attributes") or {}, ensure_ascii=False),
                        "post_response": post.get("assistant_response"),
                        "post_exposed_attributes": json.dumps(post_judge.get("exposed_attributes") or {}, ensure_ascii=False),
                        "human_pre_correct": "",
                        "human_post_correct": "",
                        "human_notes": "",
                    }
                )
                cards.append(
                    f"""
                    <details class="audit" data-search="{esc((sample_id+' '+str(pre.get('recipient'))+' '+str(pre.get('task'))).lower())}">
                      <summary><span class="tag {architecture}">{architecture}</span><strong>{esc(pre.get('persona_name'))}</strong> · context {context_idx:02d} · {esc(pre.get('recipient'))}</summary>
                      <p class="task">{esc(pre.get('task'))}</p>
                      <div class="audit-grid">
                        <article><h4>Pre-rerank response</h4><pre>{esc(pre.get('assistant_response'))}</pre><h5>Judge labels and evidence</h5>{exposure_html(pre_judge.get('exposed_attributes'))}</article>
                        <article><h4>Post-rerank response</h4><pre>{esc(post.get('assistant_response'))}</pre><h5>Judge labels and evidence</h5>{exposure_html(post_judge.get('exposed_attributes'))}</article>
                      </div>
                      <div class="audit-form"><label>Pre correct? <select data-audit="{sample_id}-pre"><option></option><option>yes</option><option>no</option><option>uncertain</option></select></label><label>Post correct? <select data-audit="{sample_id}-post"><option></option><option>yes</option><option>no</option><option>uncertain</option></select></label><label>Notes <input data-audit="{sample_id}-notes" placeholder="Stored in this browser"></label></div>
                    </details>"""
                )
    fields = list(audit_rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    return audit_rows, "".join(cards)


def bar_svg(summaries: dict[str, Any], metric: str) -> str:
    width, height = 760, 275
    values = [summaries[a]["metrics"][metric][c] for a in summaries for c in ("pre_mean", "post_mean")]
    upper = max(values) * 1.18 or 1
    bars: list[str] = []
    labels: list[str] = []
    x = 75
    for architecture in summaries:
        for condition, opacity in (("pre_mean", 0.48), ("post_mean", 1.0)):
            value = summaries[architecture]["metrics"][metric][condition]
            h = 180 * value / upper
            y = 215 - h
            bars.append(f'<rect x="{x}" y="{y:.1f}" width="62" height="{h:.1f}" rx="5" fill="{ARCH_COLORS[architecture]}" opacity="{opacity}"/>')
            bars.append(f'<text x="{x+31}" y="{y-7:.1f}" text-anchor="middle" font-size="12">{esc(display(metric,value))}</text>')
            labels.append(f'<text x="{x+31}" y="238" text-anchor="middle" font-size="11">{"pre" if condition=="pre_mean" else "post"}</text>')
            x += 70
        labels.append(f'<text x="{x-70}" y="258" text-anchor="middle" font-size="13" font-weight="700">{architecture}</text>')
        x += 38
    return f'<svg viewBox="0 0 {width} {height}" role="img"><line x1="55" y1="215" x2="735" y2="215" stroke="#aab4c3"/>{"".join(bars+labels)}</svg>'


def scatter_svg(rows: list[dict[str, Any]]) -> str:
    points=[]
    for row in rows:
        x=row.get("delta_necessary_recall"); y=row.get("delta_average_exposed_total")
        if finite(x) and finite(y): points.append((row["architecture"],x,y))
    width,height=760,400; left,right,top,bottom=70,730,25,350
    xmax=max(abs(x) for _,x,_ in points) or 1; ymax=max(abs(y) for _,_,y in points) or 1
    def sx(x:float)->float: return (left+right)/2+x/xmax*(right-left)*.46
    def sy(y:float)->float: return (top+bottom)/2-y/ymax*(bottom-top)*.46
    circles="".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{ARCH_COLORS[a]}" opacity=".43"><title>{a}: necessary {x:+.3f}, exposure {y:+.2f}</title></circle>' for a,x,y in points)
    legend="".join(f'<circle cx="{110+i*120}" cy="382" r="5" fill="{ARCH_COLORS[a]}"/><text x="{120+i*120}" y="386" font-size="12">{a}</text>' for i,a in enumerate(ARCH_COLORS))
    return f'<svg viewBox="0 0 {width} {height}" role="img"><line x1="{sx(0)}" y1="{top}" x2="{sx(0)}" y2="{bottom}" stroke="#98a4b5"/><line x1="{left}" y1="{sy(0)}" x2="{right}" y2="{sy(0)}" stroke="#98a4b5"/>{circles}<text x="400" y="15" text-anchor="middle" font-size="12">More necessary disclosure →</text><text x="15" y="190" transform="rotate(-90 15 190)" text-anchor="middle" font-size="12">More total disclosure →</text>{legend}</svg>'


def report_html(
    rows: list[dict[str, Any]], summaries: dict[str, Any], sensitivity: dict[str, Any],
    audit_cards: str, audit_count: int, iterations: int, persona_counts: dict[str, int]
) -> str:
    summary_rows=[]
    for architecture in summaries:
        for metric,label in METRICS.items():
            s=summaries[architecture]["metrics"][metric]
            cls="up" if s["mean_delta"]>0 else "down"
            summary_rows.append(f'<tr><td><span class="tag {architecture}">{architecture}</span></td><td>{label}</td><td>{display(metric,s["pre_mean"])}</td><td>{display(metric,s["post_mean"])}</td><td class="{cls}">{display(metric,s["mean_delta"],True)}</td><td>{display(metric,s["ci95_low"],True)} to {display(metric,s["ci95_high"],True)}</td><td>{s["post_higher"]} / {s["pre_higher"]} / {s["tied"]}</td></tr>')
    persona_rows=[]
    for architecture in summaries:
        for persona_idx in range(persona_counts[architecture]):
            subset=[r for r in rows if r["architecture"]==architecture and r["persona_idx"]==persona_idx]
            persona_rows.append(f'<tr><td><span class="tag {architecture}">{architecture}</span></td><td>{esc(subset[0]["persona_name"])}</td>'+''.join(f'<td>{display(k,mean([r[f"delta_{k}"] for r in subset if finite(r[f"delta_{k}"])]),True)}</td>' for k in METRICS)+'</tr>')
    sensitivity_rows=[]
    for architecture in sensitivity:
        for metric,label in METRICS.items():
            full=summaries[architecture]["metrics"][metric]["mean_delta"]
            kept=sensitivity[architecture]["metrics"][metric]["mean_delta"]
            sensitivity_rows.append(f'<tr><td><span class="tag {architecture}">{architecture}</span></td><td>{label}</td><td>{display(metric,full,True)}</td><td>{display(metric,kept,True)}</td></tr>')
    interaction_rows=[]
    for metric,label in METRICS.items():
        deltas={a:summaries[a]["metrics"][metric]["mean_delta"] for a in summaries}
        interaction_rows.append(f'<tr><td>{label}</td><td>{display(metric,deltas["list"]-deltas["graph"],True)}</td><td>{display(metric,deltas["list"]-deltas["profile"],True)}</td><td>{display(metric,deltas["graph"]-deltas["profile"],True)}</td></tr>')
    scope = ", ".join(f"{architecture}: {count}" for architecture, count in persona_counts.items())
    balanced = len(set(persona_counts.values())) == 1
    scope_note = (
        f"Balanced comparison across {next(iter(persona_counts.values()))} personas per architecture"
        if balanced else f"Available complete personas by architecture: {scope}"
    )
    architecture_warning = "" if balanced else (
        '<div class="notice"><strong>Architecture-scope asymmetry:</strong> '
        f'{esc(scope)}. Architecture contrasts are descriptive until every architecture has the same persona scope.</div>'
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reranking analysis dashboard</title><style>
    :root{{--ink:#162033;--muted:#667085;--line:#d8e0e9;--paper:#f3f6fa;--navy:#17365d;--up:#9f2f24;--down:#087e65}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}header{{padding:50px max(22px,calc((100vw - 1250px)/2));background:linear-gradient(125deg,#102a4c,#214d7d,#087e8b);color:white}}header h1{{margin:0;font-size:clamp(30px,5vw,50px)}}header p{{max-width:850px;color:#dceafa}}nav{{position:sticky;top:0;z-index:10;display:flex;gap:18px;overflow:auto;padding:12px max(18px,calc((100vw - 1250px)/2));background:#fffffff2;border-bottom:1px solid var(--line)}}nav a{{color:var(--navy);font-weight:700;text-decoration:none;white-space:nowrap}}main{{max-width:1250px;margin:auto;padding:24px 18px 70px}}section{{scroll-margin-top:60px;margin:35px 0 55px}}h2{{font-size:28px;margin-bottom:6px}}.lead,.quiet{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}.card,.panel,.figure,.audit{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 25px #17365d12}}.card{{padding:18px}}.card b{{display:block;font-size:27px;color:var(--navy)}}.panel{{overflow:auto;max-height:70vh}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 12px;border-bottom:1px solid #e8ecf1;text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:var(--navy);color:white;font-size:12px}}.tag{{display:inline-block;padding:3px 8px;border-radius:999px;color:white;font-size:12px;font-weight:800}}.tag.list{{background:{ARCH_COLORS['list']}}}.tag.graph{{background:{ARCH_COLORS['graph']}}}.tag.profile{{background:{ARCH_COLORS['profile']}}}.up{{color:var(--up);font-weight:800}}.down{{color:var(--down);font-weight:800}}.figures{{display:grid;grid-template-columns:1fr 1fr;gap:15px}}.figure{{padding:14px}}.figure h3{{margin:2px 0 8px}}svg{{width:100%;height:auto}}.audit{{margin:10px 0;overflow:hidden}}.audit>summary{{padding:14px 16px;cursor:pointer}}.audit-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:0 16px 14px}}.audit article{{background:#f7f9fc;border-radius:10px;padding:13px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;border:1px solid var(--line);padding:11px;border-radius:8px;font:13px/1.5 ui-monospace,monospace}}blockquote{{margin:4px 0 10px;padding:6px 9px;border-left:3px solid #a9b6c7;color:var(--muted)}}.task,.audit-form{{margin:0;padding:0 16px 12px;color:var(--muted)}}.audit-form{{display:flex;gap:12px;flex-wrap:wrap}}select,input{{padding:7px;border:1px solid #b9c4d2;border-radius:7px}}.notice{{padding:15px;border-left:5px solid #c98112;background:#fff5df;border-radius:8px}}@media(max-width:800px){{.figures,.audit-grid{{grid-template-columns:1fr}}}}@media print{{nav,.audit-form{{display:none}}.panel{{max-height:none}}th{{position:static}}}}
    </style></head><body><header><h1>Pre/post-reranking analysis</h1><p>{scope_note}, with 49 matched contexts per persona. Generated entirely from saved artifacts.</p></header><nav><a href="#summary">Summary</a><a href="#figures">Figures</a><a href="#personas">Personas</a><a href="#sensitivity">Sensitivity</a><a href="#interaction">Interactions</a><a href="#audit">Human audit</a><a href="#methods">Methods</a></nav><main>
    <section id="summary"><h2>Paired results with confidence intervals</h2>{architecture_warning}<p class="lead">Positive deltas mean the post-reranking messages exposed more. Intervals use a hierarchical bootstrap over personas and contexts ({iterations:,} iterations).</p><div class="panel"><table><thead><tr><th>Architecture</th><th>Metric</th><th>Pre</th><th>Post</th><th>Post − pre</th><th>95% interval</th><th>Post higher / pre higher / tie</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></section>
    <section id="figures"><h2>Figures</h2><div class="figures"><div class="figure"><h3>Total disclosure</h3>{bar_svg(summaries,'average_exposed_total')}</div><div class="figure"><h3>Necessary recall</h3>{bar_svg(summaries,'necessary_recall')}</div><div class="figure"><h3>Ambiguous exposure</h3>{bar_svg(summaries,'ambiguous_exposure_rate')}</div><div class="figure"><h3>Privacy–utility change by context</h3>{scatter_svg(rows)}</div></div></section>
    <section id="personas"><h2>Persona heterogeneity</h2><p class="lead">Each cell is the persona-level mean post-minus-pre change.</p><div class="panel"><table><thead><tr><th>Architecture</th><th>Persona</th>{''.join(f'<th>{v}</th>' for v in METRICS.values())}</tr></thead><tbody>{''.join(persona_rows)}</tbody></table></div></section>
    <section id="sensitivity"><h2>Context-discard sensitivity</h2><p class="lead">The main conclusion should not depend on scenarios flagged as unsuitable by the labeling stage.</p><div class="panel"><table><thead><tr><th>Architecture</th><th>Metric</th><th>All contexts</th><th>Discarded excluded</th></tr></thead><tbody>{''.join(sensitivity_rows)}</tbody></table></div></section>
    <section id="interaction"><h2>Architecture interaction</h2><p class="lead">These contrasts compare the magnitude of the post-minus-pre effect. Positive values mean the architecture named first changed more.</p><div class="panel"><table><thead><tr><th>Metric</th><th>List − graph</th><th>List − profile</th><th>Graph − profile</th></tr></thead><tbody>{''.join(interaction_rows)}</tbody></table></div></section>
    <section id="audit"><h2>Stratified human-audit worksheet</h2><p class="lead">{audit_count} deterministic samples: every architecture and persona is represented. Review the actual message and the GPT-5.2 labels/evidence. Browser selections are stored locally; use <code>human_audit_sample.csv</code> for durable annotation.</p>{audit_cards}</section>
    <section id="methods"><h2>Methods and limitations</h2><div class="notice"><strong>Response-repeat asymmetry:</strong> pre-reranking metrics currently use one response per context, while post-reranking metrics average ten responses. Paired context estimates remain informative, but balanced pre repeats would improve variance estimation.</div><p>The bootstrap resamples personas with replacement, then resamples contexts within each selected persona. Architecture-specific intervals use the available persona count reported above. The raw paired data are saved in <code>paired_context_deltas.csv</code>; aggregate values and intervals are saved in <code>summary.json</code>.</p></section>
    </main><script>document.querySelectorAll('[data-audit]').forEach(el=>{{const k='cimem-audit-'+el.dataset.audit;el.value=localStorage.getItem(k)||'';el.addEventListener('input',()=>localStorage.setItem(k,el.value));}});</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-run", type=Path, default=DEFAULT_RUNS["list"])
    parser.add_argument("--graph-run", type=Path, default=DEFAULT_RUNS["graph"])
    parser.add_argument("--profile-run", type=Path, default=DEFAULT_RUNS["profile"])
    parser.add_argument("--personas", type=int, default=3)
    parser.add_argument(
        "--auto-personas", action="store_true",
        help="Use every contiguous persona with complete 49-context pre/post metrics for each architecture.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument(
        "--audit-per-persona",
        type=int,
        default=12,
        help="Paired contexts sampled per architecture/persona stratum (default: 12; 108 pairs total)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    runs = {"list": args.list_run, "graph": args.graph_run, "profile": args.profile_run}
    persona_counts = {
        architecture: complete_persona_count(root) if args.auto_personas else args.personas
        for architecture, root in runs.items()
    }
    if any(count == 0 for count in persona_counts.values()):
        raise ValueError(f"No complete persona pairs for one or more architectures: {persona_counts}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for architecture, root in runs.items():
        rows.extend(load_pairs(architecture, root, persona_counts[architecture]))
    summaries = {
        architecture: summarize([row for row in rows if row["architecture"] == architecture], args.bootstrap_iterations)
        for architecture in runs
    }
    sensitivity = {
        architecture: summarize(
            [row for row in rows if row["architecture"] == architecture and not row["context_discarded"]],
            args.bootstrap_iterations,
        )
        for architecture in runs
    }
    write_pairs_csv(args.output / "paired_context_deltas.csv", rows)
    audit_rows, audit_cards = audit_sample(
        runs, persona_counts, args.audit_per_persona, args.output / "human_audit_sample.csv"
    )
    summary_payload = {
        "schema_version": 1,
        "personas": args.personas if not args.auto_personas else None,
        "persona_counts": persona_counts,
        "contexts_per_persona": 49,
        "bootstrap_iterations": args.bootstrap_iterations,
        "runs": {key: str(value.resolve()) for key, value in runs.items()},
        "summary_all_contexts": summaries,
        "summary_context_discarded_excluded": sensitivity,
        "human_audit_sample_count": len(audit_rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (args.output / "REPORT.html").write_text(
        report_html(
            rows, summaries, sensitivity, audit_cards, len(audit_rows),
            args.bootstrap_iterations, persona_counts,
        ),
        encoding="utf-8",
    )
    print((args.output / "REPORT.html").resolve())


if __name__ == "__main__":
    main()
