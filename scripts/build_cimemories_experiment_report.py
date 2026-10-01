#!/usr/bin/env python3
"""Build a portable HTML navigator for a CIMemories experiment folder."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_PIPELINE = Path(
    "research_outputs/"
    "privacy-pipeline-cimemories-dataset_cimemories-raw_mem3_20260912_054751"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def pct(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{100 * value:.1f}%"


def num(value: Any, digits: int = 2) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def metric_record(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    metrics = payload.get("metrics") or {}
    return {
        "context_idx": payload.get("context_idx"),
        "recipient": payload.get("recipient"),
        "task": payload.get("task"),
        "discarded": bool(payload.get("context_discarded")),
        "necessary_recall": metrics.get("necessary_recall"),
        "inappropriate_leak_rate": metrics.get("inappropriate_leak_rate"),
        "ambiguous_exposure_rate": metrics.get("ambiguous_exposure_rate"),
        "average_exposed_total": metrics.get("average_exposed_total"),
        "repeat_count": metrics.get("repeat_count"),
        "necessary_total": metrics.get("necessary_total"),
        "inappropriate_total": metrics.get("inappropriate_total"),
        "ambiguous_total": metrics.get("ambiguous_total"),
        "path": path,
    }


def average(records: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def artifact_card(title: str, description: str, href: str, badge: str) -> str:
    return f"""
      <a class="card" href="{esc(href)}">
        <span class="badge">{esc(badge)}</span>
        <h3>{esc(title)}</h3>
        <p>{esc(description)}</p>
        <span class="open">Open artifact →</span>
      </a>"""


def fact_list(values: list[Any]) -> str:
    if not values:
        return '<p class="muted">None recorded.</p>'
    return "<ol>" + "".join(f"<li>{esc(value)}</li>" for value in values) + "</ol>"


def exposure_list(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return '<p class="muted">No exposed attributes identified.</p>'
    return "<ul>" + "".join(
        f"<li><strong>{esc(attribute)}</strong><blockquote>{esc(evidence)}</blockquote></li>"
        for attribute, evidence in value.items()
    ) + "</ul>"


def build_report(root: Path, output: Path) -> None:
    root = root.resolve()
    manifest = load_json(root / "manifest.json")
    persona_dirs = sorted((root / "personas").glob("persona_*"))
    pre_root = root / "pre_rerank_response_evaluation"
    pre_manifest = load_json(pre_root / "manifest.json") if (pre_root / "manifest.json").is_file() else {}

    personas: list[dict[str, Any]] = []
    context_rows: list[str] = []
    response_cards: list[str] = []
    for persona_dir in persona_dirs:
        persona_idx = int(persona_dir.name.split("_")[-1])
        provenance_path = persona_dir / "provenance.json"
        provenance = load_json(provenance_path) if provenance_path.is_file() else {}
        pipeline_path = persona_dir / "pipeline/privacy_pipeline_cimemories.json"
        pipeline = load_json(pipeline_path)
        persona_name = pipeline.get("persona_name") or provenance.get("persona_name") or persona_dir.name

        post_files = sorted((persona_dir / "privacy_metrics").glob("context_*/privacy_metrics_cimemories.json"))
        post = [metric_record(path) for path in post_files]
        pre_dir = pre_root / "personas" / persona_dir.name / "privacy_metrics"
        pre_files = sorted(pre_dir.glob("context_*/privacy_metrics_cimemories.json")) if pre_dir.is_dir() else []
        pre = {r["context_idx"]: r for r in (metric_record(path) for path in pre_files)}

        personas.append(
            {
                "idx": persona_idx,
                "name": persona_name,
                "post": post,
                "pre": list(pre.values()),
                "persona_dir": persona_dir,
            }
        )

        pre_persona = pre_root / "personas" / persona_dir.name
        pre_generations = load_jsonl(pre_persona / "responses.jsonl")
        if pre_generations:
            post_responses = {
                (record.get("context_idx"), record.get("repeat_idx")): record
                for record in load_jsonl(persona_dir / "query/responses.jsonl")
            }
            post_exposure_files = sorted((persona_dir / "query").glob("exposed_attributes_*.jsonl"))
            post_exposures = (
                {
                    (record.get("context_idx"), record.get("repeat_idx")): record
                    for record in load_jsonl(post_exposure_files[-1])
                }
                if post_exposure_files
                else {}
            )
            pre_exposures = {
                (record.get("context_idx"), record.get("repeat_idx")): record
                for record in load_jsonl(pre_persona / "exposed_attributes.jsonl")
            }
            for pre_record in pre_generations:
                context_idx = int(pre_record.get("context_idx") or 0)
                repeat_idx = int(pre_record.get("repeat_idx") or 0)
                key = (context_idx, repeat_idx)
                post_record = post_responses.get(key, {})
                pre_exposure = pre_exposures.get(key, {})
                post_exposure = post_exposures.get(key, {})
                candidates = pre_record.get("pre_rerank_facts") or []
                selected = (post_record.get("rerank") or {}).get("selected_memories") or []
                source_pre = pre_persona / "generation_calls" / f"context_{context_idx:03d}_repeat_{repeat_idx:02d}.json"
                post_history = persona_dir / "query/histories" / f"context_{context_idx:03d}_repeat_{repeat_idx:02d}_history.txt"
                search_blob = " ".join(
                    [
                        persona_dir.name,
                        str(persona_name),
                        str(pre_record.get("recipient")),
                        str(pre_record.get("task")),
                        str(context_idx),
                        str(repeat_idx),
                    ]
                ).lower()
                response_cards.append(
                    f"""
                    <details class="sample" data-response-persona="{persona_idx}" data-response-search="{esc(search_blob)}">
                      <summary>
                        <span class="context-pill">{context_idx:02d}</span>
                        <span><strong>{esc(pre_record.get('recipient'))}</strong><small>{esc(pre_record.get('task'))}</small></span>
                        <span class="sample-meta">{esc(persona_name)} · repeat {repeat_idx + 1}</span>
                      </summary>
                      <div class="sample-body">
                        <div class="memory-comparison">
                          <details><summary><strong>Pre-rerank candidates</strong> · {len(candidates)} facts</summary>{fact_list(candidates)}</details>
                          <details><summary><strong>Post-rerank selection</strong> · {len(selected)} facts</summary>{fact_list(selected)}</details>
                        </div>
                        <div class="response-grid">
                          <article class="response pre"><h4>Before reranking</h4><pre>{esc(pre_record.get('assistant_response'))}</pre><h5>Exposed attributes</h5>{exposure_list(pre_exposure.get('exposed_attributes'))}</article>
                          <article class="response post"><h4>After reranking</h4><pre>{esc(post_record.get('assistant_response'))}</pre><h5>Exposed attributes</h5>{exposure_list(post_exposure.get('exposed_attributes'))}</article>
                        </div>
                        <p class="source-links"><a href="{esc(rel(source_pre, root))}">exact pre generation JSON</a> · <a href="{esc(rel(post_history, root))}">readable post history</a></p>
                      </div>
                    </details>"""
                )

        for record in post:
            context_idx = int(record["context_idx"])
            pre_record = pre.get(context_idx, {})
            post_href = rel(record["path"], root)
            pre_href = rel(pre_record["path"], root) if pre_record else ""
            history = persona_dir / "query/histories" / f"context_{context_idx:03d}_repeat_00_history.txt"
            history_link = f'<a href="{esc(rel(history, root))}">history</a>' if history.is_file() else "—"
            pre_link = f'<a href="{esc(pre_href)}">pre JSON</a>' if pre_href else "—"
            search_blob = " ".join(
                [persona_dir.name, str(persona_name), str(record["recipient"]), str(record["task"]), str(context_idx)]
            ).lower()
            context_rows.append(
                f"""
                <tr data-persona="{persona_idx}" data-search="{esc(search_blob)}">
                  <td><span class="mono">{persona_dir.name}</span><br>{esc(persona_name)}</td>
                  <td><span class="context-pill">{context_idx:02d}</span></td>
                  <td><strong>{esc(record['recipient'])}</strong><br><span class="muted">{esc(record['task'])}</span></td>
                  <td>{pct(pre_record.get('necessary_recall'))}</td>
                  <td>{pct(record.get('necessary_recall'))}</td>
                  <td>{pct(pre_record.get('inappropriate_leak_rate'))}</td>
                  <td>{pct(record.get('inappropriate_leak_rate'))}</td>
                  <td>{pct(pre_record.get('ambiguous_exposure_rate'))}</td>
                  <td>{pct(record.get('ambiguous_exposure_rate'))}</td>
                  <td>{num(pre_record.get('average_exposed_total'))}</td>
                  <td>{num(record.get('average_exposed_total'))}</td>
                  <td class="links"><a href="{esc(post_href)}">post JSON</a> · {pre_link} · {history_link}</td>
                </tr>"""
            )

    persona_rows = []
    for p in personas:
        post = p["post"]
        pre = p["pre"]
        persona_rel = rel(p["persona_dir"], root)
        persona_rows.append(
            f"""
            <tr>
              <td><span class="mono">persona_{p['idx']:03d}</span></td>
              <td><strong>{esc(p['name'])}</strong></td>
              <td>{len(post)}</td>
              <td>{len(pre) if pre else '—'}</td>
              <td>{pct(average(pre, 'necessary_recall'))}</td>
              <td>{pct(average(post, 'necessary_recall'))}</td>
              <td>{pct(average(pre, 'inappropriate_leak_rate'))}</td>
              <td>{pct(average(post, 'inappropriate_leak_rate'))}</td>
              <td>{num(average(pre, 'average_exposed_total'))}</td>
              <td>{num(average(post, 'average_exposed_total'))}</td>
              <td class="links"><a href="{persona_rel}/pipeline/privacy_pipeline_cimemories.json">pipeline</a> · <a href="{persona_rel}/query/responses.jsonl">responses</a> · <a href="{persona_rel}/query/initialized_memory/full_list_memory.html">memory</a></td>
            </tr>"""
        )

    total_post = sum(len(p["post"]) for p in personas)
    total_pre = sum(len(p["pre"]) for p in personas)
    cards = "".join(
        [
            artifact_card("Experiment manifest", "Configuration, completion state, models, and persona paths.", "manifest.json", "OVERVIEW"),
            artifact_card("Readable folder guide", "Plain-language explanation of files, stages, and naming.", "TABLE_OF_CONTENTS.md", "GUIDE"),
            artifact_card("Memory-stage summary", "Exact-match availability before and after reranking.", "memory_stage_metrics_exact_match_summary.json", "MEMORY"),
            artifact_card("Memory-stage table", "Dataset-wide rows suitable for a spreadsheet or statistics package.", "memory_stage_metrics_exact_match.csv", "CSV"),
            artifact_card("Pre-rerank analysis", "Early 10-context offline analysis; retained as a pilot artifact.", "pre_rerank_response_evaluation/offline_reranking_exposure_analysis.md", "PILOT"),
            artifact_card("Pre-rerank manifest", "Scope, models, completion state, and derived-stage artifacts.", "pre_rerank_response_evaluation/manifest.json", "PRE"),
        ]
    )

    persona_options = "".join(
        f'<option value="{p["idx"]}">persona_{p["idx"]:03d} — {esc(p["name"])}</option>' for p in personas
    )
    architecture = manifest.get("memory_mode_name") or "list-rerank"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CIMemories experiment navigator</title>
  <style>
    :root {{ --ink:#172033; --muted:#667085; --line:#dbe2ea; --paper:#f4f7fb; --card:#fff; --navy:#17365d; --blue:#2f6fed; --teal:#087e8b; --gold:#c98112; --shadow:0 12px 30px rgba(23,54,93,.09); }}
    * {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font:15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ color:white; background:linear-gradient(128deg,#102a4c,#214d7d 58%,#087e8b); padding:56px max(24px,calc((100vw - 1320px)/2)); }}
    header h1 {{ max-width:900px; margin:0 0 10px; font-size:clamp(30px,5vw,52px); line-height:1.08; letter-spacing:-.035em; }}
    header p {{ max-width:850px; margin:0; color:#dceafa; font-size:17px; }}
    nav {{ position:sticky; top:0; z-index:20; display:flex; gap:18px; overflow:auto; padding:12px max(20px,calc((100vw - 1320px)/2)); background:rgba(255,255,255,.95); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }}
    nav a {{ color:var(--navy); text-decoration:none; white-space:nowrap; font-weight:700; }}
    main {{ max-width:1320px; margin:auto; padding:28px 20px 80px; }}
    section {{ scroll-margin-top:65px; margin:32px 0 50px; }} h2 {{ margin:0 0 8px; font-size:28px; letter-spacing:-.02em; }}
    .lead,.muted {{ color:var(--muted); }} .lead {{ margin-top:0; max-width:850px; }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-top:-52px; position:relative; }}
    .stat,.card,.panel {{ background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); }}
    .stat {{ padding:20px; }} .stat b {{ display:block; font-size:29px; color:var(--navy); }} .stat span {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:16px; }}
    .card {{ display:block; padding:20px; color:inherit; text-decoration:none; transition:.18s ease; }} .card:hover {{ transform:translateY(-3px); border-color:#9bb7e4; }}
    .card h3 {{ margin:11px 0 6px; }} .card p {{ color:var(--muted); min-height:46px; }} .open {{ color:var(--blue); font-weight:750; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; color:#164a51; background:#d9f1f2; font-size:11px; font-weight:800; letter-spacing:.07em; }}
    .panel {{ overflow:hidden; }} .table-wrap {{ overflow:auto; max-height:72vh; }}
    table {{ width:100%; border-collapse:separate; border-spacing:0; }} th,td {{ padding:11px 12px; text-align:left; border-bottom:1px solid #e9edf2; vertical-align:top; white-space:nowrap; }}
    th {{ position:sticky; top:0; z-index:5; color:#fff; background:var(--navy); font-size:12px; letter-spacing:.025em; }} tbody tr:hover {{ background:#f2f7ff; }}
    .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }} .context-pill {{ display:inline-grid; place-items:center; width:31px; height:31px; border-radius:50%; background:#e8effc; color:var(--navy); font-weight:800; }}
    .links a {{ color:var(--blue); text-decoration:none; }} .controls {{ display:flex; flex-wrap:wrap; gap:10px; margin:14px 0; }} input,select {{ min-height:42px; padding:8px 11px; border:1px solid #bdc8d6; border-radius:9px; background:white; font:inherit; }} input {{ min-width:min(100%,360px); }}
    .flow {{ display:grid; grid-template-columns:repeat(5,1fr); gap:9px; align-items:center; }} .step {{ min-height:105px; padding:15px; border:1px solid var(--line); border-radius:13px; background:white; }} .step b {{ display:block; color:var(--navy); margin-bottom:5px; }} .arrow {{ display:none; }}
    .sample {{ margin:12px 0; background:white; border:1px solid var(--line); border-radius:14px; box-shadow:0 5px 18px rgba(23,54,93,.05); overflow:hidden; }}
    .sample>summary {{ display:grid; grid-template-columns:auto 1fr auto; gap:13px; align-items:center; padding:14px 17px; cursor:pointer; }} .sample>summary:hover {{ background:#f4f8ff; }} .sample summary small {{ display:block; color:var(--muted); }} .sample-meta {{ color:var(--muted); font-size:13px; }}
    .sample-body {{ padding:5px 17px 18px; border-top:1px solid var(--line); }} .memory-comparison,.response-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; }}
    .memory-comparison>details {{ padding:12px; border-radius:10px; background:#f6f8fb; }} .memory-comparison ol {{ max-height:310px; overflow:auto; padding-left:28px; }}
    .response {{ padding:16px; border-radius:12px; border-top:4px solid var(--teal); background:#f5fbfb; }} .response.post {{ border-color:var(--blue); background:#f4f7ff; }} .response h4 {{ margin:0 0 10px; font-size:18px; }} .response h5 {{ margin:16px 0 7px; }}
    .response pre {{ margin:0; padding:13px; border-radius:9px; background:white; border:1px solid var(--line); font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; overflow-wrap:anywhere; }} .response ul {{ padding-left:20px; }} blockquote {{ margin:5px 0 12px; padding:7px 10px; border-left:3px solid #a9b9cf; color:var(--muted); background:rgba(255,255,255,.65); }}
    .source-links a {{ color:var(--blue); text-decoration:none; }}
    .notice {{ padding:15px 18px; border-left:5px solid var(--gold); border-radius:8px; background:#fff6e5; color:#70490d; }}
    footer {{ padding:25px; text-align:center; color:var(--muted); }}
    @media(max-width:850px) {{ .flow,.memory-comparison,.response-grid {{ grid-template-columns:1fr; }} .sample>summary {{ grid-template-columns:auto 1fr; }} .sample-meta {{ grid-column:2; }} th,td {{ padding:9px; }} }}
    @media print {{ nav,.controls {{ display:none; }} header {{ padding:30px; }} .stats {{ margin:15px 0; }} .table-wrap {{ max-height:none; overflow:visible; }} th {{ position:static; }} }}
  </style>
</head>
<body>
  <header>
    <h1>CIMemories experiment navigator</h1>
    <p>A readable map of the <strong>{esc(architecture)}</strong> experiment, connecting saved prompts, generated messages, exposure judgments, and privacy metrics to their exact source artifacts.</p>
  </header>
  <nav>
    <a href="#overview">Overview</a><a href="#flow">Experiment flow</a><a href="#artifacts">Important artifacts</a><a href="#personas">Personas</a><a href="#contexts">Context browser</a><a href="#responses">Response inspector</a><a href="#notes">Interpretation notes</a>
  </nav>
  <main>
    <section id="overview">
      <div class="stats">
        <div class="stat"><b>{len(personas)}</b><span>personas</span></div>
        <div class="stat"><b>49</b><span>scenarios per persona</span></div>
        <div class="stat"><b>10</b><span>post-rerank generations per scenario</span></div>
        <div class="stat"><b>{total_post}</b><span>post-rerank context summaries</span></div>
        <div class="stat"><b>{total_pre}</b><span>pre-rerank context summaries</span></div>
      </div>
    </section>

    <section id="flow">
      <h2>Experiment flow</h2>
      <p class="lead">The dashboard follows the causal stages of the experiment. Each later result can be traced back to the exact saved inputs.</p>
      <div class="flow">
        <div class="step"><b>1 · Persona memory</b>Original facts are stored as list-memory entries.</div>
        <div class="step"><b>2 · Candidate retrieval</b>Up to 50 facts are retrieved for a recipient and task.</div>
        <div class="step"><b>3 · Reranking</b>The candidate set is reduced to selected task-relevant facts.</div>
        <div class="step"><b>4 · Message generation</b>GPT-5.6 Sol writes the recipient-facing message.</div>
        <div class="step"><b>5 · Evaluation</b>GPT-5.2 identifies exposed facts; deterministic code computes metrics.</div>
      </div>
    </section>

    <section id="artifacts">
      <h2>Important artifacts</h2>
      <p class="lead">These are the most useful entry points. Every card opens the original artifact, not a rewritten copy.</p>
      <div class="cards">{cards}</div>
    </section>

    <section id="personas">
      <h2>Persona summary</h2>
      <p class="lead">Post-reranking results cover all ten personas. Pre-reranking response metrics are shown only where that derived stage exists.</p>
      <div class="panel table-wrap">
        <table>
          <thead><tr><th>ID</th><th>Persona</th><th>Post contexts</th><th>Pre contexts</th><th>Pre necessary</th><th>Post necessary</th><th>Pre inappropriate</th><th>Post inappropriate</th><th>Pre exposed</th><th>Post exposed</th><th>Artifacts</th></tr></thead>
          <tbody>{''.join(persona_rows)}</tbody>
        </table>
      </div>
    </section>

    <section id="contexts">
      <h2>Scenario and metric browser</h2>
      <p class="lead">Search by persona, recipient, task, or context number. A dash means the corresponding pre-reranking result has not been generated.</p>
      <div class="controls">
        <input id="search" type="search" placeholder="Search recipient, task, persona, or context…">
        <select id="persona"><option value="">All personas</option>{persona_options}</select>
        <span id="visible-count" class="muted"></span>
      </div>
      <div class="panel table-wrap">
        <table id="contexts-table">
          <thead><tr><th>Persona</th><th>Context</th><th>Scenario</th><th>Pre necessary</th><th>Post necessary</th><th>Pre inappropriate</th><th>Post inappropriate</th><th>Pre ambiguous</th><th>Post ambiguous</th><th>Pre exposed</th><th>Post exposed</th><th>Evidence</th></tr></thead>
          <tbody>{''.join(context_rows)}</tbody>
        </table>
      </div>
    </section>

    <section id="responses">
      <h2>Human-readable pre/post response inspector</h2>
      <p class="lead">Open any sample to compare the actual messages, candidate memories, selected memories, exposed attributes, and supporting judge evidence. Everything is rendered from saved checkpoints; opening this report makes no model calls.</p>
      <div class="controls">
        <input id="response-search" type="search" placeholder="Search response samples by scenario or context…">
        <select id="response-persona"><option value="">All available personas</option>{persona_options}</select>
        <span id="response-count" class="muted"></span>
      </div>
      <div id="response-samples">{''.join(response_cards) if response_cards else '<p class="notice">No pre-reranking response checkpoints were found.</p>'}</div>
    </section>

    <section id="notes">
      <h2>Interpretation notes</h2>
      <div class="notice"><strong>Pre/post scope differs.</strong> The main post-reranking experiment contains 10 personas × 49 contexts × 10 repeats. The current pre-reranking extension contains persona 0 × 49 contexts × 3 repeats. Treat cross-persona post results as the main experiment and persona-0 pre/post results as a paired pilot until the derived stage is expanded.</div>
      <p><strong>Necessary recall</strong> measures task-relevant attributes disclosed in the response. <strong>Inappropriate leakage</strong> measures attributes labeled unsuitable for the recipient/task. <strong>Ambiguous exposure</strong> covers facts without a clear share/do-not-share classification. <strong>Mean exposed attributes</strong> is the average number of distinct persona facts revealed in one generated message.</p>
      <p class="muted">This report is static and self-contained. It performs no network requests. Links are relative, so keep it at the root of the experiment folder.</p>
    </section>
  </main>
  <footer>Generated from saved CIMemories artifacts · {esc(root.name)}</footer>
  <script>
    const search = document.getElementById('search');
    const persona = document.getElementById('persona');
    const rows = [...document.querySelectorAll('#contexts-table tbody tr')];
    const count = document.getElementById('visible-count');
    function filterRows() {{
      const q = search.value.trim().toLowerCase();
      const p = persona.value;
      let visible = 0;
      rows.forEach(row => {{
        const show = (!q || row.dataset.search.includes(q)) && (!p || row.dataset.persona === p);
        row.hidden = !show;
        if (show) visible++;
      }});
      count.textContent = `${{visible}} of ${{rows.length}} contexts shown`;
    }}
    search.addEventListener('input', filterRows);
    persona.addEventListener('change', filterRows);
    filterRows();
    const responseSearch = document.getElementById('response-search');
    const responsePersona = document.getElementById('response-persona');
    const responseSamples = [...document.querySelectorAll('.sample')];
    const responseCount = document.getElementById('response-count');
    function filterResponses() {{
      const q = responseSearch.value.trim().toLowerCase();
      const p = responsePersona.value;
      let visible = 0;
      responseSamples.forEach(sample => {{
        const show = (!q || sample.dataset.responseSearch.includes(q)) && (!p || sample.dataset.responsePersona === p);
        sample.hidden = !show;
        if (show) visible++;
      }});
      responseCount.textContent = `${{visible}} of ${{responseSamples.length}} response pairs shown`;
    }}
    responseSearch.addEventListener('input', filterResponses);
    responsePersona.addEventListener('change', filterResponses);
    filterResponses();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", nargs="?", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--output", type=Path, help="Default: <pipeline>/EXPERIMENT_REPORT.html")
    args = parser.parse_args()
    root = args.pipeline
    output = args.output or root / "EXPERIMENT_REPORT.html"
    build_report(root, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
