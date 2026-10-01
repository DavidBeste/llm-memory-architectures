from __future__ import annotations

import csv
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .persona import extract_persona_memory_statements, load_persona_entry


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_artifact(raw: str, manifest_path: Path) -> Path:
    path = Path(raw).expanduser()
    candidates = (path, Path.cwd() / path, manifest_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Referenced artifact does not exist: {raw}")


def _manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.expanduser().resolve()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return manifest_path, _read_json(manifest_path)


def _pipeline_manifests(path: Path) -> tuple[Path, list[Path], dict[str, Any]]:
    source_manifest, source = _manifest(path)
    if source.get("command") == "run_privacy_pipeline_cimemories_dataset":
        if source.get("status") != "complete":
            raise ValueError(f"Dataset pipeline is not complete: {source.get('status')}")
        manifests = []
        for persona in source.get("personas") or []:
            raw = persona.get("local_pipeline")
            if not isinstance(raw, str):
                raise ValueError("Dataset persona entry has no local_pipeline")
            pipeline = _resolve_artifact(raw, source_manifest)
            manifest = pipeline / "manifest.json" if pipeline.is_dir() else pipeline
            if not manifest.is_file():
                raise FileNotFoundError(f"Persona pipeline manifest not found: {manifest}")
            manifests.append(manifest)
        expected = int(source.get("persona_count") or 0)
        if expected < 1 or len(manifests) != expected:
            raise ValueError(f"Incomplete persona coverage: {len(manifests)}/{expected}")
        return source_manifest, manifests, source

    # A privacy-pipeline directory has contexts and a query_output_dir pointer.
    if isinstance(source.get("contexts"), list) and isinstance(source.get("artifacts"), dict):
        return source_manifest, [source_manifest], source

    # Also accept a query-run directory directly.
    if isinstance(source.get("records"), list) and source.get("memory_mode") is not None:
        return source_manifest, [source_manifest], source
    raise ValueError("Expected a completed dataset pipeline, persona pipeline, or query-run path")


def _query_manifest(pipeline_manifest: Path, pipeline: dict[str, Any]) -> Path:
    if isinstance(pipeline.get("records"), list):
        return pipeline_manifest
    artifacts = pipeline.get("artifacts") or {}
    raw = artifacts.get("query_output_dir")
    if not isinstance(raw, str):
        raise ValueError(f"Pipeline has no query_output_dir: {pipeline_manifest}")
    query_dir = _resolve_artifact(raw, pipeline_manifest)
    manifest = query_dir / "manifest.json" if query_dir.is_dir() else query_dir
    if not manifest.is_file():
        raise FileNotFoundError(f"Query manifest not found: {manifest}")
    return manifest


def _dataset_path(query: dict[str, Any], pipeline: dict[str, Any], manifest: Path) -> Path | None:
    raw = query.get("source_dataset") or pipeline.get("dataset_name")
    if not isinstance(raw, str):
        return None
    path = Path(raw).expanduser()
    for candidate in (path, Path.cwd() / path, manifest.parent / path):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _records(query: dict[str, Any], query_manifest: Path) -> list[dict[str, Any]]:
    artifacts = query.get("artifacts") or {}
    raw = artifacts.get("responses_jsonl")
    if isinstance(raw, str):
        path = _resolve_artifact(raw, query_manifest)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = query.get("records") or []
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Query run contains no response records: {query_manifest}")
    return rows


def _clean_rerank(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if key not in {"raw_response", "prompt"}}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _compact_properties(
    properties: dict[str, Any], *, include_embeddings: bool = False
) -> dict[str, Any]:
    """Keep inspectable graph metadata without exporting high-dimensional vectors."""
    return {
        str(key): _json_safe(value)
        for key, value in properties.items()
        if include_embeddings or "embedding" not in str(key).lower()
    }


def _read_full_graph(
    *,
    group_id: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    try:
        from neo4j import GraphDatabase, READ_ACCESS
    except ImportError as exc:
        raise RuntimeError(
            "--export-full-graph requires the neo4j Python package in the active environment"
        ) from exc
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session(database="neo4j", default_access_mode=READ_ACCESS) as session:
            node_rows = session.run(
                """
                MATCH (n {group_id: $group_id})
                RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS properties
                ORDER BY coalesce(n.created_at, datetime({epochMillis: 0})), n.uuid
                """,
                group_id=group_id,
            ).data()
            relationship_rows = session.run(
                """
                MATCH (source)-[relationship]->(target)
                WHERE relationship.group_id = $group_id
                RETURN elementId(relationship) AS element_id,
                       type(relationship) AS type,
                       elementId(source) AS source_element_id,
                       elementId(target) AS target_element_id,
                       properties(relationship) AS properties
                ORDER BY coalesce(relationship.created_at, datetime({epochMillis: 0})), relationship.uuid
                """,
                group_id=group_id,
            ).data()
    finally:
        driver.close()
    nodes = [
        {
            "element_id": row["element_id"],
            "labels": row["labels"],
            "properties": _compact_properties(
                row["properties"], include_embeddings=include_embeddings
            ),
        }
        for row in node_rows
    ]
    relationships = [
        {
            "element_id": row["element_id"],
            "type": row["type"],
            "source_element_id": row["source_element_id"],
            "target_element_id": row["target_element_id"],
            "properties": _compact_properties(
                row["properties"], include_embeddings=include_embeddings
            ),
        }
        for row in relationship_rows
    ]
    return {"group_id": group_id, "nodes": nodes, "relationships": relationships}


def _graphml(graph: dict[str, Any]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="labels" for="node" attr.name="labels" attr.type="string"/>',
        '  <key id="name" for="node" attr.name="name" attr.type="string"/>',
        '  <key id="summary" for="node" attr.name="summary" attr.type="string"/>',
        '  <key id="uuid" for="all" attr.name="uuid" attr.type="string"/>',
        '  <key id="relationship_type" for="edge" attr.name="relationship_type" attr.type="string"/>',
        '  <key id="fact" for="edge" attr.name="fact" attr.type="string"/>',
        '  <graph id="G" edgedefault="directed">',
    ]
    for node in graph["nodes"]:
        props = node["properties"]
        lines.extend([
            f'    <node id="{html.escape(node["element_id"], quote=True)}">',
            f'      <data key="labels">{html.escape(",".join(node["labels"]))}</data>',
            f'      <data key="name">{html.escape(str(props.get("name") or ""))}</data>',
            f'      <data key="summary">{html.escape(str(props.get("summary") or props.get("content") or ""))}</data>',
            f'      <data key="uuid">{html.escape(str(props.get("uuid") or ""))}</data>',
            "    </node>",
        ])
    for edge in graph["relationships"]:
        props = edge["properties"]
        lines.extend([
            f'    <edge id="{html.escape(edge["element_id"], quote=True)}" source="{html.escape(edge["source_element_id"], quote=True)}" target="{html.escape(edge["target_element_id"], quote=True)}">',
            f'      <data key="relationship_type">{html.escape(edge["type"])}</data>',
            f'      <data key="fact">{html.escape(str(props.get("fact") or ""))}</data>',
            f'      <data key="uuid">{html.escape(str(props.get("uuid") or ""))}</data>',
            "    </edge>",
        ])
    lines.extend(["  </graph>", "</graphml>", ""])
    return "\n".join(lines)


def _full_graph_svg(graph: dict[str, Any], persona_name: str) -> str:
    size = 3200
    center = size / 2
    entities = [node for node in graph["nodes"] if "Entity" in node["labels"]]
    episodes = [node for node in graph["nodes"] if "Episodic" in node["labels"]]
    positions: dict[str, tuple[float, float]] = {}
    for nodes, radius, phase in ((entities, 1250, 0.0), (episodes, 670, math.pi / max(1, len(episodes)))):
        for index, node in enumerate(nodes):
            angle = phase + 2 * math.pi * index / max(1, len(nodes))
            positions[node["element_id"]] = (center + radius * math.cos(angle), center + radius * math.sin(angle))
    edge_lines = []
    for edge in graph["relationships"]:
        source = positions.get(edge["source_element_id"])
        target = positions.get(edge["target_element_id"])
        if source is None or target is None:
            continue
        color = "#94a3b8" if edge["type"] == "MENTIONS" else "#e07a24"
        fact = str(edge["properties"].get("fact") or edge["type"])
        edge_lines.append(
            f'<line x1="{source[0]:.1f}" y1="{source[1]:.1f}" x2="{target[0]:.1f}" y2="{target[1]:.1f}" stroke="{color}" stroke-width="3" opacity="0.55"><title>{html.escape(edge["type"] + ": " + fact)}</title></line>'
        )
    node_shapes = []
    for index, node in enumerate(graph["nodes"]):
        x, y = positions[node["element_id"]]
        props = node["properties"]
        episodic = "Episodic" in node["labels"]
        label = f"Episode {index + 1}" if episodic else str(props.get("name") or props.get("uuid") or "Entity")
        display = label if len(label) <= 28 else label[:25] + "…"
        fill = "#dbeafe" if episodic else "#fff7ed"
        stroke = "#2474b5" if episodic else "#e07a24"
        tooltip = json.dumps(props, ensure_ascii=False, indent=2)
        node_shapes.append(
            f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="42" fill="{fill}" stroke="{stroke}" stroke-width="5"><title>{html.escape(tooltip)}</title></circle>'
            f'<text x="{x:.1f}" y="{y + 65:.1f}" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#172033">{html.escape(display)}</text></g>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}"><rect width="100%" height="100%" fill="white"/>
<text x="70" y="75" font-family="sans-serif" font-size="42" font-weight="700" fill="#172033">Full Graphiti graph: {html.escape(persona_name)}</text>
<text x="70" y="125" font-family="sans-serif" font-size="24" fill="#475569">{len(entities)} entities · {len(episodes)} episodes · {len(graph['relationships'])} relationships · orange RELATES_TO · gray MENTIONS</text>
{''.join(edge_lines)}{''.join(node_shapes)}</svg>'''


def _full_graph_html(graph: dict[str, Any], persona_name: str) -> str:
    """Return an offline interactive viewer for the exact exported Graphiti topology."""
    graph_json = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")
    template = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Interactive Graphiti graph: __TITLE__</title>
<style>
:root{--ink:#172033;--muted:#64748b;--line:#dbe4ee;--blue:#2474b5;--orange:#e07a24;--wash:#f5f8fb}
*{box-sizing:border-box}body{margin:0;color:var(--ink);font:14px/1.45 "Segoe UI",Arial,sans-serif;background:var(--wash);overflow:hidden}
header{height:74px;padding:12px 20px;background:linear-gradient(120deg,#17365d,#236fa1);color:white;display:flex;align-items:center;justify-content:space-between;gap:20px}
h1{font-size:21px;margin:0}header p{margin:2px 0 0;color:#dbeafe}.layout{height:calc(100vh - 74px);display:grid;grid-template-columns:minmax(0,1fr) 360px}
.graph-pane{position:relative;background:white;overflow:hidden}.toolbar{position:absolute;z-index:3;top:14px;left:14px;right:14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;pointer-events:none}
.toolbar>*{pointer-events:auto}input[type=search]{width:min(390px,45vw);padding:9px 12px;border:1px solid #aebdcb;border-radius:9px;background:#fffffff2;font:inherit}
button,.toggle{border:1px solid #b8c8d8;border-radius:9px;background:#fffffff2;padding:8px 11px;cursor:pointer}.toggle{display:flex;gap:6px;align-items:center}
#graph{width:100%;height:100%;cursor:grab;user-select:none}#graph.dragging{cursor:grabbing}.semantic{stroke:var(--orange);stroke-width:2.5;opacity:.66}.mentions{stroke:#94a3b8;stroke-width:1.4;opacity:.36}.edge-label{fill:#9a4b10;font:700 11px "Segoe UI",Arial,sans-serif;text-anchor:middle;paint-order:stroke;stroke:white;stroke-width:4px;stroke-linejoin:round;pointer-events:none}
.edge-hit{stroke:transparent;stroke-width:13;cursor:pointer}.node{cursor:pointer}.node circle{fill:#fff7ed;stroke:var(--orange);stroke-width:3}.node.episode rect{fill:#dbeafe;stroke:var(--blue);stroke-width:3}
.node text{fill:var(--ink);font:13px "Segoe UI",Arial,sans-serif;text-anchor:middle;paint-order:stroke;stroke:white;stroke-width:4px;stroke-linejoin:round}.dim{opacity:.09!important}.selected circle,.selected rect{stroke:#16a34a!important;stroke-width:6!important}.neighbor circle,.neighbor rect{stroke:#22c55e!important}
aside{border-left:1px solid var(--line);background:#fff;overflow:auto;padding:18px}aside h2{font-size:17px;color:#17365d;margin:0 0 8px}aside h3{font-size:14px;margin:20px 0 6px}.hint{color:var(--muted)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;border:1px solid var(--line);border-radius:9px;padding:10px;font:12px/1.45 Consolas,monospace}
.fact{border-left:4px solid var(--orange);padding:7px 9px;margin:7px 0;background:#fff7ed}.legend{position:absolute;left:14px;bottom:14px;background:#fffffff0;border:1px solid var(--line);border-radius:9px;padding:8px 11px}.dot{display:inline-block;width:10px;height:10px;margin:0 4px 0 10px}.dot:first-child{margin-left:0}
@media(max-width:850px){.layout{grid-template-columns:1fr;grid-template-rows:65vh 1fr;overflow:auto}.graph-pane{min-height:65vh}aside{border-left:0;border-top:1px solid var(--line)}}
</style></head><body><header><div><h1>Interactive Graphiti graph: __TITLE__</h1><p id="summary"></p></div><div>Offline exact-topology viewer</div></header>
<div class="layout"><div class="graph-pane"><div class="toolbar"><input id="search" type="search" placeholder="Search entities or relationship facts…"><label class="toggle"><input id="provenance" type="checkbox"> Show episodes and MENTIONS</label><label class="toggle"><input id="labels" type="checkbox" checked> Node labels</label><label class="toggle"><input id="edge-labels" type="checkbox" checked> Relationship names</label><button id="fit">Reset view</button></div>
<svg id="graph" viewBox="-950 -650 1900 1300"><defs><marker id="arrowSemantic" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#e07a24"/></marker><marker id="arrowMention" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#94a3b8"/></marker></defs><g id="viewport"></g></svg>
<div class="legend"><span class="dot" style="background:#fff7ed;border:2px solid #e07a24"></span>Entity <span class="dot" style="background:#e07a24"></span>RELATES_TO <span class="dot" style="background:#dbeafe;border:2px solid #2474b5"></span>Episode <span class="dot" style="background:#94a3b8"></span>MENTIONS</div></div>
<aside><h2 id="detail-title">How to read this view</h2><div id="detail"><p>The default view contains only semantic entity-to-entity relationships. Click a node to isolate its one-hop neighborhood and inspect complete facts. Click empty space to clear it.</p><p class="hint">Episodes are batches of source statements. Enable provenance to see which episode mentioned each entity.</p></div></aside></div>
<script id="graph-data" type="application/json">__GRAPH_DATA__</script><script>
const data=JSON.parse(document.getElementById('graph-data').textContent),svg=document.getElementById('graph'),viewport=document.getElementById('viewport'),NS='http://www.w3.org/2000/svg';
const nodeById=new Map(data.nodes.map(n=>[n.element_id,n])),positions=new Map();let selected=null,transform={x:0,y:0,k:1},drag=null;
const isEpisode=n=>n.labels.includes('Episodic'), name=n=>n.properties.name||n.properties.uuid||(isEpisode(n)?'Episode':'Entity'), predicate=e=>e.properties.name||e.type;
function computeLayout(){const entities=data.nodes.filter(n=>!isEpisode(n)),episodes=data.nodes.filter(isEpisode),degree=new Map(entities.map(n=>[n.element_id,0]));data.relationships.filter(e=>e.type==='RELATES_TO').forEach(e=>{degree.set(e.source_element_id,(degree.get(e.source_element_id)||0)+1);degree.set(e.target_element_id,(degree.get(e.target_element_id)||0)+1)});entities.sort((a,b)=>(degree.get(b.element_id)||0)-(degree.get(a.element_id)||0)||name(a).localeCompare(name(b)));if(entities.length){positions.set(entities[0].element_id,{x:0,y:0});let offset=1,ring=1;while(offset<entities.length){const count=Math.min(10+ring*6,entities.length-offset),radius=235+ring*185;for(let i=0;i<count;i++){const angle=-Math.PI/2+2*Math.PI*i/count;positions.set(entities[offset+i].element_id,{x:radius*Math.cos(angle),y:radius*Math.sin(angle)})}offset+=count;ring++}const episodeRadius=235+ring*185+170;episodes.forEach((n,i)=>{const a=-Math.PI/2+2*Math.PI*i/Math.max(1,episodes.length);positions.set(n.element_id,{x:episodeRadius*Math.cos(a),y:episodeRadius*Math.sin(a)})})}}
function el(tag,attrs={}){const x=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>x.setAttribute(k,v));return x}function connectedIds(id){const out=new Set([id]);data.relationships.forEach(e=>{if(e.source_element_id===id)out.add(e.target_element_id);if(e.target_element_id===id)out.add(e.source_element_id)});return out}
function showNode(n){selected=n.element_id;render();document.getElementById('detail-title').textContent=name(n);const related=data.relationships.filter(e=>e.source_element_id===n.element_id||e.target_element_id===n.element_id);const facts=related.filter(e=>e.type==='RELATES_TO').map(e=>'<div class="fact">'+escapeHtml(e.properties.fact||'RELATES_TO')+'</div>').join('')||'<p class="hint">No semantic facts.</p>';document.getElementById('detail').innerHTML='<h3>Properties</h3><pre>'+escapeHtml(JSON.stringify(n.properties,null,2))+'</pre><h3>Connected semantic facts</h3>'+facts+'<h3>Topology</h3><p>'+related.length+' total adjacent edges.</p>'}
function showEdge(e){document.getElementById('detail-title').textContent=predicate(e);const fact=e.properties.fact||'This provenance edge records an entity mention.';document.getElementById('detail').innerHTML='<p><b>'+escapeHtml(name(nodeById.get(e.source_element_id)))+'</b> → <b>'+escapeHtml(name(nodeById.get(e.target_element_id)))+'</b></p><p class="hint">Neo4j relationship type: <code>'+escapeHtml(e.type)+'</code></p><h3>Relationship fact</h3><div class="fact">'+escapeHtml(fact)+'</div><h3>All stored properties</h3><pre>'+escapeHtml(JSON.stringify(e.properties,null,2))+'</pre>'}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function render(){viewport.replaceChildren();const provenance=document.getElementById('provenance').checked,showLabels=document.getElementById('labels').checked,showEdgeLabels=document.getElementById('edge-labels').checked,q=document.getElementById('search').value.trim().toLowerCase(),neighbors=selected?connectedIds(selected):null;const visibleNodes=data.nodes.filter(n=>provenance||!isEpisode(n)),visibleIds=new Set(visibleNodes.map(n=>n.element_id));const edges=data.relationships.filter(e=>visibleIds.has(e.source_element_id)&&visibleIds.has(e.target_element_id)&&(provenance||e.type==='RELATES_TO'));const matches=new Set();if(q){visibleNodes.forEach(n=>{if((name(n)+' '+JSON.stringify(n.properties)).toLowerCase().includes(q))matches.add(n.element_id)});edges.forEach(e=>{if(JSON.stringify(e.properties).toLowerCase().includes(q)){matches.add(e.source_element_id);matches.add(e.target_element_id)}})}
edges.forEach(e=>{const source=positions.get(e.source_element_id),target=positions.get(e.target_element_id),dx=target.x-source.x,dy=target.y-source.y,length=Math.max(1,Math.hypot(dx,dy)),sourceRadius=isEpisode(nodeById.get(e.source_element_id))?36:32,targetRadius=isEpisode(nodeById.get(e.target_element_id))?42:38,a={x:source.x+dx*sourceRadius/length,y:source.y+dy*sourceRadius/length},b={x:target.x-dx*targetRadius/length,y:target.y-dy*targetRadius/length},dim=(q&&!matches.has(e.source_element_id)&&!matches.has(e.target_element_id))||(neighbors&&!neighbors.has(e.source_element_id)&&!neighbors.has(e.target_element_id));const line=el('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:(e.type==='MENTIONS'?'mentions':'semantic')+(dim?' dim':''),'marker-end':e.type==='MENTIONS'?'url(#arrowMention)':'url(#arrowSemantic)'});const title=el('title');title.textContent=predicate(e)+': '+(e.properties.fact||'provenance link');line.append(title);viewport.append(line);const hit=el('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'edge-hit'+(dim?' dim':'')});hit.onclick=ev=>{ev.stopPropagation();showEdge(e)};viewport.append(hit);if(showEdgeLabels&&e.type==='RELATES_TO'){const label=el('text',{x:(a.x+b.x)/2,y:(a.y+b.y)/2-6,class:'edge-label'+(dim?' dim':'')});label.textContent=predicate(e).replaceAll('_',' ');viewport.append(label)}});
visibleNodes.forEach((n,index)=>{const p=positions.get(n.element_id),episode=isEpisode(n),dim=(q&&!matches.has(n.element_id))||(neighbors&&!neighbors.has(n.element_id)),g=el('g',{class:'node'+(episode?' episode':'')+(dim?' dim':'')+(selected===n.element_id?' selected':neighbors&&neighbors.has(n.element_id)?' neighbor':'')});if(episode)g.append(el('rect',{x:p.x-30,y:p.y-24,width:60,height:48,rx:9}));else g.append(el('circle',{cx:p.x,cy:p.y,r:selected===n.element_id?37:30}));const title=el('title');title.textContent=name(n);g.append(title);if(showLabels){const t=el('text',{x:p.x,y:p.y+50});let label=episode?'Episode '+(data.nodes.filter(isEpisode).indexOf(n)+1):name(n);t.textContent=label.length>34?label.slice(0,31)+'…':label;g.append(t)}g.onclick=ev=>{ev.stopPropagation();showNode(n)};viewport.append(g)});viewport.setAttribute('transform',`translate(${transform.x} ${transform.y}) scale(${transform.k})`)}
function reset(){selected=null;transform={x:0,y:0,k:1};document.getElementById('detail-title').textContent='How to read this view';document.getElementById('detail').innerHTML='<p>The default view contains only semantic entity-to-entity relationships. Click a node to isolate its one-hop neighborhood and inspect complete facts. Click empty space to clear it.</p><p class="hint">Episodes are batches of source statements. Enable provenance to see which episode mentioned each entity.</p>';render()}
svg.addEventListener('wheel',e=>{e.preventDefault();transform.k=Math.max(.25,Math.min(4,transform.k*(e.deltaY<0?1.12:.89)));render()},{passive:false});svg.addEventListener('pointerdown',e=>{if(e.target===svg){drag={x:e.clientX,y:e.clientY,tx:transform.x,ty:transform.y};svg.setPointerCapture(e.pointerId);svg.classList.add('dragging')}});svg.addEventListener('pointermove',e=>{if(drag){transform.x=drag.tx+(e.clientX-drag.x)/transform.k;transform.y=drag.ty+(e.clientY-drag.y)/transform.k;render()}});svg.addEventListener('pointerup',()=>{drag=null;svg.classList.remove('dragging')});svg.addEventListener('click',e=>{if(e.target===svg)reset()});
document.getElementById('search').addEventListener('input',render);document.getElementById('provenance').addEventListener('change',render);document.getElementById('labels').addEventListener('change',render);document.getElementById('edge-labels').addEventListener('change',render);document.getElementById('fit').onclick=reset;document.getElementById('summary').textContent=data.nodes.filter(n=>!isEpisode(n)).length+' entities · '+data.nodes.filter(isEpisode).length+' episodes · '+data.relationships.filter(e=>e.type==='RELATES_TO').length+' semantic relationships';computeLayout();render();
</script></body></html>'''
    return template.replace("__TITLE__", html.escape(persona_name)).replace("__GRAPH_DATA__", graph_json)


def _retrieval_svg(snapshot: dict[str, Any], persona_name: str, context_idx: int, recipient: str) -> str:
    selected = _selected(snapshot)
    width = 1500
    height = max(360, 185 + 105 * len(selected))
    cards = []
    for index, fact in enumerate(selected):
        y = 155 + index * 105
        wrapped = [fact[pos : pos + 105] for pos in range(0, len(fact), 105)] or [""]
        lines = "".join(
            f'<text x="95" y="{y + 38 + line_index * 23}" font-family="sans-serif" font-size="18" fill="#172033">{html.escape(line)}</text>'
            for line_index, line in enumerate(wrapped[:3])
        )
        cards.append(f'<rect x="70" y="{y}" width="1360" height="85" rx="12" fill="#ecfdf5" stroke="#3a9855" stroke-width="2"/>{lines}')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>
<text x="70" y="55" font-family="sans-serif" font-size="30" font-weight="700" fill="#172033">Context {context_idx}: {html.escape(recipient)}</text>
<text x="70" y="95" font-family="sans-serif" font-size="18" fill="#475569">{html.escape(persona_name)} · {_candidate_count(snapshot) or 0} returned candidates · {len(selected)} selected after reranking</text>
{''.join(cards)}</svg>'''


def _snapshot(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(record.get("graph_memory"), dict):
        source = record["graph_memory"]
        snapshot = {**source, "rerank": _clean_rerank(source.get("rerank"))}
        return "graph", snapshot
    if isinstance(record.get("profile_memory"), dict):
        source = record["profile_memory"]
        snapshot = {
            key: item for key, item in source.items()
            if key != "profile"
        }
        snapshot["rerank"] = _clean_rerank(source.get("rerank"))
        return "profile", snapshot
    if isinstance(record.get("rerank"), dict):
        return "list-rerank", _clean_rerank(record["rerank"])
    if isinstance(record.get("list_search"), dict):
        return "list-search", record["list_search"]
    for key in ("attacker_rag", "attacker_prompt_injection"):
        if isinstance(record.get(key), dict):
            return key, record[key]
    return "prompt-only", {"prompt": record.get("prompt")}


def _selected(snapshot: dict[str, Any]) -> list[str]:
    values = snapshot.get("selected_memories")
    if not isinstance(values, list) and isinstance(snapshot.get("rerank"), dict):
        values = snapshot["rerank"].get("selected_memories")
    if not isinstance(values, list):
        values = snapshot.get("hits") if isinstance(snapshot.get("hits"), list) else []
    return [str(value.get("fact") if isinstance(value, dict) and "fact" in value else value) for value in values]


def _candidate_count(snapshot: dict[str, Any]) -> int | None:
    value = snapshot.get("candidate_count") or snapshot.get("fact_count")
    if value is None and isinstance(snapshot.get("rerank"), dict):
        value = snapshot["rerank"].get("candidate_count")
    return int(value) if isinstance(value, (int, float)) else None


def _candidate_text(candidate: Any) -> str:
    if isinstance(candidate, dict):
        for key in ("fact", "memory", "text", "content"):
            value = candidate.get(key)
            if isinstance(value, str):
                return value
    return str(candidate) if candidate is not None else ""


def _candidates_from_rerank_prompt(prompt: Any) -> list[dict[str, Any]]:
    """Recover the exact list candidates embedded in an LLM-reranker prompt."""
    if not isinstance(prompt, str):
        return []
    marker = "Candidate memories:"
    marker_at = prompt.find(marker)
    if marker_at < 0:
        return []
    remainder = prompt[marker_at + len(marker):].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(remainder)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _pre_rerank_candidates(record: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Return architecture, normalized candidates, and completeness metadata."""
    source: dict[str, Any]
    raw: list[Any]
    architecture_level_dump = False
    if isinstance(record.get("graph_memory"), dict):
        architecture = "graph"
        source = record["graph_memory"]
        raw = source.get("facts") if isinstance(source.get("facts"), list) else []
    elif isinstance(record.get("profile_memory"), dict):
        architecture = "profile"
        source = record["profile_memory"]
        if isinstance(source.get("pre_rerank_memory_items"), list):
            # Mode 19 has profile facts that deliberately bypass the event
            # reranker. Its fair pre-rerank architecture output is therefore
            # the persistent profile plus every query-selected event.
            raw = source["pre_rerank_memory_items"]
            architecture_level_dump = True
        else:
            raw = source.get("candidates") if isinstance(source.get("candidates"), list) else []
    elif isinstance(record.get("rerank"), dict):
        architecture = "list"
        source = record["rerank"]
        raw = _candidates_from_rerank_prompt(source.get("prompt"))
        if not raw and isinstance(source.get("scores"), list):
            raw = source["scores"]
    else:
        raise ValueError("Saved record contains no reranked memory result")

    selected = set(_selected(source))
    candidates: list[dict[str, Any]] = []
    for fallback_index, item in enumerate(raw):
        details = dict(item) if isinstance(item, dict) else {"fact": item}
        text = _candidate_text(item)
        index = details.get("index", details.get("candidate_index", fallback_index))
        retained = details.get("retained_after_rerank")
        candidates.append(
            {
                "candidate_index": int(index) if isinstance(index, (int, float)) else fallback_index,
                "fact": text,
                "selected_after_rerank": (
                    retained if isinstance(retained, bool) else text in selected
                ),
                "provenance": {
                    key: value
                    for key, value in details.items()
                    if key not in {"index", "candidate_index", "fact", "memory", "text", "content"}
                },
            }
        )
    retrieval_options = source.get("retrieval_options")
    expected = (
        retrieval_options.get("pre_rerank_memory_item_count")
        if architecture_level_dump and isinstance(retrieval_options, dict)
        else source.get("candidate_count")
    )
    if not isinstance(expected, (int, float)) and isinstance(source.get("rerank"), dict):
        expected = source["rerank"].get("candidate_count")
    expected_count = int(expected) if isinstance(expected, (int, float)) else None
    complete = expected_count is not None and len(candidates) == expected_count
    candidate_limit = None if architecture_level_dump else source.get("candidate_limit")
    if (
        not architecture_level_dump
        and candidate_limit is None
        and isinstance(source.get("rerank"), dict)
    ):
        candidate_limit = source["rerank"].get("candidate_limit")
    return architecture, candidates, {
        "saved_candidate_count": len(candidates),
        "reported_candidate_count": expected_count,
        "complete_reranker_input": complete,
        "candidate_limit": candidate_limit,
        "candidate_limit_reached": (
            bool(candidate_limit and expected_count is not None and expected_count >= int(candidate_limit))
        ),
        "metric_scope": (
            "persistent_profile_plus_all_query_selected_events"
            if architecture_level_dump
            else "reranker_input_candidates"
        ),
        "recovery_source": (
            "graph_memory.facts"
            if architecture == "graph"
            else "profile_memory.pre_rerank_memory_items"
            if architecture_level_dump
            else "profile_memory.candidates"
            if architecture == "profile"
            else "rerank.prompt"
            if _candidates_from_rerank_prompt(source.get("prompt"))
            else "rerank.scores_partial_fallback"
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _architecture_svg(mode_name: str, source_count: int, backend_count: int | None) -> str:
    architecture = (
        "Atomic entries → all candidates → contextual reranker → selected memories"
        if "list" in mode_name
        else "Atomic statements → normalized episodes → Graphiti facts → contextual reranker"
        if "graph" in mode_name
        else "Atomic statements → categorized profile → atomic flattening → contextual reranker"
        if "profile" in mode_name
        else "Initialized memory → contextual retrieval → selected memories"
    )
    count = "unknown" if backend_count is None else str(backend_count)
    labels = architecture.split(" → ")
    width = 290 * len(labels)
    nodes = []
    edges = []
    for index, label in enumerate(labels):
        x = 25 + index * 290
        nodes.append(f'<rect x="{x}" y="75" width="230" height="90" rx="16" fill="#eff6ff" stroke="#2474b5" stroke-width="2"/>')
        nodes.append(f'<text x="{x + 115}" y="112" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="600" fill="#172033">{html.escape(label)}</text>')
        if index < len(labels) - 1:
            edges.append(f'<line x1="{x + 230}" y1="120" x2="{x + 280}" y2="120" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="230" viewBox="0 0 {width} 230">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#475569"/></marker></defs>
<rect width="100%" height="100%" fill="white"/><text x="25" y="35" font-family="sans-serif" font-size="24" font-weight="700" fill="#172033">{html.escape(mode_name)} memory flow</text>
{''.join(edges)}{''.join(nodes)}
<text x="25" y="205" font-family="sans-serif" font-size="14" fill="#475569">Source statements: {source_count}; saved backend/profile entries: {count}. Diagram describes the logged transformation stages.</text></svg>'''


def _memory_dump_html(
    *,
    persona_name: str,
    persona_idx: int,
    mode_name: str,
    backend: dict[str, Any],
    exact_data_href: str = "../initialization.json",
) -> str:
    """Create a portable, dependency-free view of a list or profile backend dump."""

    def scalar(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        if value is None:
            return "null"
        return str(value)

    def card(title: str, value: Any, *, badge: str = "") -> str:
        text = scalar(value)
        badge_html = f'<span class="badge">{html.escape(badge)}</span>' if badge else ""
        return (
            f'<article class="memory-card" data-search="{html.escape((title + " " + text).lower(), quote=True)}">'
            f'<div class="card-head"><h3>{html.escape(title)}</h3>{badge_html}</div>'
            f'<pre>{html.escape(text)}</pre></article>'
        )

    sections: list[str] = []
    item_count = 0
    if "atomic_entries" in backend:
        entries = backend.get("atomic_entries") or []
        item_count = len(entries)
        cards = "".join(
            card(
                f'Memory {int(entry.get("index", index)) + 1}',
                entry.get("memory_statement", ""),
                badge="atomic entry",
            )
            for index, entry in enumerate(entries)
            if isinstance(entry, dict)
        )
        sections.append(
            f'<section><div class="section-head"><h2>Atomic memory entries</h2><span>{item_count} entries</span></div>'
            f'<div class="card-grid">{cards}</div></section>'
        )
    else:
        profile = backend.get("profile")
        if isinstance(profile, dict):
            for category, values in profile.items():
                category_cards = []
                if isinstance(values, dict):
                    for key, value in values.items():
                        category_cards.append(card(str(key), value, badge=str(category)))
                elif isinstance(values, list):
                    for index, value in enumerate(values):
                        category_cards.append(card(f"Item {index + 1}", value, badge=str(category)))
                else:
                    category_cards.append(card(str(category), values, badge="profile field"))
                item_count += len(category_cards)
                sections.append(
                    f'<section><div class="section-head"><h2>{html.escape(str(category))}</h2>'
                    f'<span>{len(category_cards)} fields</span></div><div class="card-grid">{"".join(category_cards)}</div></section>'
                )
        else:
            sections.append(
                '<section><div class="empty">No consolidated profile object was saved in this run.</div></section>'
            )
        supplementary = {
            key: value for key, value in backend.items() if key != "profile" and value not in (None, {}, [])
        }
        if supplementary:
            sections.append(
                '<section><div class="section-head"><h2>Initialization and source data</h2>'
                f'<span>{len(supplementary)} groups</span></div><div class="card-grid">'
                + "".join(card(str(key), value, badge="metadata") for key, value in supplementary.items())
                + "</div></section>"
            )

    title = f"{mode_name} memory: {persona_name}"
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe4ee;--accent:#236fa1;--wash:#f4f8fb;--card:#fff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 "Segoe UI",Arial,sans-serif}}
header{{background:linear-gradient(120deg,#17365d,#236fa1);color:white;padding:2.5rem max(3vw,24px)}}
header h1{{font-size:clamp(1.7rem,3vw,2.6rem);margin:0 0 .4rem}} header p{{margin:.2rem 0;color:#dbeafe}}
.toolbar{{position:sticky;top:0;z-index:2;display:flex;gap:1rem;align-items:center;padding:1rem max(3vw,24px);background:#ffffffee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
input{{width:min(620px,100%);padding:.75rem 1rem;border:1px solid #b8c8d8;border-radius:10px;font:inherit}}
#count{{color:var(--muted);white-space:nowrap}} main{{padding:1rem max(3vw,24px) 3rem}}
section{{margin:1.7rem 0 2.5rem}} .section-head{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:2px solid #b8d4e7;margin-bottom:1rem}}
h2{{margin:0;color:#17365d}} .section-head span{{color:var(--muted)}} .card-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}}
.memory-card{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:1rem;box-shadow:0 3px 12px #17365d12}}
.card-head{{display:flex;gap:.7rem;align-items:center;justify-content:space-between}} h3{{margin:0;font-size:1rem;color:#17365d}}
.badge{{background:#e0f2fe;color:#075985;border-radius:999px;padding:.15rem .55rem;font-size:.75rem;white-space:nowrap}}
pre{{font:14px/1.5 "Cascadia Mono",Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:.8rem 0 0;color:#27364a}}
.empty{{padding:2rem;border:1px dashed #94a3b8;border-radius:12px;background:white;color:var(--muted)}}
.hidden{{display:none}} footer{{padding:1.5rem max(3vw,24px);background:white;border-top:1px solid var(--line);color:var(--muted)}}
@media print{{.toolbar{{display:none}} body{{background:white}} .memory-card{{box-shadow:none;break-inside:avoid}}}}
</style></head><body><header><h1>{html.escape(persona_name)}</h1>
<p>Persona {persona_idx} · {html.escape(mode_name)} · complete initialized memory dump · {item_count} primary entries</p></header>
<div class="toolbar"><input id="search" type="search" placeholder="Search keys and memory contents…" autofocus><span id="count"></span></div>
<main>{''.join(sections)}</main><footer>Self-contained offline export. Exact machine-readable data remains in <code>{html.escape(exact_data_href)}</code>.</footer>
<script>
const cards=[...document.querySelectorAll('.memory-card')], input=document.getElementById('search'), count=document.getElementById('count');
function filter(){{const q=input.value.trim().toLowerCase();let shown=0;cards.forEach(c=>{{const visible=!q||c.dataset.search.includes(q);c.classList.toggle('hidden',!visible);if(visible)shown++;}});count.textContent=shown+' of '+cards.length+' cards';}}
input.addEventListener('input',filter);filter();
</script></body></html>'''


def _html_page(title: str, mode_name: str, personas: list[dict[str, Any]]) -> str:
    sections = []
    for persona in personas:
        rows = []
        for context in persona["contexts"]:
            memories = "".join(f"<li>{html.escape(item)}</li>" for item in context["selected_memories"])
            rows.append(
                "<tr>"
                f"<td>{context['context_idx']}</td><td>{html.escape(context['recipient'])}</td>"
                f"<td>{html.escape(context['task'])}</td><td>{context['candidate_count'] if context['candidate_count'] is not None else 'n/a'}</td>"
                f"<td>{len(context['selected_memories'])}</td><td>{context['repeat_count']}</td>"
                f"<td><details><summary>View</summary><ul>{memories}</ul><a href=\"{html.escape(context['snapshot_path'])}\">Exact JSON</a>"
                + (f" · <a href=\"{html.escape(context['visualization_path'])}\">Retrieval SVG</a>" if context.get("visualization_path") else "")
                + "</details></td></tr>"
            )
        graph_links = ""
        if persona.get("full_graph_directory"):
            graph_dir = html.escape(persona["full_graph_directory"])
            graph_links = (
                f' · <a href="{graph_dir}/full_graph.html">Interactive full graph</a>'
                f' · <a href="{graph_dir}/full_graph.svg">Static SVG</a>'
                f' · <a href="{graph_dir}/full_graph.graphml">GraphML</a>'
                f' · <a href="{graph_dir}/full_graph.json">Graph JSON</a>'
            )
        memory_dump_link = ""
        if persona.get("memory_dump_path"):
            memory_dump_link = (
                f' · <a href="{html.escape(persona["memory_dump_path"])}">Formatted memory dump</a>'
            )
        sections.append(
            f"<section><h2>Persona {persona['persona_idx']}: {html.escape(persona['persona_name'])}</h2>"
            f"<p><a href=\"{html.escape(persona['initialization_path'])}\">Initialization snapshot</a> · "
            f"{persona['source_statement_count']} source statements · {len(persona['contexts'])} scenarios{memory_dump_link}{graph_links}</p>"
            "<table><thead><tr><th>#</th><th>Recipient</th><th>Task</th><th>Candidates</th><th>Selected</th><th>Repeats</th><th>Snapshot</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>
body{{font:15px system-ui,sans-serif;color:#172033;margin:2rem;max-width:1500px}} h1,h2{{color:#17365d}} section{{margin:2.5rem 0}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #cbd5e1;padding:.45rem;vertical-align:top}} th{{background:#eaf2f8;text-align:left}} tr:nth-child(even){{background:#f8fafc}} details{{min-width:9rem}} li{{margin:.3rem 0}} code{{background:#f1f5f9;padding:.15rem .3rem}}
</style></head><body><h1>{html.escape(title)}</h1><p>Architecture: <strong>{html.escape(mode_name)}</strong>. Each scenario is exported once because all response repetitions reuse the same prepared memory snapshot.</p><p><a href="architecture_flow.svg">Architecture flow</a> · <a href="manifest.json">Provenance manifest</a> · <a href="contexts.csv">Context index CSV</a></p>{''.join(sections)}</body></html>'''


def export_memory_snapshots(
    pipeline_path: str,
    *,
    output_dir: str | None = None,
    export_full_graph: bool = False,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> Path:
    source_manifest, pipeline_manifests, source = _pipeline_manifests(Path(pipeline_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"memory-snapshots_{source_manifest.parent.name}_{timestamp}"
    output = Path(output_dir).expanduser() if output_dir else Path("research_outputs") / default_name
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    exported: list[dict[str, Any]] = []
    csv_rows: list[list[Any]] = []
    mode_names: set[str] = set()
    source_paths: list[str] = []
    full_graph_exports: list[str] = []
    for pipeline_manifest in pipeline_manifests:
        pipeline = _read_json(pipeline_manifest)
        query_manifest = _query_manifest(pipeline_manifest, pipeline)
        query = _read_json(query_manifest)
        rows = _records(query, query_manifest)
        first = rows[0]
        persona_idx = int(query.get("user_idx", pipeline.get("persona_idx", first.get("user_idx", 0))))
        persona_name = str(query.get("persona_name") or pipeline.get("persona_name") or first.get("persona_name") or f"Persona {persona_idx}")
        mode = int(query.get("memory_mode", pipeline.get("memory_mode", first.get("memory_mode", -1))))
        mode_name = str(query.get("memory_mode_name") or pipeline.get("memory_mode_name") or f"mode-{mode}")
        mode_names.add(mode_name)
        source_paths.append(str(query_manifest.parent))

        dataset = _dataset_path(query, pipeline, query_manifest)
        statements = list(first.get("attributes") or [])
        if dataset is not None:
            statements = extract_persona_memory_statements(load_persona_entry(str(dataset), persona_idx))

        persona_dir = output / "personas" / f"persona_{persona_idx:03d}"
        contexts_dir = persona_dir / "contexts"
        contexts_dir.mkdir(parents=True)
        backend: dict[str, Any] = {}
        backend_count: int | None = None
        if "profile" in mode_name and isinstance(query.get("profile_memory"), dict):
            profile_config = query["profile_memory"]
            init = profile_config.get("init") or {}
            profile = init.get("profile")
            if profile is None:
                for row in rows:
                    if isinstance(row.get("profile_memory"), dict) and row["profile_memory"].get("profile"):
                        profile = row["profile_memory"]["profile"]
                        break
            backend = {
                "profile": profile,
                "inserted_statements": profile_config.get("inserted_statements", statements),
                "initialization": {key: value for key, value in init.items() if key != "profile"},
            }
            if isinstance(profile, dict):
                backend_count = sum(len(value) for value in profile.values() if isinstance(value, dict))
        elif "graph" in mode_name:
            graph_config = query.get("graph_memory") or {}
            backend = {
                "source_statements": statements,
                "graph_configuration": {key: value for key, value in graph_config.items() if key != "contexts"},
                "limitation": "The saved run contains returned facts and episode provenance, not a complete Neo4j entity/edge topology dump.",
            }
            backend_count = int((query.get("memory_initialization") or {}).get("memory_statements_inserted") or 0) or None
        else:
            backend = {"atomic_entries": [{"index": index, "memory_statement": value} for index, value in enumerate(statements)]}
            backend_count = len(statements)
        initialization = {
            "persona_idx": persona_idx,
            "persona_name": persona_name,
            "memory_mode": mode,
            "memory_mode_name": mode_name,
            "source_dataset": str(dataset) if dataset else None,
            "source_statement_count": len(statements),
            "memory_initialization": query.get("memory_initialization"),
            "backend_snapshot": backend,
        }
        _write_json(persona_dir / "initialization.json", initialization)

        memory_dump_path = None
        if "graph" not in mode_name:
            memory_dir = persona_dir / "memory"
            memory_dir.mkdir()
            dump_file = memory_dir / "full_memory.html"
            dump_file.write_text(
                _memory_dump_html(
                    persona_name=persona_name,
                    persona_idx=persona_idx,
                    mode_name=mode_name,
                    backend=backend,
                ),
                encoding="utf-8",
            )
            memory_dump_path = str(
                Path("personas") / f"persona_{persona_idx:03d}" / "memory" / "full_memory.html"
            )

        full_graph_directory = None
        if export_full_graph and "graph" in mode_name:
            graph_config = query.get("graph_memory") or {}
            group_id = graph_config.get("group_id") or (query.get("memory_initialization") or {}).get("graph_group_id")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"Graph run has no recorded group_id: {query_manifest}")
            if not all(isinstance(value, str) and value for value in (neo4j_uri, neo4j_user, neo4j_password)):
                raise ValueError("--export-full-graph requires configured Neo4j URI, user, and password")
            graph = _read_full_graph(
                group_id=group_id,
                neo4j_uri=str(neo4j_uri),
                neo4j_user=str(neo4j_user),
                neo4j_password=str(neo4j_password),
            )
            if not graph["nodes"]:
                raise ValueError(
                    f"No live Neo4j nodes found for recorded Graphiti group {group_id!r}"
                )
            graph_dir = persona_dir / "graph"
            graph_dir.mkdir()
            _write_json(graph_dir / "full_graph.json", graph)
            (graph_dir / "full_graph.graphml").write_text(_graphml(graph), encoding="utf-8")
            (graph_dir / "full_graph.svg").write_text(
                _full_graph_svg(graph, persona_name), encoding="utf-8"
            )
            (graph_dir / "full_graph.html").write_text(
                _full_graph_html(graph, persona_name), encoding="utf-8"
            )
            relationship_types: dict[str, int] = {}
            for relationship in graph["relationships"]:
                relationship_types[relationship["type"]] = relationship_types.get(relationship["type"], 0) + 1
            statistics = {
                "group_id": group_id,
                "node_count": len(graph["nodes"]),
                "entity_count": sum("Entity" in node["labels"] for node in graph["nodes"]),
                "episode_count": sum("Episodic" in node["labels"] for node in graph["nodes"]),
                "relationship_count": len(graph["relationships"]),
                "relationship_types": relationship_types,
                "vectors_omitted": True,
            }
            _write_json(graph_dir / "graph_statistics.json", statistics)
            full_graph_directory = str(Path("personas") / f"persona_{persona_idx:03d}" / "graph")
            full_graph_exports.append(full_graph_directory)

        by_context: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_context.setdefault(int(row["context_idx"]), []).append(row)
        context_summaries = []
        for context_idx, repeats in sorted(by_context.items()):
            architecture, snapshot = _snapshot(repeats[0])
            selected = _selected(snapshot)
            context_payload = {
                "persona_idx": persona_idx,
                "persona_name": persona_name,
                "memory_mode": mode,
                "memory_mode_name": mode_name,
                "context_idx": context_idx,
                "recipient": str(repeats[0].get("recipient") or ""),
                "task": str(repeats[0].get("task") or ""),
                "repeat_count": len(repeats),
                "shared_across_repeats": True,
                "snapshot_type": architecture,
                "selected_memories": selected,
                "retrieval": snapshot,
            }
            relative = Path("personas") / f"persona_{persona_idx:03d}" / "contexts" / f"context_{context_idx:03d}.json"
            _write_json(output / relative, context_payload)
            visualization_path = None
            if export_full_graph and "graph" in mode_name:
                svg_relative = relative.with_suffix(".svg")
                (output / svg_relative).write_text(
                    _retrieval_svg(
                        snapshot,
                        persona_name,
                        context_idx,
                        context_payload["recipient"],
                    ),
                    encoding="utf-8",
                )
                visualization_path = str(svg_relative)
            summary = {
                "context_idx": context_idx,
                "recipient": context_payload["recipient"],
                "task": context_payload["task"],
                "candidate_count": _candidate_count(snapshot),
                "selected_memories": selected,
                "repeat_count": len(repeats),
                "snapshot_path": str(relative),
                "visualization_path": visualization_path,
            }
            context_summaries.append(summary)
            csv_rows.append([persona_idx, persona_name, mode, mode_name, context_idx, summary["recipient"], summary["task"], summary["candidate_count"], len(selected), len(repeats), str(relative)])
        exported.append({
            "persona_idx": persona_idx,
            "persona_name": persona_name,
            "source_statement_count": len(statements),
            "initialization_path": str(Path("personas") / f"persona_{persona_idx:03d}" / "initialization.json"),
            "memory_dump_path": memory_dump_path,
            "full_graph_directory": full_graph_directory,
            "contexts": context_summaries,
        })

    if len(mode_names) != 1:
        raise ValueError(f"Pipeline contains inconsistent memory modes: {sorted(mode_names)}")
    mode_name = next(iter(mode_names))
    if export_full_graph and "graph" not in mode_name:
        raise ValueError("--export-full-graph is only valid for graph-memory pipeline runs")
    with (output / "contexts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["persona_idx", "persona", "mode", "mode_name", "context_idx", "recipient", "task", "candidate_count", "selected_count", "repeat_count", "snapshot_path"])
        writer.writerows(csv_rows)
    backend_counts = []
    for persona in exported:
        init = _read_json(output / persona["initialization_path"])
        backend_snapshot = init.get("backend_snapshot") or {}
        if "atomic_entries" in backend_snapshot:
            backend_counts.append(len(backend_snapshot["atomic_entries"]))
        elif isinstance(backend_snapshot.get("profile"), dict):
            backend_counts.append(sum(len(value) for value in backend_snapshot["profile"].values() if isinstance(value, dict)))
    (output / "architecture_flow.svg").write_text(
        _architecture_svg(mode_name, sum(p["source_statement_count"] for p in exported), sum(backend_counts) if backend_counts else None),
        encoding="utf-8",
    )
    manifest = {
        "command": "export_memory_snapshots",
        "created_at": timestamp,
        "source_pipeline": str(source_manifest.parent),
        "source_query_runs": source_paths,
        "memory_mode_name": mode_name,
        "persona_count": len(exported),
        "context_snapshot_count": len(csv_rows),
        "full_graph_exported": export_full_graph,
        "full_graph_directories": full_graph_exports,
        "snapshot_semantics": "One prepared memory snapshot per persona/context; shared by all saved response repetitions. Provider raw responses and duplicate reranker prompts are omitted, while memory candidates, facts, provenance, scores, and selections are retained.",
        "artifacts": ["index.html", "architecture_flow.svg", "contexts.csv", "personas/*/initialization.json", "personas/*/contexts/context_*.json", *( ["personas/*/memory/full_memory.html"] if "graph" not in mode_name else []), *( ["personas/*/contexts/context_*.svg", "personas/*/graph/full_graph.html", "personas/*/graph/full_graph.json", "personas/*/graph/full_graph.graphml", "personas/*/graph/full_graph.svg", "personas/*/graph/graph_statistics.json"] if export_full_graph else [])],
    }
    _write_json(output / "manifest.json", manifest)
    title = f"Memory snapshots: {mode_name}"
    (output / "index.html").write_text(_html_page(title, mode_name, exported), encoding="utf-8")
    (output / "README.md").write_text(
        f"# {title}\n\nOpen `index.html` for the browsable report. The export contains {len(exported)} personas and {len(csv_rows)} scenario snapshots. Each context snapshot is stored once and records how many response repetitions reused it."
        + (" Each persona also has a self-contained, searchable HTML rendering of its complete initialized list or profile memory." if "graph" not in mode_name else "")
        + (" Full live Graphiti topology is exported per persona as an interactive offline HTML viewer, JSON, GraphML, and SVG; compact retrieval SVGs are exported per scenario." if export_full_graph else "")
        + " See `manifest.json` for provenance.\n",
        encoding="utf-8",
    )
    return output


def _pre_rerank_context_html(payload: dict[str, Any]) -> str:
    status = payload["candidate_set"]
    architecture_scope = (
        status.get("metric_scope")
        == "persistent_profile_plus_all_query_selected_events"
    )
    item_label = "raw memory items" if architecture_scope else "candidates"
    cards = []
    for candidate in payload["candidates"]:
        selected = bool(candidate["selected_after_rerank"])
        provenance = candidate.get("provenance") or {}
        metadata = html.escape(json.dumps(provenance, ensure_ascii=False, indent=2))
        fact = html.escape(candidate["fact"])
        search = html.escape(
            f"{candidate['fact']} {json.dumps(provenance, ensure_ascii=False)}".lower(),
            quote=True,
        )
        cards.append(
            f'<article class="candidate {"selected" if selected else ""}" data-search="{search}" '
            f'data-selected="{str(selected).lower()}"><div class="head"><span>#{candidate["candidate_index"]}</span>'
            f'<span class="badge">{"selected" if selected else "not selected"}</span></div>'
            f'<p>{fact}</p><details><summary>Provenance</summary><pre>{metadata}</pre></details></article>'
        )
    warning = "" if status["complete_reranker_input"] else (
        '<div class="warning">This older artifact does not contain every reported pre-rerank candidate. '
        'The export includes every candidate that can be recovered, and marks the set incomplete.</div>'
    )
    title = f'Context {payload["context_idx"]}: {payload["recipient"]}'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe4ee;--blue:#246b9e;--green:#23804a;--wash:#f4f8fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.5 "Segoe UI",Arial,sans-serif}}
header{{padding:2.2rem max(3vw,24px);color:white;background:linear-gradient(120deg,#17365d,#246b9e)}}h1{{margin:0 0 .4rem}}header p{{margin:.2rem 0;color:#dbeafe}}
.toolbar{{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;gap:.8rem;align-items:center;padding:1rem max(3vw,24px);background:#fffffff2;border-bottom:1px solid var(--line)}}
input[type=search]{{flex:1;min-width:260px;padding:.7rem 1rem;border:1px solid #a9bed0;border-radius:10px;font:inherit}}label{{color:var(--muted)}}main{{padding:1.2rem max(3vw,24px) 3rem}}
.warning{{padding:1rem;border:1px solid #e6a43a;background:#fff7df;border-radius:10px;margin-bottom:1rem}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}}
.candidate{{background:white;border:1px solid var(--line);border-left:5px solid #94a3b8;border-radius:12px;padding:1rem;box-shadow:0 3px 12px #17365d12}}.candidate.selected{{border-left-color:var(--green);background:#f3fff7}}
.head{{display:flex;justify-content:space-between;color:var(--muted)}}.badge{{padding:.12rem .5rem;border-radius:999px;background:#e8eef4;font-size:.76rem}}.selected .badge{{background:#d9f5e3;color:#176334}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.4 Consolas,monospace}}.hidden{{display:none}}@media print{{.toolbar{{display:none}}.candidate{{break-inside:avoid;box-shadow:none}}body{{background:white}}}}
</style></head><body><header><h1>{html.escape(title)}</h1><p>{html.escape(payload["persona_name"])} · {html.escape(payload["memory_mode_name"])} · {len(payload["candidates"])} {item_label} before reranking</p><p>{html.escape(payload["task"])}</p></header>
<div class="toolbar"><input id="search" type="search" placeholder="Search candidate facts and provenance…" autofocus><label><input id="selected" type="checkbox"> selected only</label><span id="count"></span></div>
<main>{warning}<div class="grid">{''.join(cards)}</div></main><script>
const cards=[...document.querySelectorAll('.candidate')],search=document.getElementById('search'),only=document.getElementById('selected'),count=document.getElementById('count');function filter(){{const q=search.value.trim().toLowerCase();let n=0;cards.forEach(c=>{{const show=(!q||c.dataset.search.includes(q))&&(!only.checked||c.dataset.selected==='true');c.classList.toggle('hidden',!show);if(show)n++}});count.textContent=n+' of '+cards.length+' {item_label}'}}search.oninput=filter;only.onchange=filter;filter();
</script></body></html>'''


def _pre_rerank_index_html(mode_name: str, personas: list[dict[str, Any]]) -> str:
    sections = []
    for persona in personas:
        rows = []
        for context in persona["contexts"]:
            status = "complete" if context["complete"] else "partial"
            cap = "yes" if context["candidate_limit_reached"] else "no"
            rows.append(
                f'<tr><td>{context["context_idx"]}</td><td>{html.escape(context["recipient"])}</td>'
                f'<td>{html.escape(context["task"])}</td><td>{context["candidate_count"]}</td>'
                f'<td>{context["selected_count"]}</td><td>{cap}</td><td class="{status}">{status}</td>'
                f'<td><a href="{html.escape(context["html_path"])}">Browse</a> · '
                f'<a href="{html.escape(context["json_path"])}">JSON</a></td></tr>'
            )
        sections.append(
            f'<section><h2>Persona {persona["persona_idx"]}: {html.escape(persona["persona_name"])}</h2>'
            '<table><thead><tr><th>#</th><th>Recipient</th><th>Task</th><th>Candidates</th><th>Selected</th><th>Cap reached</th><th>Recovery</th><th>Files</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></section>'
        )
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pre-rerank candidates</title><style>
body{{font:15px/1.5 system-ui,sans-serif;color:#172033;margin:2rem;max-width:1600px}}h1,h2{{color:#17365d}}section{{margin:2.5rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cbd5e1;padding:.5rem;vertical-align:top}}th{{background:#eaf2f8;text-align:left}}tr:nth-child(even){{background:#f8fafc}}.complete{{color:#176334}}.partial{{color:#9a5b00;font-weight:700}}
</style></head><body><h1>Pre-rerank candidate export</h1><p>Architecture: <strong>{html.escape(mode_name)}</strong>. These are the saved facts available immediately before the final reranking stage. Each scenario appears once because repetitions reuse the same candidate set.</p><p><a href="contexts.csv">Context summary CSV</a> · <a href="candidates.csv">All candidates CSV</a> · <a href="candidates.jsonl">All candidates JSONL</a> · <a href="manifest.json">Provenance manifest</a></p>{''.join(sections)}</body></html>'''


def export_pre_rerank_candidates(pipeline_path: str, *, output_dir: str | None = None) -> Path:
    """Export the exact saved candidate facts immediately before reranking."""
    source_manifest, pipeline_manifests, _ = _pipeline_manifests(Path(pipeline_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"pre-rerank-candidates_{source_manifest.parent.name}_{timestamp}"
    output = (Path(output_dir).expanduser() if output_dir else Path("research_outputs") / default_name).resolve()
    output.mkdir(parents=True, exist_ok=False)

    personas: list[dict[str, Any]] = []
    context_rows: list[list[Any]] = []
    candidate_rows: list[list[Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    mode_names: set[str] = set()
    incomplete: list[str] = []
    source_runs: list[str] = []
    for pipeline_manifest in pipeline_manifests:
        pipeline = _read_json(pipeline_manifest)
        query_manifest = _query_manifest(pipeline_manifest, pipeline)
        query = _read_json(query_manifest)
        rows = _records(query, query_manifest)
        first = rows[0]
        persona_idx = int(query.get("user_idx", pipeline.get("persona_idx", first.get("user_idx", 0))))
        persona_name = str(query.get("persona_name") or pipeline.get("persona_name") or first.get("persona_name") or f"Persona {persona_idx}")
        mode = int(query.get("memory_mode", pipeline.get("memory_mode", first.get("memory_mode", -1))))
        mode_name = str(query.get("memory_mode_name") or pipeline.get("memory_mode_name") or f"mode-{mode}")
        mode_names.add(mode_name)
        source_runs.append(str(query_manifest.parent))
        by_context: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_context.setdefault(int(row["context_idx"]), []).append(row)
        contexts = []
        for context_idx, repeats in sorted(by_context.items()):
            record = repeats[0]
            architecture, candidates, status = _pre_rerank_candidates(record)
            if not status["complete_reranker_input"]:
                incomplete.append(f"persona_{persona_idx:03d}/context_{context_idx:03d}")
            recipient = str(record.get("recipient") or "")
            task = str(record.get("task") or "")
            payload = {
                "persona_idx": persona_idx,
                "persona_name": persona_name,
                "memory_mode": mode,
                "memory_mode_name": mode_name,
                "architecture": architecture,
                "context_idx": context_idx,
                "recipient": recipient,
                "task": task,
                "repeat_count": len(repeats),
                "shared_across_repeats": True,
                "candidate_set": status,
                "candidates": candidates,
            }
            relative = Path("personas") / f"persona_{persona_idx:03d}" / "contexts" / f"context_{context_idx:03d}"
            (output / relative.parent).mkdir(parents=True, exist_ok=True)
            _write_json(output / relative.with_suffix(".json"), payload)
            (output / relative.with_suffix(".html")).write_text(_pre_rerank_context_html(payload), encoding="utf-8")
            selected_count = sum(bool(item["selected_after_rerank"]) for item in candidates)
            context_rows.append([persona_idx, persona_name, mode, mode_name, context_idx, recipient, task, len(candidates), selected_count, status["candidate_limit"], status["candidate_limit_reached"], status["complete_reranker_input"], status["recovery_source"]])
            for candidate in candidates:
                row = {
                    "persona_idx": persona_idx,
                    "persona_name": persona_name,
                    "memory_mode": mode,
                    "memory_mode_name": mode_name,
                    "context_idx": context_idx,
                    "recipient": recipient,
                    "task": task,
                    **candidate,
                }
                jsonl_rows.append(row)
                candidate_rows.append([persona_idx, persona_name, mode, mode_name, context_idx, recipient, task, candidate["candidate_index"], candidate["fact"], candidate["selected_after_rerank"], json.dumps(candidate["provenance"], ensure_ascii=False)])
            contexts.append({
                "context_idx": context_idx,
                "recipient": recipient,
                "task": task,
                "candidate_count": len(candidates),
                "selected_count": selected_count,
                "candidate_limit_reached": status["candidate_limit_reached"],
                "complete": status["complete_reranker_input"],
                "json_path": str(relative.with_suffix(".json")),
                "html_path": str(relative.with_suffix(".html")),
            })
        personas.append({"persona_idx": persona_idx, "persona_name": persona_name, "contexts": contexts})

    if len(mode_names) != 1:
        raise ValueError(f"Pipeline contains inconsistent memory modes: {sorted(mode_names)}")
    mode_name = next(iter(mode_names))
    with (output / "contexts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["persona_idx", "persona", "mode", "mode_name", "context_idx", "recipient", "task", "candidate_count", "selected_count", "candidate_limit", "candidate_limit_reached", "complete_reranker_input", "recovery_source"])
        writer.writerows(context_rows)
    with (output / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["persona_idx", "persona", "mode", "mode_name", "context_idx", "recipient", "task", "candidate_index", "fact", "selected_after_rerank", "provenance_json"])
        writer.writerows(candidate_rows)
    (output / "candidates.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in jsonl_rows), encoding="utf-8")
    manifest = {
        "command": "export_pre_rerank_candidates",
        "created_at": timestamp,
        "source_pipeline": str(source_manifest.parent),
        "source_query_runs": source_runs,
        "memory_mode_name": mode_name,
        "persona_count": len(personas),
        "context_count": len(context_rows),
        "candidate_occurrence_count": len(candidate_rows),
        "all_candidate_sets_complete": not incomplete,
        "incomplete_candidate_sets": incomplete,
        "semantics": "Candidates are the saved facts immediately before contextual reranking, not the complete initialized memory or hidden backend-internal search state.",
        "artifacts": ["index.html", "contexts.csv", "candidates.csv", "candidates.jsonl", "personas/*/contexts/context_*.json", "personas/*/contexts/context_*.html"],
    }
    _write_json(output / "manifest.json", manifest)
    (output / "index.html").write_text(_pre_rerank_index_html(mode_name, personas), encoding="utf-8")
    (output / "README.md").write_text(
        f"# Pre-rerank candidates: {mode_name}\n\nOpen `index.html` to browse {len(candidate_rows)} candidate occurrences across {len(context_rows)} persona/scenario sets. `candidates.csv` and `candidates.jsonl` provide analysis-ready long-form exports. Candidate sets that reached the configured cap are marked; this indicates that additional backend candidates may have existed before the cap.\n",
        encoding="utf-8",
    )
    return output
