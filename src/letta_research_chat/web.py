from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import threading
import traceback
from typing import Any, Callable
import uuid

import requests

from .attacker_rag import AttackerRagStore

try:
    from flask import Flask, Response, jsonify, request
except ImportError as exc:  # pragma: no cover - exercised only without the gui extra installed.
    Flask = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]
    request = None  # type: ignore[assignment]
    _FLASK_IMPORT_ERROR = exc
else:
    _FLASK_IMPORT_ERROR = None

from .agent import AgentClient
from .cli import (
    agent_llm_config_kwargs,
    compare_privacy_pipeline_cimemories_multi_reports,
    compare_privacy_pipeline_cimemories_reports,
    create_conversation_or_raise,
    extract_assistant_reply,
    format_all_archival_memories,
    format_conversation_history,
    init_archival_from_persona_file,
    parse_memory_mode,
    print_privacy_pipeline_cimemories_report,
    print_privacy_pipeline_cimemories_memory_queries,
    run_archival_search_limit_sweep_cimemories,
    run_compare_privacy_modes_cimemories,
    run_get_context_labeling_cimemories,
    run_privacy_pipeline_cimemories,
    run_query_recipient_with_task,
    run_query_recipients_with_tasks_experiment,
)
from .config import LettaConfig
from .conversations import ConversationClient
from .http import HttpClient
from .memory import MemoryClient


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Letta Research Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1c2430;
      --muted: #667085;
      --line: #d7dde6;
      --accent: #0b6bcb;
      --accent-dark: #064f99;
      --ok: #16794c;
      --warn: #9a5b00;
      --err: #b42318;
      --code: #101828;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
      min-height: 36px;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    button:hover { border-color: var(--accent); }
    button.primary:hover { background: var(--accent-dark); }
    button:disabled { opacity: .6; cursor: wait; }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      min-height: 36px;
    }
    textarea { resize: vertical; min-height: 76px; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: minmax(300px, 360px) minmax(420px, 1fr) minmax(340px, 430px);
      grid-template-rows: auto 1fr;
    }
    header {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    h2 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase; color: var(--muted); letter-spacing: 0; }
    .status {
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      overflow: hidden;
    }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--warn); flex: 0 0 auto; }
    .dot.ok { background: var(--ok); }
    .dot.err { background: var(--err); }
    .pane {
      min-height: 0;
      overflow: auto;
      padding: 14px;
      border-right: 1px solid var(--line);
    }
    .pane:last-child { border-right: 0; }
    .section {
      margin-bottom: 18px;
    }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }
    .label { display: block; color: var(--muted); font-size: 12px; margin: 8px 0 4px; }
    .chat {
      display: flex;
      flex-direction: column;
      min-height: 0;
      padding: 0;
      background: #eef2f6;
    }
    .messages {
      flex: 1;
      min-height: 0;
      overflow: auto;
      padding: 16px;
    }
    .composer {
      border-top: 1px solid var(--line);
      background: #fff;
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
    }
    .composer textarea { min-height: 52px; max-height: 160px; }
    .msg {
      max-width: 84%;
      margin: 0 0 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .msg.user { margin-left: auto; background: #e8f2ff; border-color: #b7d6fa; }
    .msg.assistant { margin-right: auto; }
    .msg.system { margin-right: auto; background: #fff8e5; border-color: #f0d795; }
    .msg .role { font-size: 11px; color: var(--muted); text-transform: uppercase; margin-bottom: 5px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    .list {
      display: grid;
      gap: 8px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fff;
      cursor: pointer;
    }
    .item:hover { border-color: var(--accent); }
    .item strong { display: block; font-size: 13px; margin-bottom: 4px; overflow-wrap: anywhere; }
    .item small { color: var(--muted); overflow-wrap: anywhere; }
    .badge {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      color: var(--muted);
      background: #fff;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
    }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric b { font-size: 18px; }
    .viewer {
      max-height: 42vh;
      overflow: auto;
      background: var(--code);
      color: #e4e7ec;
      border-radius: 8px;
      padding: 10px;
    }
    .chart {
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }
    .chart-row {
      display: grid;
      grid-template-columns: minmax(110px, 1fr) minmax(130px, 2fr) 60px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .bar-track {
      height: 10px;
      background: #e6ebf2;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar {
      height: 100%;
      width: 0%;
      background: var(--accent);
    }
    .bar.leak { background: var(--err); }
    .bar.ambiguous { background: var(--warn); }
    .tabs {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-bottom: 8px;
    }
    .tabs button.active {
      border-color: var(--accent);
      color: var(--accent);
      background: #eef6ff;
    }
    .markdown-preview {
      max-height: 52vh;
      overflow: auto;
      background: #fff;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .markdown-preview table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12px; }
    .markdown-preview th, .markdown-preview td { border: 1px solid var(--line); padding: 6px; text-align: left; vertical-align: top; }
    .markdown-preview th { background: #f2f5f9; }
    .markdown-preview h1 { font-size: 18px; margin: 0 0 10px; }
    .markdown-preview h2 { font-size: 15px; color: var(--text); text-transform: none; margin: 14px 0 8px; }
    .markdown-preview code { background: #eef2f6; border-radius: 4px; padding: 1px 4px; }
    @media (max-width: 1080px) {
      .app { grid-template-columns: 1fr; grid-template-rows: auto auto 70vh auto; height: auto; min-height: 100vh; }
      header, .pane, .chat { grid-column: 1; }
      .pane { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Letta Research Chat</h1>
      <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Starting</span></div>
    </header>

    <aside class="pane">
      <section class="section">
        <h2>Session</h2>
        <div class="panel">
          <div><span class="badge" id="agentName"></span></div>
          <p id="sessionMeta"></p>
          <div class="grid2">
            <button id="refreshBtn">Refresh</button>
            <button id="newChatBtn">New Chat</button>
          </div>
        </div>
      </section>

      <section class="section">
        <h2>Memory</h2>
        <label class="label">Memory text</label>
        <textarea id="memoryText" placeholder="Add an archival memory without an LLM call"></textarea>
        <button id="rememberBtn" class="primary">Remember</button>
        <div class="row" style="margin-top:8px">
          <input id="searchQuery" placeholder="Search archival memory">
          <button id="searchBtn">Search</button>
        </div>
        <div class="row" style="margin-top:8px">
          <input id="archivalLimit" type="number" min="1" value="25">
          <button id="archivalBtn">List</button>
        </div>
        <div class="viewer" style="margin-top:8px"><pre id="memoryOutput">(memory output)</pre></div>
      </section>

      <section class="section">
        <h2>Attacker RAG</h2>
        <label class="label">Attacker-controlled content</label>
        <textarea id="ragContent" placeholder="Content stored in the attacker RAG file"></textarea>
        <div class="grid2" style="margin-top:8px">
          <button id="ragSetBtn" class="primary">Save Content</button>
          <button id="ragShowBtn">Show File</button>
        </div>
        <div class="row" style="margin-top:8px">
          <input id="ragSourceFile" placeholder="source file to load">
          <button id="ragLoadBtn">Load</button>
        </div>
        <div class="row" style="margin-top:8px">
          <input id="ragSearchQuery" placeholder="Simulated websearch query">
          <button id="ragSearchBtn">Search</button>
        </div>
        <div class="viewer" style="margin-top:8px"><pre id="ragOutput">(attacker RAG output)</pre></div>
      </section>

      <section class="section">
        <h2>Persona Load</h2>
        <div class="grid2">
          <input id="initFile" placeholder="dataset.json">
          <input id="initIndex" type="number" min="0" value="0">
        </div>
        <button id="initBtn" style="margin-top:8px">Init Archival</button>
      </section>
    </aside>

    <main class="chat">
      <div id="messages" class="messages"></div>
      <form id="chatForm" class="composer">
        <textarea id="chatInput" placeholder="Type a message to the active Letta conversation"></textarea>
        <button class="primary" type="submit">Send</button>
      </form>
    </main>

    <aside class="pane">
      <section class="section">
        <h2>Experiments</h2>
        <div class="panel">
          <label class="label">Dataset</label>
          <input id="dataset" value="data_openai_gpt-oss-120b_short.json">
          <div class="grid3">
            <div><label class="label">Persona</label><input id="persona" type="number" min="0" value="0"></div>
            <div><label class="label">Memory</label><select id="memoryMode">
              <optgroup label="List memory">
                <option value="all">All memories</option>
                <option value="agent-search" selected>Agent search</option>
                <option value="list-search">CLI list search</option>
                <option value="list-rerank">CLI list rerank</option>
              </optgroup>
              <optgroup label="Graph memory">
                <option value="graph">Graph</option>
                <option value="graph-normalized">Graph normalized</option>
                <option value="graph-rerank">Graph rerank</option>
              </optgroup>
              <optgroup label="Attacker memory">
                <option value="attacker-search">Attacker search</option>
                <option value="attacker-inject">Attacker injection</option>
              </optgroup>
              <optgroup label="Profile memory">
                <option value="profile">Profile</option>
                <option value="profile-normalized">Profile normalized</option>
                <option value="profile-domain">Profile domain-batched</option>
                <option value="profile-labeled">Profile labeled</option>
                <option value="profile-schema">Profile schema-guided</option>
                <option value="profile-generic">Profile generic</option>
                <option value="profile-json">Profile JSON retrieval</option>
                <option value="profile-locomo">Profile LoCoMo</option>
                <option value="profile-locomo-rerank">Profile LoCoMo rerank</option>
                <option value="profile-locomo-context-rerank">Profile LoCoMo context + rerank</option>
                <option value="profile-locomo-events-rerank">Profile LoCoMo profile + event rerank</option>
              </optgroup>
            </select></div>
            <div><label class="label">Context</label><input id="contextIdx" type="number" min="0" value="0"></div>
          </div>
          <div class="grid2" style="margin-top:8px">
            <button data-job="query_one">Simulate Context</button>
            <button data-job="query_all">Run Context Set</button>
            <button data-job="pipeline">Privacy Pipeline</button>
            <button data-job="compare_modes">Compare all vs agent search</button>
            <button data-job="sweep">Sweep Limit</button>
            <button data-job="label_cimemories">Label Context</button>
          </div>
        </div>
      </section>

      <section class="section">
        <h2>Jobs</h2>
        <div id="jobs" class="list"></div>
        <div class="viewer" style="margin-top:8px"><pre id="jobLog">(select a job)</pre></div>
      </section>

      <section class="section">
        <h2>Results</h2>
        <button id="resultsBtn">Refresh Results</button>
        <div id="metrics" class="metric-grid"></div>
        <div id="results" class="list" style="margin-top:8px"></div>
        <div class="viewer" style="margin-top:8px"><pre id="resultViewer">(select a result)</pre></div>
      </section>

      <section class="section">
        <h2>Visualization Dashboard</h2>
        <div class="panel">
          <div class="tabs">
            <button class="active" data-viz-tab="pipelines">Runs</button>
            <button data-viz-tab="markdown">Markdown</button>
            <button data-viz-tab="queries">Queries</button>
          </div>
          <div class="grid2">
            <button id="dashboardBtn">Refresh</button>
            <button id="pipelineReportBtn">Print Run Table</button>
            <button id="memoryQueryReportBtn">Build Query Report</button>
            <button id="compareSelectedBtn">Compare Selected Runs</button>
          </div>
          <div id="dashboardMetrics" class="metric-grid"></div>
          <div id="dashboardChart" class="chart"></div>
          <div id="pipelineList" class="list" style="margin-top:8px"></div>
          <div id="markdownList" class="list" style="margin-top:8px"></div>
          <div id="queryList" class="list" style="margin-top:8px"></div>
          <div id="markdownPreview" class="markdown-preview" style="margin-top:8px">(select a Markdown report)</div>
        </div>
      </section>
    </aside>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    let selectedJobId = null;
    let selectedPipelinePath = null;
    let selectedComparePaths = new Set();
    let activeVizTab = 'pipelines';
    let busy = false;

    async function api(path, opts = {}) {
      const res = await fetch(path, {
        headers: {'Content-Type': 'application/json'},
        ...opts,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }

    function setStatus(ok, text) {
      $('statusDot').className = 'dot ' + (ok === true ? 'ok' : ok === false ? 'err' : '');
      $('statusText').textContent = text;
    }

    function addMessage(role, text) {
      const el = document.createElement('div');
      el.className = 'msg ' + role;
      el.innerHTML = `<div class="role">${role}</div><pre></pre>`;
      el.querySelector('pre').textContent = text || '';
      $('messages').appendChild(el);
      $('messages').scrollTop = $('messages').scrollHeight;
    }

    function fields() {
      return {
        dataset_name: $('dataset').value.trim(),
        persona_idx: Number($('persona').value),
        user_idx: Number($('persona').value),
        memory_mode: $('memoryMode').value,
        context_idx: Number($('contextIdx').value),
      };
    }

    async function refreshSession() {
      try {
        const s = await api('/api/session');
        $('agentName').textContent = s.agent_name;
        $('sessionMeta').textContent = `agent=${s.agent_id} model=${s.agent_model} conversation=${s.conversation_id || 'agent thread'} base=${s.base_url}`;
        if (s.connected === false) {
          setStatus(false, s.error || 'Letta server unavailable');
        } else {
          setStatus(true, s.use_convos ? 'Conversation API active' : 'Agent thread mode');
        }
        $('messages').innerHTML = '';
        (s.history || []).forEach(m => addMessage(m.role, m.text));
      } catch (e) {
        setStatus(false, e.message);
      }
    }

    async function sendChat(evt) {
      evt.preventDefault();
      if (busy) return;
      const text = $('chatInput').value.trim();
      if (!text) return;
      $('chatInput').value = '';
      addMessage('user', text);
      busy = true;
      try {
        const data = await api('/api/chat', {method: 'POST', body: JSON.stringify({message: text})});
        addMessage('assistant', data.reply || '[no assistant message]');
      } catch (e) {
        addMessage('system', e.message);
      } finally {
        busy = false;
      }
    }

    async function memoryAction(path, payload) {
      try {
        const data = await api(path, {method: 'POST', body: JSON.stringify(payload)});
        $('memoryOutput').textContent = data.text || JSON.stringify(data, null, 2);
      } catch (e) {
        $('memoryOutput').textContent = e.message;
      }
    }

    async function ragAction(path, payload) {
      try {
        const data = await api(path, {method: 'POST', body: JSON.stringify(payload)});
        $('ragOutput').textContent = data.text || JSON.stringify(data, null, 2);
        if (typeof data.content === 'string') $('ragContent').value = data.content;
      } catch (e) {
        $('ragOutput').textContent = e.message;
      }
    }

    async function startJob(kind) {
      try {
        const data = await api('/api/jobs', {method: 'POST', body: JSON.stringify({kind, ...fields()})});
        selectedJobId = data.job_id;
        await refreshJobs();
      } catch (e) {
        $('jobLog').textContent = e.message;
      }
    }

    async function refreshJobs() {
      const data = await api('/api/jobs');
      $('jobs').innerHTML = '';
      data.jobs.forEach(j => {
        const el = document.createElement('div');
        el.className = 'item';
        el.innerHTML = `<strong>${j.kind} <span class="badge">${j.status}</span></strong><small>${j.started_at || ''} ${j.result || ''}</small>`;
        el.onclick = async () => {
          selectedJobId = j.id;
          const detail = await api('/api/jobs/' + j.id);
          $('jobLog').textContent = detail.log || '(no output yet)';
        };
        $('jobs').appendChild(el);
      });
      if (selectedJobId) {
        try {
          const detail = await api('/api/jobs/' + selectedJobId);
          $('jobLog').textContent = detail.log || '(no output yet)';
        } catch {}
      }
    }

    async function refreshResults() {
      const data = await api('/api/results');
      $('results').innerHTML = '';
      $('metrics').innerHTML = '';
      if (data.summary) {
        Object.entries(data.summary).forEach(([k, v]) => {
          const el = document.createElement('div');
          el.className = 'metric';
          el.innerHTML = `<span>${k}</span><b>${v}</b>`;
          $('metrics').appendChild(el);
        });
      }
      data.results.forEach(r => {
        const el = document.createElement('div');
        el.className = 'item';
        el.innerHTML = `<strong>${r.name}</strong><small>${r.kind} - ${r.modified_at}</small>`;
        el.onclick = async () => {
          const detail = await api('/api/results/' + encodeURIComponent(r.path));
          $('resultViewer').textContent = detail.text || JSON.stringify(detail, null, 2);
        };
        $('results').appendChild(el);
      });
    }

    function fmtPercent(value) {
      return value === null || value === undefined ? 'n/a' : (Number(value) * 100).toFixed(1) + '%';
    }

    function fmtSeconds(value) {
      return value === null || value === undefined ? 'n/a' : (Number(value) / 1000).toFixed(2) + 's';
    }

    function fmtTokens(value) {
      return value === null || value === undefined ? 'n/a' : Number(value).toFixed(1);
    }

    function renderMarkdown(text) {
      const escape = (s) => String(s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
      const lines = String(text || '').split(/\r?\n/);
      let html = '';
      let inTable = false;
      const closeTable = () => { if (inTable) { html += '</tbody></table>'; inTable = false; } };
      for (const line of lines) {
        if (/^\|.*\|$/.test(line.trim())) {
          const cells = line.trim().slice(1, -1).split('|').map(c => c.trim());
          if (cells.every(c => /^:?-{3,}:?$/.test(c))) continue;
          if (!inTable) { html += '<table><tbody>'; inTable = true; }
          html += '<tr>' + cells.map(c => '<td>' + escape(c.replace(/\*\*/g, '').replace(/\*/g, '')) + '</td>').join('') + '</tr>';
          continue;
        }
        closeTable();
        if (line.startsWith('# ')) html += '<h1>' + escape(line.slice(2)) + '</h1>';
        else if (line.startsWith('## ')) html += '<h2>' + escape(line.slice(3)) + '</h2>';
        else if (!line.trim()) html += '<br>';
        else html += '<p>' + escape(line).replace(/`([^`]+)`/g, '<code>$1</code>') + '</p>';
      }
      closeTable();
      return html || '(empty report)';
    }

    function setVizTab(tab) {
      activeVizTab = tab;
      document.querySelectorAll('[data-viz-tab]').forEach(btn => btn.classList.toggle('active', btn.dataset.vizTab === tab));
      $('pipelineList').style.display = tab === 'pipelines' ? 'grid' : 'none';
      $('markdownList').style.display = tab === 'markdown' ? 'grid' : 'none';
      $('queryList').style.display = tab === 'queries' ? 'grid' : 'none';
    }

    async function refreshDashboard() {
      const data = await api('/api/dashboard');
      $('dashboardMetrics').innerHTML = '';
      Object.entries(data.summary || {}).forEach(([k, v]) => {
        const el = document.createElement('div');
        el.className = 'metric';
        el.innerHTML = `<span>${k}</span><b>${v}</b>`;
        $('dashboardMetrics').appendChild(el);
      });
      $('dashboardChart').innerHTML = '';
      (data.pipelines || []).slice(0, 12).forEach(p => {
        const task = Math.max(0, Math.min(1, Number(p.necessary_recall || 0)));
        const leak = Math.max(0, Math.min(1, Number(p.leak_rate || 0)));
        const ambiguous = Math.max(0, Math.min(1, Number(p.ambiguous_exposure_rate || 0)));
        const row = document.createElement('div');
        row.className = 'chart-row';
        row.innerHTML = `<span>${p.label}</span><div><div class="bar-track"><div class="bar" style="width:${task * 100}%"></div></div><div class="bar-track" style="margin-top:3px"><div class="bar leak" style="width:${leak * 100}%"></div></div><div class="bar-track" style="margin-top:3px"><div class="bar ambiguous" style="width:${ambiguous * 100}%"></div></div></div><span>${fmtPercent(task)} / ${fmtPercent(leak)} / ${fmtPercent(ambiguous)}</span>`;
        $('dashboardChart').appendChild(row);
      });
      $('pipelineList').innerHTML = '';
      (data.pipelines || []).forEach(p => {
        const el = document.createElement('div');
        el.className = 'item';
        const checked = selectedComparePaths.has(p.path) ? 'checked' : '';
        el.innerHTML = `<strong><input type="checkbox" ${checked} style="width:auto;min-height:auto;margin-right:6px"> ${p.label}</strong><small>mode=${p.memory_mode} contexts=${p.context_count} C=${fmtPercent(p.necessary_recall)} L=${fmtPercent(p.leak_rate)} A=${fmtPercent(p.ambiguous_exposure_rate)} Tmean=${fmtSeconds(p.online_mean_ms)} Tp95=${fmtSeconds(p.online_p95_ms)} C/s=${p.completion_per_second === null || p.completion_per_second === undefined ? 'n/a' : Number(p.completion_per_second).toFixed(4)} tokens/query=${fmtTokens(p.exact_tokens_per_query)} coverage=${fmtPercent(p.exact_token_coverage)}<br>${p.path}</small>`;
        el.querySelector('input').onclick = (evt) => {
          evt.stopPropagation();
          if (evt.target.checked) selectedComparePaths.add(p.path); else selectedComparePaths.delete(p.path);
        };
        el.onclick = () => {
          selectedPipelinePath = p.path;
          $('markdownPreview').innerHTML = `<h1>${p.label}</h1><p>${p.path}</p><p>Task completion: ${fmtPercent(p.necessary_recall)}<br>Private leak rate: ${fmtPercent(p.leak_rate)}<br>Ambiguous exposure rate: ${fmtPercent(p.ambiguous_exposure_rate)}<br>Online latency mean/p95: ${fmtSeconds(p.online_mean_ms)} / ${fmtSeconds(p.online_p95_ms)}<br>Completion/second: ${p.completion_per_second === null || p.completion_per_second === undefined ? 'n/a' : Number(p.completion_per_second).toFixed(4)}<br>Exact tokens/query: ${fmtTokens(p.exact_tokens_per_query)} (${fmtPercent(p.exact_token_coverage)} coverage)</p>`;
        };
        $('pipelineList').appendChild(el);
      });
      $('markdownList').innerHTML = '';
      (data.markdown_reports || []).forEach(r => {
        const el = document.createElement('div');
        el.className = 'item';
        el.innerHTML = `<strong>${r.name}</strong><small>${r.report_type} - ${r.modified_at}<br>${r.path}</small>`;
        el.onclick = async () => {
          const detail = await api('/api/results/' + encodeURIComponent(r.path));
          $('markdownPreview').innerHTML = renderMarkdown(detail.text || '');
        };
        $('markdownList').appendChild(el);
      });
      $('queryList').innerHTML = '';
      (data.query_reports || []).forEach(r => {
        const el = document.createElement('div');
        el.className = 'item';
        el.innerHTML = `<strong>${r.name}</strong><small>${r.row_count} query events - ${r.modified_at}<br>${r.path}</small>`;
        el.onclick = async () => {
          const detail = await api('/api/results/' + encodeURIComponent(r.path));
          $('markdownPreview').innerHTML = `<pre>${detail.text || ''}</pre>`;
        };
        $('queryList').appendChild(el);
      });
      setVizTab(activeVizTab);
    }

    async function startDashboardJob(kind) {
      const payload = {kind};
      if (kind === 'memory_query_report') payload.pipeline_path = selectedPipelinePath;
      if (kind === 'compare_selected') payload.paths = Array.from(selectedComparePaths);
      try {
        const data = await api('/api/dashboard/jobs', {method: 'POST', body: JSON.stringify(payload)});
        selectedJobId = data.job_id;
        await refreshJobs();
      } catch (e) {
        $('jobLog').textContent = e.message;
      }
    }

    $('chatForm').onsubmit = sendChat;
    $('refreshBtn').onclick = refreshSession;
    $('newChatBtn').onclick = async () => { await api('/api/newchat', {method: 'POST', body: '{}'}); await refreshSession(); };
    $('rememberBtn').onclick = () => memoryAction('/api/memory/remember', {text: $('memoryText').value});
    $('searchBtn').onclick = () => memoryAction('/api/memory/search', {query: $('searchQuery').value});
    $('archivalBtn').onclick = () => memoryAction('/api/memory/list', {limit: Number($('archivalLimit').value)});
    $('ragSetBtn').onclick = () => ragAction('/api/attacker-rag/set', {content: $('ragContent').value});
    $('ragShowBtn').onclick = () => ragAction('/api/attacker-rag/show', {});
    $('ragLoadBtn').onclick = () => ragAction('/api/attacker-rag/load', {source_file: $('ragSourceFile').value});
    $('ragSearchBtn').onclick = () => ragAction('/api/attacker-rag/search', {query: $('ragSearchQuery').value});
    $('initBtn').onclick = () => startJob('init_archival');
    $('resultsBtn').onclick = refreshResults;
    $('dashboardBtn').onclick = refreshDashboard;
    $('pipelineReportBtn').onclick = () => startDashboardJob('pipeline_report');
    $('memoryQueryReportBtn').onclick = () => startDashboardJob('memory_query_report');
    $('compareSelectedBtn').onclick = () => startDashboardJob('compare_selected');
    document.querySelectorAll('[data-viz-tab]').forEach(btn => btn.onclick = () => setVizTab(btn.dataset.vizTab));
    document.querySelectorAll('[data-job]').forEach(btn => btn.onclick = () => startJob(btn.dataset.job));

    refreshSession();
    refreshJobs();
    refreshResults();
    refreshDashboard();
    setInterval(refreshJobs, 2500);
  </script>
</body>
</html>
"""


@dataclass
class WebSession:
    cfg: LettaConfig
    http: HttpClient
    agent: AgentClient
    convos: ConversationClient
    mem: MemoryClient
    agent_id: str
    active_agent_model: str
    use_convos: bool
    conversation_id: str | None
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class JobRecord:
    id: str
    kind: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    log: str = ""
    result: str | None = None
    error: str | None = None
    future: Future[Any] | None = None


class JobStore:
    def __init__(self, max_workers: int = 2) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.RLock()
        self.jobs: dict[str, JobRecord] = {}

    def submit(self, kind: str, fn: Callable[[], Any]) -> JobRecord:
        job = JobRecord(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            status="queued",
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        with self.lock:
            self.jobs[job.id] = job
        job.future = self.executor.submit(self._run, job, fn)
        return job

    def _run(self, job: JobRecord, fn: Callable[[], Any]) -> None:
        with self.lock:
            job.status = "running"
            job.started_at = datetime.now().isoformat(timespec="seconds")
        stream = StringIO()
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                result = fn()
            with self.lock:
                job.status = "finished"
                job.result = str(result) if result is not None else ""
        except Exception as exc:
            with self.lock:
                job.status = "failed"
                job.error = str(exc)
            stream.write("\n")
            stream.write(traceback.format_exc())
        finally:
            with self.lock:
                job.log = stream.getvalue()
                job.finished_at = datetime.now().isoformat(timespec="seconds")

    def list(self) -> list[JobRecord]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> JobRecord | None:
        with self.lock:
            return self.jobs.get(job_id)


class SessionState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.session: WebSession | None = None
        self.error: str | None = None

    def get(self) -> WebSession:
        with self.lock:
            if self.session is None:
                try:
                    self.session = _create_session()
                    self.error = None
                except Exception as exc:
                    self.error = str(exc)
                    raise
            return self.session

def _require_flask() -> None:
    if _FLASK_IMPORT_ERROR is not None:
        raise RuntimeError("Flask is not installed. Install with: pip install -e '.[gui]'") from _FLASK_IMPORT_ERROR


def _create_session() -> WebSession:
    cfg = LettaConfig()
    http = HttpClient()
    agent = AgentClient(cfg.base_url, http)
    convos = ConversationClient(cfg.base_url, http)
    mem = MemoryClient(cfg.base_url, http)
    active_agent_model = cfg.agent_model
    agent_id, _ = agent.get_or_create_agent_id(
        cfg.agent_name,
        model=active_agent_model,
        **agent_llm_config_kwargs(cfg),
        archival_search_limit=cfg.archival_search_limit,
    )
    try:
        agent.update_agent_model(
            agent_id,
            active_agent_model,
            model_endpoint_type=cfg.agent_model_endpoint_type,
            model_endpoint=cfg.agent_model_endpoint,
            reasoning_effort=cfg.agent_reasoning_effort,
            context_window=cfg.agent_context_window,
        )
        agent.update_agent_embedding_config(
            agent_id,
            embedding_endpoint_type=cfg.embedding_endpoint_type,
            embedding_endpoint=cfg.embedding_endpoint,
            embedding_model=cfg.embedding_model,
            embedding_dim=cfg.embedding_dim,
        )
    except Exception:
        pass
    try:
        agent.update_agent_system_prompt(agent_id, archival_search_limit=cfg.archival_search_limit)
    except Exception:
        pass

    use_convos = False
    conversation_id: str | None = None
    try:
        latest_id = convos.get_latest_conversation_id(agent_id, limit=100)
        if latest_id:
            conversation_id = latest_id
            use_convos = True
        else:
            conversation_id = create_conversation_or_raise(convos, agent_id)
            use_convos = True
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            use_convos = False
            conversation_id = None
        else:
            raise
    return WebSession(
        cfg=cfg,
        http=http,
        agent=agent,
        convos=convos,
        mem=mem,
        agent_id=agent_id,
        active_agent_model=active_agent_model,
        use_convos=use_convos,
        conversation_id=conversation_id,
    )


def _history_for_ui(session: WebSession) -> list[dict[str, str]]:
    if not session.use_convos or not session.conversation_id:
        return []
    payload = session.convos.get_conversation_messages(session.conversation_id, limit=100)
    text = format_conversation_history(payload)
    items: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        marker = None
        if line.endswith("USER:") or line == "USER:":
            marker = "user"
        elif line.endswith("ASSISTANT:") or line == "ASSISTANT:":
            marker = "assistant"
        elif line.endswith("SYSTEM:") or line == "SYSTEM:":
            marker = "system"
        if marker:
            if current_role and current_lines:
                items.append({"role": current_role, "text": "\n".join(current_lines).strip()})
            current_role = marker
            current_lines = []
        elif current_role:
            current_lines.append(line)
    if current_role and current_lines:
        items.append({"role": current_role, "text": "\n".join(current_lines).strip()})
    return [item for item in items if item["text"]]


def _json_error(message: str, status: int = 400) -> tuple[Any, int]:
    return jsonify({"error": message}), status


def _read_json_request() -> dict[str, Any]:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def _job_summary(job: JobRecord) -> dict[str, Any]:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "result": job.result,
        "error": job.error,
    }


def _latest_results(root: Path = Path("research_outputs"), limit: int = 40) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in ("**/privacy_pipeline_cimemories.json", "**/responses.jsonl", "**/privacy_metrics_cimemories.json", "**/*.md"):
        files.extend(root.glob(pattern))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for path in files[:limit]:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        results.append(
            {
                "name": path.parent.name if path.name in {"responses.jsonl", "privacy_pipeline_cimemories.json"} else path.name,
                "path": str(rel),
                "kind": path.name,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return results


def _load_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_rate(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _ambiguous_exposure_rate(payload: dict[str, Any]) -> float | None:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        stored_rate = _safe_rate(metrics.get("ambiguous_exposure_rate"))
        if stored_rate is not None:
            return stored_rate

    ambiguous = payload.get("ambiguous_attributes")
    repeated = payload.get("repeated_exposed_records")
    if not isinstance(ambiguous, list):
        return None
    ambiguous_set = {item for item in ambiguous if isinstance(item, str)}
    if not ambiguous_set:
        return None
    if not isinstance(repeated, list) or not repeated:
        exposed = payload.get("exposed_attributes")
        if not isinstance(exposed, dict):
            return None
        return len(ambiguous_set & set(exposed)) / len(ambiguous_set)
    exposure_events = 0
    valid_repeats = 0
    for record in repeated:
        if not isinstance(record, dict):
            continue
        exposed = record.get("exposed_attributes")
        if not isinstance(exposed, dict):
            continue
        exposure_events += len(ambiguous_set & set(exposed))
        valid_repeats += 1
    if not valid_repeats:
        return None
    return exposure_events / (len(ambiguous_set) * valid_repeats)


def _pipeline_label(summary: dict[str, Any], path: Path) -> str:
    memory_mode = summary.get("memory_mode")
    limit = (
        summary.get("archival_search_limit")
        or summary.get("graphiti_search_limit")
        or ((summary.get("rerank") or {}).get("candidate_limit") if isinstance(summary.get("rerank"), dict) else None)
    )
    if limit is None:
        return f"mem{memory_mode} {path.parent.name[-15:]}"
    return f"mem{memory_mode} limit{limit}"


def _pipeline_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    contexts = summary.get("contexts")
    if not isinstance(contexts, list):
        contexts = []
    necessary_rates: list[float] = []
    leak_rates: list[float] = []
    ambiguous_rates: list[float] = []
    exposed_totals: list[float] = []
    context_rows: list[dict[str, Any]] = []
    for context in contexts:
        if not isinstance(context, dict):
            continue
        metrics_path_raw = context.get("privacy_metrics_cimemories_json")
        if not isinstance(metrics_path_raw, str):
            continue
        metrics_payload = _load_json_file(Path(metrics_path_raw))
        if not isinstance(metrics_payload, dict):
            continue
        metrics = metrics_payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        necessary_recall = _safe_rate(metrics.get("necessary_recall"))
        leak_rate = _safe_rate(metrics.get("inappropriate_leak_rate"))
        ambiguous_rate = _ambiguous_exposure_rate(metrics_payload)
        exposed_total = _safe_rate(metrics.get("average_exposed_total"))
        if necessary_recall is not None:
            necessary_rates.append(necessary_recall)
        if leak_rate is not None:
            leak_rates.append(leak_rate)
        if ambiguous_rate is not None:
            ambiguous_rates.append(ambiguous_rate)
        if exposed_total is not None:
            exposed_totals.append(exposed_total)
        context_rows.append(
            {
                "context_idx": context.get("context_idx"),
                "recipient": context.get("recipient"),
                "necessary_recall": necessary_recall,
                "leak_rate": leak_rate,
                "ambiguous_exposure_rate": ambiguous_rate,
                "average_exposed_total": exposed_total,
            }
        )
    avg = lambda values: (sum(values) / len(values)) if values else None
    return {
        "necessary_recall": avg(necessary_rates),
        "leak_rate": avg(leak_rates),
        "ambiguous_exposure_rate": avg(ambiguous_rates),
        "average_exposed_total": avg(exposed_totals),
        "contexts_with_metrics": len(context_rows),
        "context_rows": context_rows,
    }


def _pipeline_efficiency(summary: dict[str, Any]) -> dict[str, Any]:
    efficiency = summary.get("efficiency")
    if not isinstance(efficiency, dict):
        return {
            "online_mean_ms": None,
            "online_p95_ms": None,
            "exact_tokens_per_query": None,
            "exact_token_coverage": None,
            "completion_per_second": None,
        }
    timings = efficiency.get("timings_ms")
    tokens = efficiency.get("tokens")
    quality = efficiency.get("quality_normalized")
    deployed = timings.get("online_deployed_estimate") if isinstance(timings, dict) else None
    coverages = []
    if isinstance(tokens, dict):
        for key in ("generation_exact_coverage", "memory_preparation_exact_coverage"):
            value = _safe_rate(tokens.get(key))
            if value is not None:
                coverages.append(value)
    return {
        "online_mean_ms": deployed.get("mean") if isinstance(deployed, dict) else None,
        "online_p95_ms": deployed.get("p95") if isinstance(deployed, dict) else None,
        "exact_tokens_per_query": (
            tokens.get("deployed_exact_tokens_per_query") if isinstance(tokens, dict) else None
        ),
        "exact_token_coverage": min(coverages) if coverages else None,
        "completion_per_second": (
            quality.get("completion_rate_per_second") if isinstance(quality, dict) else None
        ),
    }


def _dashboard_pipelines(root: Path = Path("research_outputs"), limit: int = 80) -> list[dict[str, Any]]:
    paths = sorted(
        root.glob("**/privacy_pipeline_cimemories.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if root.exists() else []
    rows: list[dict[str, Any]] = []
    for path in paths[:limit]:
        summary = _load_json_file(path)
        if not isinstance(summary, dict):
            continue
        metrics = _pipeline_metrics(summary)
        efficiency = _pipeline_efficiency(summary)
        rel = path.resolve().relative_to(Path.cwd().resolve())
        rows.append(
            {
                "name": path.parent.name,
                "label": _pipeline_label(summary, path),
                "path": str(rel),
                "dir_path": str(path.parent.resolve().relative_to(Path.cwd().resolve())),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "dataset_name": summary.get("dataset_name"),
                "persona_idx": summary.get("persona_idx"),
                "persona_name": summary.get("persona_name"),
                "memory_mode": summary.get("memory_mode"),
                "agent_model": summary.get("agent_model"),
                "context_count": summary.get("context_count") or len(summary.get("contexts") or []),
                "scenario_repeats": summary.get("scenario_repeats"),
                **metrics,
                **efficiency,
            }
        )
    return rows


def _markdown_report_type(path: Path) -> str:
    name = path.name
    if "memory_queries" in name:
        return "memory queries"
    if "multi_comparison" in name:
        return "multi-run comparison"
    if "multi_summary" in name:
        return "multi-run summary"
    if "comparison" in name:
        return "comparison"
    return "markdown"


def _dashboard_markdown(root: Path = Path("research_outputs"), limit: int = 80) -> list[dict[str, Any]]:
    paths = sorted(root.glob("**/*.md"), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
    rows = []
    for path in paths[:limit]:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        rows.append(
            {
                "name": path.name,
                "path": str(rel),
                "report_type": _markdown_report_type(path),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def _dashboard_query_reports(root: Path = Path("research_outputs"), limit: int = 60) -> list[dict[str, Any]]:
    paths = sorted(
        root.glob("**/privacy_pipeline_cimemories_memory_queries_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if root.exists() else []
    rows = []
    for path in paths[:limit]:
        payload = _load_json_file(path)
        report_rows = payload.get("rows") if isinstance(payload, dict) else []
        rel = path.resolve().relative_to(Path.cwd().resolve())
        rows.append(
            {
                "name": path.name,
                "path": str(rel),
                "row_count": len(report_rows) if isinstance(report_rows, list) else 0,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def _dashboard_summary(pipelines: list[dict[str, Any]], markdown_reports: list[dict[str, Any]]) -> dict[str, Any]:
    best_completion = max((p["necessary_recall"] for p in pipelines if p.get("necessary_recall") is not None), default=None)
    best_leak = min((p["leak_rate"] for p in pipelines if p.get("leak_rate") is not None), default=None)
    best_ambiguous = min(
        (
            p["ambiguous_exposure_rate"]
            for p in pipelines
            if p.get("ambiguous_exposure_rate") is not None
        ),
        default=None,
    )
    return {
        "pipeline runs": len(pipelines),
        "Markdown reports": len(markdown_reports),
        "best completion": f"{best_completion * 100:.1f}%" if best_completion is not None else "n/a",
        "best leak": f"{best_leak * 100:.1f}%" if best_leak is not None else "n/a",
        "best ambiguous exposure": (
            f"{best_ambiguous * 100:.1f}%" if best_ambiguous is not None else "n/a"
        ),
    }


def _result_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"files": len(results)}
    for item in results:
        if item["kind"] == "privacy_pipeline_cimemories.json":
            try:
                payload = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
            except Exception:
                continue
            aggregate = payload.get("aggregate") or payload.get("summary") or {}
            for key in ("task_completion_rate", "privacy_violation_rate", "average_task_completion_rate", "average_privacy_violation_rate"):
                value = aggregate.get(key) if isinstance(aggregate, dict) else None
                if isinstance(value, (int, float)):
                    summary[key] = f"{value:.1%}" if value <= 1 else f"{value:.1f}"
            break
    return summary


def _safe_result_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    resolved = (Path.cwd() / path).resolve()
    roots = [(Path.cwd() / "research_outputs").resolve(), Path.cwd().resolve()]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError("Path is outside the workspace.")
    if not resolved.is_file():
        raise FileNotFoundError(raw_path)
    return resolved


def _safe_workspace_artifact_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    resolved = (Path.cwd() / path).resolve()
    roots = [(Path.cwd() / "research_outputs").resolve(), Path.cwd().resolve()]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError("Path is outside the workspace.")
    if not resolved.exists():
        raise FileNotFoundError(raw_path)
    return resolved


def create_app() -> Any:
    _require_flask()
    app = Flask(__name__)
    state = SessionState()
    jobs = JobStore()

    @app.get("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/api/session")
    def api_session() -> Any:
        try:
            session = state.get()
            with session.lock:
                return jsonify(
                    {
                        "connected": True,
                        "base_url": session.cfg.base_url,
                        "agent_name": session.cfg.agent_name,
                        "agent_id": session.agent_id,
                        "agent_model": session.active_agent_model,
                        "use_convos": session.use_convos,
                        "conversation_id": session.conversation_id,
                        "history": _history_for_ui(session),
                    }
                )
        except Exception as exc:
            cfg = LettaConfig()
            return jsonify(
                {
                    "connected": False,
                    "error": str(exc),
                    "base_url": cfg.base_url,
                    "agent_name": cfg.agent_name,
                    "agent_id": "",
                    "agent_model": cfg.agent_model,
                    "use_convos": False,
                    "conversation_id": None,
                    "history": [],
                }
            )

    @app.post("/api/chat")
    def api_chat() -> Any:
        try:
            payload = _read_json_request()
            message = str(payload.get("message") or "").strip()
            if not message:
                return _json_error("message is required")
            session = state.get()
            with session.lock:
                if session.use_convos and session.conversation_id:
                    resp = session.convos.send_conversation_message(session.conversation_id, message)
                else:
                    resp = session.agent.send_agent_message(session.agent_id, message)
            return jsonify({"reply": extract_assistant_reply(resp) or ""})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/newchat")
    def api_newchat() -> Any:
        try:
            session = state.get()
            with session.lock:
                if session.use_convos:
                    session.conversation_id = create_conversation_or_raise(session.convos, session.agent_id)
                else:
                    session.agent.reset_messages(session.agent_id)
            return jsonify({"ok": True, "conversation_id": session.conversation_id})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/memory/list")
    def api_memory_list() -> Any:
        try:
            payload = _read_json_request()
            limit = max(1, int(payload.get("limit") or 50))
            session = state.get()
            data = session.mem.list_archival_passages(session.agent_id, limit=limit, ascending=False)
            return jsonify({"text": format_all_archival_memories(data), "raw": data})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/memory/search")
    def api_memory_search() -> Any:
        try:
            payload = _read_json_request()
            query = str(payload.get("query") or "").strip()
            if not query:
                return _json_error("query is required")
            session = state.get()
            data = session.mem.search_archival_memory(session.agent_id, query, limit=session.cfg.archival_search_limit)
            return jsonify({"text": format_all_archival_memories(data), "raw": data})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/memory/remember")
    def api_memory_remember() -> Any:
        try:
            payload = _read_json_request()
            text = str(payload.get("text") or "").strip()
            if not text:
                return _json_error("text is required")
            session = state.get()
            data = session.mem.insert_archival_memory(session.agent_id, text)
            return jsonify({"text": "Inserted into archival memory.", "raw": data})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/attacker-rag/set")
    def api_attacker_rag_set() -> Any:
        try:
            payload = _read_json_request()
            content = str(payload.get("content") or "")
            cfg = LettaConfig()
            path = AttackerRagStore(cfg.attacker_rag_file).write_content(content)
            return jsonify({"text": f"Wrote attacker RAG content to {path}", "content": content, "path": str(path)})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/attacker-rag/load")
    def api_attacker_rag_load() -> Any:
        try:
            payload = _read_json_request()
            source_file = str(payload.get("source_file") or "").strip()
            if not source_file:
                return _json_error("source_file is required")
            cfg = LettaConfig()
            store = AttackerRagStore(cfg.attacker_rag_file)
            path = store.load_from_file(source_file)
            content = store.read_content()
            return jsonify({"text": f"Loaded {source_file} into {path}", "content": content, "path": str(path)})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/attacker-rag/show")
    def api_attacker_rag_show() -> Any:
        try:
            cfg = LettaConfig()
            store = AttackerRagStore(cfg.attacker_rag_file)
            content = store.read_content()
            return jsonify(
                {
                    "text": f"content_file={cfg.attacker_rag_file}\n\n{content or '(empty)'}",
                    "content": content,
                    "path": cfg.attacker_rag_file,
                }
            )
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/attacker-rag/search")
    def api_attacker_rag_search() -> Any:
        try:
            payload = _read_json_request()
            query = str(payload.get("query") or "").strip()
            if not query:
                return _json_error("query is required")
            cfg = LettaConfig()
            result = AttackerRagStore(cfg.attacker_rag_file).search(query, limit=cfg.attacker_rag_search_limit)
            return jsonify({"text": json.dumps(result.to_json(), indent=2, ensure_ascii=False), "raw": result.to_json()})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.get("/api/jobs")
    def api_jobs() -> Any:
        return jsonify({"jobs": [_job_summary(job) for job in jobs.list()]})

    @app.get("/api/jobs/<job_id>")
    def api_job(job_id: str) -> Any:
        job = jobs.get(job_id)
        if job is None:
            return _json_error("job not found", 404)
        payload = _job_summary(job)
        payload["log"] = job.log
        return jsonify(payload)

    @app.post("/api/jobs")
    def api_create_job() -> Any:
        try:
            payload = _read_json_request()
            kind = str(payload.get("kind") or "").strip()
            dataset_name = str(payload.get("dataset_name") or "").strip()
            user_idx = int(payload.get("user_idx", payload.get("persona_idx", 0)))
            persona_idx = int(payload.get("persona_idx", user_idx))
            memory_mode = parse_memory_mode(payload.get("memory_mode", "agent-search"))
            context_idx = int(payload.get("context_idx", 0))
            if kind not in {
                "init_archival",
                "query_one",
                "query_all",
                "pipeline",
                "compare_modes",
                "sweep",
                "label_cimemories",
            }:
                return _json_error("unsupported job kind")
            if not dataset_name:
                return _json_error("dataset_name is required")

            def run_job() -> Any:
                session = state.get()
                with session.lock:
                    if kind == "init_archival":
                        return init_archival_from_persona_file(session.mem, session.agent_id, dataset_name, user_idx)
                    if kind == "query_one":
                        return run_query_recipient_with_task(
                            cfg=session.cfg,
                            http=session.http,
                            agent=session.agent,
                            convos=session.convos,
                            mem=session.mem,
                            agent_id=session.agent_id,
                            agent_model=session.active_agent_model,
                            use_convos=session.use_convos,
                            conversation_id=session.conversation_id,
                            dataset_name=dataset_name,
                            user_idx=user_idx,
                            memory_mode=memory_mode,
                            context_idx=context_idx,
                        )
                    if kind == "query_all":
                        return run_query_recipients_with_tasks_experiment(
                            cfg=session.cfg,
                            http=session.http,
                            agent=session.agent,
                            convos=session.convos,
                            mem=session.mem,
                            dataset_name=dataset_name,
                            user_idx=user_idx,
                            memory_mode=memory_mode,
                            use_convos=session.use_convos,
                            agent_model=session.active_agent_model,
                        )
                    if kind == "pipeline":
                        return run_privacy_pipeline_cimemories(
                            cfg=session.cfg,
                            http=session.http,
                            agent=session.agent,
                            convos=session.convos,
                            mem=session.mem,
                            dataset_name=dataset_name,
                            persona_idx=persona_idx,
                            memory_mode=memory_mode,
                            use_convos=session.use_convos,
                            agent_model=session.active_agent_model,
                        )
                    if kind == "compare_modes":
                        return run_compare_privacy_modes_cimemories(
                            cfg=session.cfg,
                            http=session.http,
                            agent=session.agent,
                            convos=session.convos,
                            mem=session.mem,
                            dataset_name=dataset_name,
                            persona_idx=persona_idx,
                            use_convos=session.use_convos,
                            agent_model=session.active_agent_model,
                        )
                    if kind == "sweep":
                        return run_archival_search_limit_sweep_cimemories(
                            cfg=session.cfg,
                            http=session.http,
                            agent=session.agent,
                            convos=session.convos,
                            mem=session.mem,
                            dataset_name=dataset_name,
                            persona_idx=persona_idx,
                            use_convos=session.use_convos,
                            agent_model=session.active_agent_model,
                            memory_mode=memory_mode,
                        )
                    if kind == "label_cimemories":
                        return run_get_context_labeling_cimemories(
                            http=session.http,
                            dataset_name=dataset_name,
                            persona_idx=persona_idx,
                            context_idx=context_idx,
                        )
                    raise ValueError(kind)

            job = jobs.submit(kind, run_job)
            return jsonify({"job_id": job.id})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.get("/api/results")
    def api_results() -> Any:
        results = _latest_results()
        return jsonify({"results": results, "summary": _result_summary(results)})

    @app.get("/api/results/<path:result_path>")
    def api_result(result_path: str) -> Any:
        try:
            path = _safe_result_path(result_path)
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > 200_000:
                text = text[:200_000] + "\n\n[truncated]"
            return jsonify({"path": str(path.relative_to(Path.cwd())), "text": text})
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.get("/api/dashboard")
    def api_dashboard() -> Any:
        pipelines = _dashboard_pipelines()
        markdown_reports = _dashboard_markdown()
        query_reports = _dashboard_query_reports()
        return jsonify(
            {
                "summary": _dashboard_summary(pipelines, markdown_reports),
                "pipelines": pipelines,
                "markdown_reports": markdown_reports,
                "query_reports": query_reports,
            }
        )

    @app.post("/api/dashboard/jobs")
    def api_dashboard_jobs() -> Any:
        try:
            payload = _read_json_request()
            kind = str(payload.get("kind") or "").strip()
            if kind == "pipeline_report":
                raw_path = str(payload.get("pipeline_path") or "").strip()
                if not raw_path:
                    return _json_error("select a pipeline run first")
                path = _safe_workspace_artifact_path(raw_path)
                pipeline_path = path.parent if path.is_file() and path.name == "privacy_pipeline_cimemories.json" else path

                def run_pipeline_report() -> Any:
                    print_privacy_pipeline_cimemories_report(str(pipeline_path))
                    return pipeline_path

                job = jobs.submit("pipeline_report", run_pipeline_report)
                return jsonify({"job_id": job.id})
            if kind == "memory_query_report":
                raw_path = str(payload.get("pipeline_path") or "").strip()
                if not raw_path:
                    return _json_error("select a pipeline run first")
                path = _safe_workspace_artifact_path(raw_path)
                pipeline_path = path.parent if path.is_file() and path.name == "privacy_pipeline_cimemories.json" else path

                def run_memory_query_report() -> Any:
                    print_privacy_pipeline_cimemories_memory_queries(str(pipeline_path))
                    return pipeline_path

                job = jobs.submit("memory_query_report", run_memory_query_report)
                return jsonify({"job_id": job.id})
            if kind == "compare_selected":
                raw_paths = payload.get("paths")
                if not isinstance(raw_paths, list) or len(raw_paths) < 2:
                    return _json_error("select at least two pipeline runs")
                paths: list[str] = []
                for raw_path in raw_paths:
                    path = _safe_workspace_artifact_path(str(raw_path))
                    pipeline_path = path.parent if path.is_file() and path.name == "privacy_pipeline_cimemories.json" else path
                    paths.append(str(pipeline_path))

                def run_compare_selected() -> Any:
                    if len(paths) == 2:
                        return compare_privacy_pipeline_cimemories_reports(paths[0], paths[1])
                    return compare_privacy_pipeline_cimemories_multi_reports(paths)

                job = jobs.submit("compare_selected", run_compare_selected)
                return jsonify({"job_id": job.id})
            return _json_error("unsupported dashboard job kind")
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/results/compare")
    def api_results_compare() -> Any:
        try:
            payload = _read_json_request()
            paths = [str(item) for item in payload.get("paths") or []]
            if len(paths) < 2:
                return _json_error("at least two paths are required")

            def run_compare() -> Any:
                if len(paths) == 2:
                    return compare_privacy_pipeline_cimemories_reports(paths[0], paths[1])
                return compare_privacy_pipeline_cimemories_multi_reports(paths)

            job = jobs.submit("compare_results", run_compare)
            return jsonify({"job_id": job.id})
        except Exception as exc:
            return _json_error(str(exc), 500)

    return app


def main() -> None:
    _require_flask()
    app = create_app()
    host = "127.0.0.1"
    port = 5000
    print(f"Serving Letta Research Chat GUI at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
