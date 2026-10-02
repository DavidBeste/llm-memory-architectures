# letta-research-chat

A research-oriented Python package + CLI for interacting with a Letta (MemGPT) server.
This repo is structured so the interactive chat is just one consumer of the library; automation scripts can import the package and run experiments programmatically.

For anonymous paper-artifact installation, verification, pinned services, and
the exact CIMemories configurations, see [ARTIFACT.md](ARTIFACT.md).
For the complete reviewer replication sequence, see
[REPRODUCING_EXPERIMENTS.md](REPRODUCING_EXPERIMENTS.md).

## Features

- Create or resume an agent by name
- Resume latest conversation if available (server-dependent), otherwise create one
- Interactive CLI with command shortcuts:
  - `/core`, `/archival`, `/archival_search`, `/remember`, `/newchat`, `/use`, etc.
- Bulk persona initialization: parse `memory_statement` / `memory_statements` from a persona JSON file and insert into archival memory
- Clean library modules for reuse in research automation scripts

## Install

### Option A: editable install (recommended for research)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Option B: plain dependencies
```bash
pip install -r requirements.txt
```

### Optional dependencies
Install every optional integration in one go:
```bash
pip install -e '.[all]'
```

## Run the CLI

```bash
letta-chat
```

## Run the browser GUI

The CLI remains the default interface, but the same package can also serve a
local Flask UI for interactive chat, memory inspection, experiment launch, job
logs, and `research_outputs` previews.

Install the GUI extra:
```bash
pip install -e '.[gui]'
```

Then start the web server:
```bash
letta-chat-web
```

Open `http://127.0.0.1:5000` in your browser. The GUI uses the same environment
variables as the CLI and talks to the same Letta server. Long-running experiment
actions run as background jobs inside the web process; use the Jobs panel to
watch captured logs and the Results panel to inspect generated JSON, JSONL, and
Markdown outputs. The Visualization Dashboard summarizes completed CIMemories
pipeline runs, charts task-completion versus leak-rate metrics, previews saved
Markdown reports, and exposes the report-oriented CLI commands for selected
runs: per-run pipeline tables, memory-query Markdown/JSON report generation,
and multi-run comparisons.

Environment variables:
- `LETTA_BASE_URL` (default: `http://localhost:8283/v1`)
- `LETTA_AGENT_NAME` (default: `memory-chat`)
- `LETTA_AGENT_MODEL` (default: `gpt-5.2`, or `VLLM_MODEL` when set; used for Letta agent chat/query commands only)
- `LETTA_AGENT_MODEL_ENDPOINT_TYPE` (default: `openai`; Letta model endpoint type for newly created or updated agents)
- `LETTA_AGENT_MODEL_ENDPOINT` (default: `https://api.openai.com/v1`, or `VLLM_BASE_URL` when set; set to `http://host.docker.internal:11434/v1` from Docker or `http://localhost:11434/v1` from a local process for Ollama)
- `LETTA_AGENT_REASONING_EFFORT` (optional; reasoning effort sent in the Letta agent LLM config for newly created and updated agents, for example `medium` for GPT-5.6; omitted when unset)
- `LETTA_AGENT_CONTEXT_WINDOW` (default: `128000`; Letta agent context window)
- `LETTA_EMBEDDING_ENDPOINT_TYPE` (default: `openai`; embedding endpoint type for newly created agents)
- `LETTA_EMBEDDING_ENDPOINT` (default: `https://api.openai.com/v1`; set to a separate local embedding endpoint such as `http://host.docker.internal:8001/v1` when using local embeddings)
- `LETTA_EMBEDDING_MODEL` (default: `text-embedding-3-small`; embedding model for newly created agents)
- `LETTA_EMBEDDING_DIM` (default: `1536`; embedding dimension for newly created agents)
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`; used by direct judge calls and, unless role-specific overrides are set, reranking calls; set to `http://localhost:11434/v1` for Ollama or `http://127.0.0.1:8000/v1` for vLLM)
- `OPENAI_API_STYLE` (default: inferred from `OPENAI_BASE_URL`; use `chat_completions` for Ollama-compatible endpoints, otherwise `responses`)
- `LETTA_JUDGE_MODEL` (default: `gpt-5.2`, or `VLLM_MODEL` when set; model used by direct judge/evaluation commands when they do not specify an override)
- `LETTA_CONTEXT_LABEL_MODEL` (default: `gpt-5`; model used only to create CIMemories context labels. Complete cached labelings are reused only when they were produced by this model)
- `LETTA_RESEARCH_OPENAI_TIMEOUT` (default: `180`; timeout in seconds for direct judge/rerank LLM requests, useful for slower local models)
- `LETTA_SERVER_TIMEOUT` (default: `120`; timeout in seconds for requests from `letta-chat` to the Letta server, including response generation. Increase this for slower remote generation models; it changes only how long the client waits and does not change model prompts or inference parameters)
- `LETTA_JUDGE_MAX_TOKENS` (default: `2048`; maximum output tokens for direct judge/rerank LLM requests, useful for preventing local chat-completions models from generating oversized malformed JSON)
- `LETTA_JUDGE_TRUNCATION_RETRY_MAX_TOKENS` (default: `8192`; during pre/post-rerank response replay, retry only an exposure-judge response explicitly truncated with `status=incomplete` and `reason=max_output_tokens` at this ceiling; set to `0` to disable this compatibility fallback. The nominal judge configuration and checkpoint fingerprint remain unchanged, and both attempts are recorded.)
- `LETTA_JUDGE_REASONING_EFFORT` (optional; reasoning effort for direct judge calls through the OpenAI Responses API, for example `minimal` for inexpensive classification with `gpt-5-nano`; omitted by default so the provider chooses its model default)
- `LETTA_JUDGE_ERROR_DIR` (default: `research_outputs/judge-error-responses`; directory where the complete raw provider response is saved when a judge call returns no usable output text)
- `LETTA_MEMORY_STAGE_JUDGE_MODEL` (optional; model used only by `/compute_memory_stage_metrics`, falling back to `LETTA_JUDGE_MODEL`; an explicit `--judge-model` takes precedence)
- `LETTA_MEMORY_STAGE_JUDGE_BASE_URL` (optional; independently route direct returned-memory classification without changing response-exposure judging)
- `LETTA_MEMORY_STAGE_JUDGE_API_STYLE` (optional: `responses` or `chat_completions`; inferred from the memory-stage endpoint when omitted)
- `LETTA_MEMORY_STAGE_JUDGE_API_KEY` (optional; credential used only for direct returned-memory classification; Together endpoints also accept `TOGETHER_API_KEY`; the key is never saved)
- `LETTA_MEMORY_STAGE_JUDGE_REASONING_EFFORT` (optional; memory-stage-only reasoning setting, falling back to `LETTA_JUDGE_REASONING_EFFORT`)
- `LETTA_MEMORY_STAGE_JUDGE_MAX_TOKENS` (optional; memory-stage-only output ceiling, falling back to `LETTA_JUDGE_MAX_TOKENS`)
- `LETTA_MEMORY_STAGE_JUDGE_TIMEOUT` (optional; memory-stage-only timeout in seconds, falling back to the generic judge timeout)
- `LETTA_MEMORY_STAGE_JUDGE_STREAM` (optional boolean; enables streamed chat completions only for the direct returned-memory judge, for providers/models that require streaming)
- `LETTA_EXPOSED_JUDGE_SCHEMA` (default: `indices` for chat-completions endpoints, `attributes` for Responses API; use `indices` for local models to reduce malformed long JSON outputs)
- `VLLM_BASE_URL` (optional; OpenAI-compatible vLLM base URL, for example `http://127.0.0.1:8000/v1` from the host or `http://host.docker.internal:8000/v1` from Docker)
- `VLLM_API_KEY` (optional; bearer token for vLLM direct judge/rerank calls)
- `VLLM_MODEL` (optional; vLLM model id, for example `qwen3-8b-gptq`; used as a fallback for Letta agent, judge, and rerank model settings)
- `LETTA_ARCHIVAL_SEARCH_LIMIT` (default: `10`; used as `top_k` for both agent archival searches and direct `/archival_search` REST calls)
- `LETTA_ATTACKER_RAG_FILE` (default: `research_outputs/attacker_rag_content.txt`; local disk file used by attacker-controlled simulated web-search mode)
- `LETTA_ATTACKER_RAG_SEARCH_LIMIT` (default: `5`; number of attacker RAG chunks returned as simulated web-search results)
- `LETTA_RERANK_MODE` (default: `lexical`; for `list-rerank`, `graph-rerank`, `profile-locomo-rerank`, `profile-locomo-context-rerank`, and `profile-locomo-events-rerank`, use `lexical` or `llm`)
- `LETTA_RERANK_CANDIDATE_SOURCE` (default: `search`; for list/graph reranking use `search` or `all`; mode 17 uses all profile facts, mode 18 uses all facts in native context, and mode 19 reranks only Memobase-selected past events while preserving the profile)
- `LETTA_RERANK_CANDIDATE_LIMIT` (default: `200`; candidate pool size for list, graph, and profile modes 17/18; mode 19 reranks every event already selected within Memobase's native context budget)
- `LETTA_RERANK_OUTPUT_LIMIT` (default: `20`; maximum selected list/graph facts or profile events injected by reranked modes; mode 19 preserves the current profile separately)
- `LETTA_RERANK_MODEL` (default: `gpt-5.2`, or `VLLM_MODEL` when set; model used when `LETTA_RERANK_MODE=llm`)
- `LETTA_RERANK_BASE_URL` (optional; independently route list/graph/profile LLM reranking without changing the exposure judge endpoint)
- `LETTA_RERANK_API_STYLE` (optional: `responses` or `chat_completions`; inferred from the role-specific endpoint when omitted)
- `LETTA_RERANK_API_KEY` (optional; credential used only for role-specific reranking; for Together endpoints, `TOGETHER_API_KEY` is also accepted as a fallback)
- `LETTA_RERANK_TIMEOUT` (optional; reranker-only request timeout, falling back to `LETTA_RESEARCH_OPENAI_TIMEOUT`)
- `LETTA_RERANK_MAX_TOKENS` (optional; reranker-only output limit, falling back to `LETTA_JUDGE_MAX_TOKENS`)
- `LETTA_RERANK_REASONING_EFFORT` (optional; reranker-only Responses API reasoning setting, falling back to `LETTA_JUDGE_REASONING_EFFORT`)
- `LETTA_RERANK_CHECKPOINT_DIR` (default: `research_outputs/rerank-checkpoints`; content-addressed cache for successful LLM reranker calls. A checkpoint is reused only when the provider route, model, output/reasoning settings, complete prompt, candidate list, task, query, and output limit match exactly. API keys are never stored or fingerprinted)
- `LETTA_RERANK_PROMPT_MODE` (default: `legacy`; use `legacy` to reproduce the original indices-plus-rationale prompt exactly, or `indices_only` to retain the same candidates and selection criteria while requiring only `{"selected_indices":[...]}` for more reliable small/local-model output)
- `GRAPHITI_NEO4J_URI` (default: `bolt://localhost:7687`; used by the `graph*` memory choices)
- `GRAPHITI_NEO4J_USER` (default: `neo4j`; used by the `graph*` memory choices)
- `GRAPHITI_NEO4J_PASSWORD` (default: `password`; used by the `graph*` memory choices)
- `GRAPHITI_GROUP_PREFIX` (default: `letta-research-chat`; namespace prefix for Graphiti experiment graphs)
- `GRAPHITI_SEARCH_LIMIT` (default: `20`; graph facts injected for graph memory modes)
- `GRAPHITI_INGEST_LIMIT` (optional; limits how many persona memory strings are encoded as graph episodes for quick smoke tests)
- `GRAPHITI_BATCH_SIZE` (default: `12`; normalized statements per episode for `graph-normalized` and `graph-rerank`)
- `GRAPHITI_LLM_BASE_URL` (optional; OpenAI-compatible chat endpoint for Graphiti extraction, defaults to `VLLM_BASE_URL` when set)
- `GRAPHITI_LLM_API_KEY` (optional; bearer token for the Graphiti LLM endpoint, defaults to `VLLM_API_KEY` when set)
- `GRAPHITI_LLM_MODEL` (optional; model used by Graphiti extraction, defaults to `VLLM_MODEL` when set)
- `GRAPHITI_LLM_REASONING_EFFORT` (optional; reasoning effort for Graphiti extraction; defaults to `LETTA_AGENT_REASONING_EFFORT`. Official OpenAI and generic OpenAI-compatible endpoints both receive an explicitly configured effort. Set it to `none` to preserve the generic client's legacy request behavior.)
- `GRAPHITI_LLM_STRUCTURED_OUTPUT_MODE` (default: `json_object`; use `json_schema` if your local endpoint reliably supports structured JSON schema output)
- `GRAPHITI_LLM_MAX_TOKENS` (default: `4096`; maximum Graphiti extraction output tokens)
- `GRAPHITI_LLM_ERROR_DIR` (default: `research_outputs/graphiti-error-responses`; saves provider responses after empty or malformed structured output, including finish reasons and usage but never request headers or credentials)
- `GRAPHITI_EMBEDDING_BASE_URL` (optional; OpenAI-compatible embedding endpoint for Graphiti, defaults to `LETTA_EMBEDDING_ENDPOINT` when set)
- `GRAPHITI_EMBEDDING_API_KEY` (optional; bearer token for Graphiti embeddings, defaults to `VLLM_API_KEY` when set)
- `GRAPHITI_EMBEDDING_MODEL` (optional; embedding model for Graphiti, defaults to `LETTA_EMBEDDING_MODEL` when set)
- `GRAPHITI_EMBEDDING_DIM` (default: `LETTA_EMBEDDING_DIM` or `1024`; embedding dimension stored by Graphiti)
- `GRAPHITI_RERANKER_BASE_URL` (optional; OpenAI-compatible chat endpoint for Graphiti reranking, defaults to `GRAPHITI_LLM_BASE_URL` or `VLLM_BASE_URL`)
- `GRAPHITI_RERANKER_API_KEY` (optional; bearer token for Graphiti reranking, defaults to `VLLM_API_KEY` when set)
- `GRAPHITI_RERANKER_MODEL` (optional; model used by Graphiti reranking, defaults to `GRAPHITI_LLM_MODEL` or `VLLM_MODEL`)
- `MEMOBASE_PROJECT_URL` (default: `http://localhost:8019`; used by the `profile*` memory choices)
- `MEMOBASE_API_KEY` (default: `secret`; used by the `profile*` memory choices)
- `MEMOBASE_CONTEXT_MAX_TOKEN_SIZE` (default: `1000`; maximum native Memobase context size; LoCoMo modes 16, 18, and 19 use at least `3000`)

Graph memory mode is optional. Install it with:
```bash
pip install -e '.[graph]'
```
It also requires Neo4j 5.26+. By default Graphiti uses OpenAI credentials for
extraction and embeddings, but graph modes can run locally when the `GRAPHITI_*`
LLM, embedding, and reranker endpoint variables above point at OpenAI-compatible
local servers.

Profile memory mode is optional. Install it with:
```bash
pip install -e '.[profile]'
```
It requires a Memobase server or cloud project. The package above is only the
client; nothing in this repo starts a server. To self-host:

```bash
git clone https://github.com/memodb-io/memobase.git
cd memobase/src/server
cp .env.example .env                        # ships with port 8019, token "secret"
cp ./api/config.yaml.example ./api/config.yaml
docker compose build && docker compose up -d
curl http://localhost:8019/api/v1/healthcheck
```

`.env` defaults already match this repo (`API_EXPORT_PORT=8019`,
`ACCESS_TOKEN=secret`). Run `/check_local_backends` to confirm the CLI can reach
it, and `/memobase_config` to print the project's active profile schema.

**Memobase does its own extraction with its own LLM**, configured in that
server's `config.yaml` (`llm_api_key`, `best_llm_model`, `embedding_model`) — not
by any variable in this repo. That model materially changes results, so record
it alongside your runs. New manifests capture the project profile config and its
fingerprint under `profile_memory.server_profile_config`, and the fingerprint
participates in `--resume-compatible` matching so runs made against differently
configured deployments are not silently reused.

Note that reasoning models (`gpt-5.*`, o-series) do not work as Memobase
extractors without patching its OpenAI adapter: it sends `max_tokens`, which
those models reject in favour of `max_completion_tokens`.

Memobase is a *distilling* memory — it summarises blobs, then extracts what it
judges important, then merges and compresses. Its defaults are tuned for chat
histories and discard most of a fact-dense persona. For attribute-level recall
on CIMemories-style data, these settings matter more than anything in this repo:

| Setting | Default | Suggested | Why |
|---|---:|---:|---|
| `max_chat_blob_buffer_token_size` | 1024 | 100 | Largest single lever. Controls how many blobs get collapsed into one summary before extraction. |
| `max_profile_subtopics` | 15 | 200 | Above this, `organize_profiles` compresses a topic to `N//2+1` entries. |
| `max_pre_profile_token_size` | 128 | 2048 | Above this, `re_summary` paraphrases a slot to half length, losing exact values. |
| `additional_user_profiles` | — | your domains | Steers extraction toward the domains you care about. Use this, not `overwrite_user_profiles`. |

Measured on one CIMemories persona (147 attributes), those changes moved
attribute recall from 8.8% to 34.7%. A stronger extraction model did not help.
`profile_retention_probe.py` reproduces this measurement without running any
generation.

Example:
```bash
export LETTA_BASE_URL="http://localhost:8283/v1"
export LETTA_AGENT_NAME="memory-chat"
export LETTA_AGENT_MODEL="gpt-5.2"
export LETTA_AGENT_MODEL_ENDPOINT="https://api.openai.com/v1"
export LETTA_AGENT_REASONING_EFFORT="medium"
export LETTA_ARCHIVAL_SEARCH_LIMIT="10"
export LETTA_RERANK_MODE="lexical"
export LETTA_RERANK_CANDIDATE_SOURCE="search"
export LETTA_RERANK_CANDIDATE_LIMIT="200"
export LETTA_RERANK_OUTPUT_LIMIT="20"
letta-chat
```

Ollama example for agent responses through Letta:
```bash
export LETTA_AGENT_MODEL="qwen3.6:35b-a3b"
export LETTA_AGENT_MODEL_ENDPOINT_TYPE="openai"
export LETTA_AGENT_MODEL_ENDPOINT="http://host.docker.internal:11434/v1"
export LETTA_AGENT_CONTEXT_WINDOW="262144"
letta-chat
```

If `letta-chat` runs directly on the host instead of inside Docker, use
`http://localhost:11434/v1` for `LETTA_AGENT_MODEL_ENDPOINT`. For this repo's
direct judge/rerank calls, use:
```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_STYLE="chat_completions"
export OPENAI_API_KEY="ollama"
export LETTA_JUDGE_MODEL="qwen3.6:35b-a3b"
export LETTA_RERANK_MODEL="qwen3.6:35b-a3b"
```

To rerank with an OpenAI-compatible Together model while keeping exposure
judging on OpenAI, leave `OPENAI_BASE_URL` pointed at OpenAI and configure the
reranker independently:
```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_STYLE="responses"
export LETTA_JUDGE_MODEL="gpt-5.2"

export LETTA_RERANK_BASE_URL="https://api.together.xyz/v1"
export LETTA_RERANK_API_STYLE="chat_completions"
export LETTA_RERANK_API_KEY="$TOGETHER_API_KEY"
export LETTA_RERANK_MODEL="Prism-ML/Ternary-Bonsai-27B"
export LETTA_RERANK_MAX_TOKENS="2048"
```
Role-specific reranker routing is saved without credentials in experiment
manifests and participates in compatible-run matching.

vLLM example for agent responses through Letta:
```bash
export VLLM_API_KEY="your-vllm-token"
export VLLM_MODEL="qwen3-8b-gptq"
export LETTA_AGENT_MODEL_ENDPOINT_TYPE="openai"
export LETTA_AGENT_MODEL_ENDPOINT="http://host.docker.internal:8000/v1"
export LETTA_AGENT_CONTEXT_WINDOW="32768"
export LETTA_EMBEDDING_ENDPOINT_TYPE="openai"
export LETTA_EMBEDDING_ENDPOINT="http://host.docker.internal:8001/v1"
export LETTA_EMBEDDING_MODEL="bge-m3"
export LETTA_EMBEDDING_DIM="1024"
letta-chat
```

If `letta-chat` and Letta run directly on the host, use
`http://127.0.0.1:8000/v1` for `LETTA_AGENT_MODEL_ENDPOINT`. For this repo's
direct judge/rerank calls through vLLM, use:
```bash
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="your-vllm-token"
export VLLM_MODEL="qwen3-8b-gptq"
export OPENAI_API_STYLE="chat_completions"
```

For fully local Graphiti graph memory with the same vLLM chat and embedding
servers, add:
```bash
export GRAPHITI_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export GRAPHITI_LLM_API_KEY="$VLLM_API_KEY"
export GRAPHITI_LLM_MODEL="qwen3-8b-gptq"
export GRAPHITI_LLM_STRUCTURED_OUTPUT_MODE="json_object"
export GRAPHITI_EMBEDDING_BASE_URL="http://127.0.0.1:8001/v1"
export GRAPHITI_EMBEDDING_API_KEY="$VLLM_API_KEY"
export GRAPHITI_EMBEDDING_MODEL="bge-m3"
export GRAPHITI_EMBEDDING_DIM="1024"
export GRAPHITI_RERANKER_BASE_URL="http://127.0.0.1:8000/v1"
export GRAPHITI_RERANKER_API_KEY="$VLLM_API_KEY"
export GRAPHITI_RERANKER_MODEL="qwen3-8b-gptq"
```

When local reasoning models return visible `<think>...</think>` blocks, the CLI
keeps raw provider responses in JSON artifacts but strips those thinking blocks
from displayed assistant replies, Markdown responses, and downstream judging
inputs so saved outputs resemble OpenAI-style final answers.

### Reproducible CIMemories architecture launchers

The repository includes a non-secret experiment configuration in
`experiment_templates/cimemories_retrieval50_rerank20.env` and a short launcher
for the query-dependent 50-candidate to 20-output comparison. The launcher does
not read a repository `.env`, print credentials, or place them in command-line
arguments. It uses an existing `OPENAI_API_KEY` when available; otherwise it
reads the first line of `openai_token.txt` locally and exports it only to the
launcher process and its children.

The default environment policy is strict: pre-existing non-secret experiment
variables are cleared before the template is applied, preventing stale values
in a shell or `screen` session from changing the configuration. Credentials are
preserved. Pass `--allow-env-overrides` only when intentionally overriding the
template for a different deployment or experiment.

Run exactly one architecture per session:

```bash
scripts/run_cimemories_memory_experiment.sh list
scripts/run_cimemories_memory_experiment.sh graph
scripts/run_cimemories_memory_experiment.sh profile
```

For an inexpensive end-to-end check, add `--persona 0`. This uses the same
configuration and evaluates all 49 scenarios for only that zero-based persona:

```bash
scripts/run_cimemories_memory_experiment.sh list --persona 0
scripts/run_cimemories_memory_experiment.sh graph --persona 0
scripts/run_cimemories_memory_experiment.sh profile --persona 0
```

The defaults are `cimemories_raw.json`, `labels_qwen.json`, GPT-5.6 Sol with
medium reasoning, and a shared LLM reranker output ceiling of 20. List and graph
use query-dependent native retrieval with at most 50 atomic candidates and
full-size 3072-dimensional `text-embedding-3-large` embeddings; Graphiti batches
12 input statements per episode. Profile uses mode 19 with a 3000-token native
Memobase context: its complete current-profile section is preserved, while all
Past Events selected by Memobase for the query are reranked and at most 20 are
retained. This intentionally follows each architecture's practical retrieval
interface instead of imposing a fact-count cutoff on the persistent profile.
Across the three model-specific experiment templates, response exposure is
judged by GPT-5.2 and direct returned-memory attributes are judged by
`gpt-6-sol`; these evaluator roles are fixed independently of the response and
reranking model. List memory uses deterministic exact matching for the direct
memory stage and therefore does not call the semantic memory judge.
Inspect the generated command without starting a run:

```bash
scripts/run_cimemories_memory_experiment.sh list --dry-run
```

Pass `--resume-compatible` only when intentionally reusing stages with the same
complete configuration. The corrected search/50 configurations will not match
the older all/200 query runs. Alternate dataset and label files can be supplied
with `--dataset FILE` and `--labels-file FILE`. Use `--token-file FILE` for a
different credential file, or `--no-token-file` to require an already exported
`OPENAI_API_KEY`. Add `--allow-env-overrides` to retain existing non-secret
model, endpoint, embedding, Graphiti, and Memobase settings instead of applying
the strict template defaults.

If the named agent already exists, the CLI updates that agent's `llm_config.model`,
endpoint type, endpoint URL, and context window from the `LETTA_AGENT_*` settings
on startup. Inside the CLI, `/model` shows the configured and live model, and
`/model <name>` updates the active Letta agent model and endpoint settings for
the rest of the session. Research outputs produced by agent-backed commands
record the agent model in their metadata.

On startup, the CLI refreshes the agent system prompt with the current memory
tool contract. Letta's built-in `archival_memory_search` agent tool uses
`top_k` rather than `limit`; the CLI instructs agents to pass
`top_k=LETTA_ARCHIVAL_SEARCH_LIMIT`.

## CLI Commands

Commands are grouped by what you are trying to accomplish. Indentation and
arrows show composition: a command below another command wraps or extends that
workflow rather than introducing an unrelated one.

```text
one context query
└─ full-persona query batch
   └─ CIMemories pipeline (query + label + judge + metrics)
      ├─ whole-dataset pipeline
      ├─ two-mode comparison
      └─ retrieval-limit sweep

code-style memory setup
└─ generate a prompted solution
   └─ emit + compile + test the solution
```

> **Notation:** `<value>` is required, `[value]` is optional, and `...` means
> that the preceding argument may be repeated.

### Session and status

- `/exit`
- `/id`
- `/model [name]`: show or update the active Letta agent model
- `/cid`
- `/check_local_backends`: probe Letta, the configured OpenAI-compatible agent endpoint, embedding endpoint, direct judge endpoint, Neo4j socket, Graphiti readiness, and Memobase reachability
- `/memobase_config`: print the Memobase project's active profile schema and its fingerprint, and warn when `overwrite_user_profiles` is set (which a past `profile-locomo` run may have left behind)
- `/memobase_server_config [config.yaml]`: inspect the on-disk Memobase server YAML (or `MEMOBASE_SERVER_CONFIG_PATH`) without printing credentials or private profile definitions. It shows an allowlisted set of performance-sensitive values, file metadata, and a full-file fingerprint; because Memobase exposes no runtime endpoint for these settings, this cannot prove that the running process has reloaded the file.
- `/memory_modes`: show every readable memory name, its legacy numeric alias, and a one-line explanation
- `/newchat`: create a new conversation if supported, otherwise reset agent thread messages
- `/convos`: list conversations for this agent
- `/use <conversation_id>`: switch to a specific conversation id
- `/history`: formatted messages in current conversation, including tool-call arguments and tool-return content when available

### Memory and retrieval

- `/core`
- `/archival [N]`
- `/archival_search <query>`
- `/remember <text>`: insert archival memory directly (no LLM call)
- `/init_archival <file> <index>`: bulk-load persona memory statements into archival memory

#### Attacker-controlled RAG

- `/attacker_rag_set <text>`: overwrite the attacker-controlled RAG corpus file without touching archival memory
- `/attacker_rag_load <file>`: copy a local file into the configured attacker RAG corpus file
- `/attacker_rag_show`: print the attacker RAG file path and current content
- `/attacker_rag_search <query>`: run the local attacker RAG search directly

### Inspect saved runs

- `/run_history <pipeline_or_query_path> <context> [repeat|all]`: print saved experiment conversation history for a context from a pipeline directory, query output directory, manifest JSON, or `responses.jsonl`; `repeat` is 1-based and defaults to all repeats
- `/run_history_beautified <pipeline_or_query_path> <context> [repeat|all]`: print the same saved experiment history with dedicated sections for system prompt, user prompt, memory search query, retrieved memories, LLM response, expected share attributes, and actually shared attributes

### Code-style evaluation

Each level includes the work of the level above it:

1. `/init_code_style_memory <file> <index>` — store one dataset entry as a natural-language, list-based code-style preference. Entries have the shape `[{"id": ..., "original_name": ..., "neutral_name": ..., "source_files": [...], "code": "..."}]`.
2. ↳ `/run_code_style_prompt <memory_dataset> <memory_idx> <prompt_dataset_jsonl> <prompt_idx> [emit_file=0|1]` — initialize that memory and generate one solution. With `emit_file=1`, also write the assembled task, `_unsafe.c`, and `_test.py` files under `research_outputs/code-style-prompt-runs/`.
3. ↳ `/run_code_style_prompt_eval <memory_dataset> <memory_idx> <prompt_dataset_jsonl> <prompt_idx>` — run the prompt flow, emit its files, compile with `commons.py compile_all_in`, and execute pytest.

### Privacy experiments

#### 1. Query

- `/query_recipient_with_task <dataset_name> <user_idx> <memory> <context_idx>`: build the downstream user prompt for a dataset context, send it to the active Letta session, and print both prompt and response
- `/calibrate_reranker <dataset> <persona> [--pilot-contexts N] [--reasoning-efforts low,high,max] [--max-tokens N]`: initialize list memory once, retrieve each pilot context once, and compare reranker structured-output validity across provider reasoning settings without generating downstream messages or running privacy judges. Requires `LETTA_RERANK_PROMPT_MODE=indices_only`; defaults to 10 contexts, all three supported GLM effort levels, and 2048 output tokens. Successful calls use the normal exact checkpoint cache, and JSON plus Markdown reports are saved under `research_outputs/reranker-calibration_*`.
- ↳ `/query_recipients_with_tasks <dataset_name> <user_idx> <memory>`: wrap the single-context query across the full context set for one persona, saving prompts, responses, and exact histories under a unique output directory

#### 2. Label, judge, and score

- `/get_exposed_attributes <responses_jsonl>`: run a GPT-5.2 judge over each saved assistant response and store the exposed-attribute judgments on disk
- `/get_context_labeling <dataset> <persona> <privacy_persona> <context>`: ask a GPT-5.2 model to split one persona's full memory list into `share` vs `private` for a specific context and privacy persona, then save the result under `research_outputs`
- `/get_context_labeling_cimemories <dataset> <persona> <context>`: run CIMemories-style labeling with `LETTA_CONTEXT_LABEL_MODEL` (default: GPT-5) across all three Westin personas, 10 samples each, abstention handling, Westin-prior mixing, zero-entropy filtering, model-aware resumable sample caching, and progress logging
- `/compute_privacy_metrics <exposed_attributes_jsonl> <context_labeling_json>`: align one context-labeling scenario with the matching exposed-attributes record and save sharing/leakage metrics under `research_outputs`
- `/compute_privacy_metrics_cimemories <exposed_attributes_jsonl> <context_labeling_cimemories_json>`: compute task completion, private leakage, and ambiguous exposure metrics from CIMemories `necessary_attributes`, `inappropriate_attributes`, and `ambiguous_attributes` labels

#### 3. Run composed workflows

- `/run_privacy_pipeline_cimemories <dataset> <persona[,persona...]> <memory[,memory...]> [--repeats N] [--resume-compatible] [--with-pre-rerank] [--labels-file <labels.json>] [--skip-memory-stage-metrics]`: run the full CIMemories privacy pipeline end to end across all valid contexts, querying each scenario 10 times by default (or `N` times), reusing any already-complete context labelings, and computing per-context average exposure percentages. Comma-separated persona indices and memory modes run their Cartesian product sequentially; the complete selector and supplied label coverage are validated before paid work begins. Multi-cell runs continue after an individual failure and finish with exact recovery commands, while a single-cell invocation retains fail-fast behavior. With `--resume-compatible`, also reuse a complete matching response-generation stage and zero-error exposure judgments when their recorded dataset, persona, models, reasoning effort, architecture parameters, contexts, repeats, and attributes match. When combined with `--with-pre-rerank`, an existing complete compatible post pipeline with the same labels file is reused in place, so rerunning the same single command or matrix resumes its original pre-rerank checkpoints instead of creating another experiment folder. Before response generation, the query stage atomically saves `query_resume_state.json` with the initialized-memory identity, exact retrieval/reranking results, prompts, and preparation provenance. A compatible retry reopens that same query directory and experiment agent, skips memory initialization, retrieval, reranking, and prompt construction, then reuses every completed context/repeat checkpoint under `research_outputs/query-generation-checkpoints`; only missing response cells are generated. Model, prompt, persona, memory mode, architecture parameters, or generation-configuration changes produce different fingerprints and cannot mix outputs. Initial post-response exposure judging is likewise checkpointed atomically per context/repeat in the reused query run's `exposure_calls/` directory for list, graph, and profile memory. Its fingerprint covers the exact response, attributes, scenario identity, and judge configuration; successful calls are reused while failed calls are retried. With `--labels-file`, strictly import the matching persona's embedded Qwen labels for every context; missing or incompatible coverage aborts before response generation and never falls back to label-generation calls. For reranked list, graph, or profile modes, `--with-pre-rerank` immediately runs the existing resumable pre-rerank response generation, exposure judging, and privacy metrics against the selected pipeline, using the same `--repeats` value and storing the derived stage inside that experiment folder. If the chained stage is interrupted, either rerun the same command with `--resume-compatible` or invoke `/generate_pre_rerank_responses <created-pipeline-path> --repeats N` directly; both reuse completed pre-rerank generation and judging checkpoints. Reranked modes automatically compute resumable pre/post-rerank memory-stage metrics after the final response metrics; pass `--skip-memory-stage-metrics` to opt out.
- ↳ `/run_privacy_pipeline_cimemories_dataset <dataset> <memory> [--repeats N] [--reuse] [--resume-compatible] [--labels-file <labels.json>] [--skip-memory-stage-metrics]`: wrap the full pipeline across every persona, using 10 scenario repeats by default or `N` when specified. `--reuse` copies compatible complete persona pipelines; `--resume-compatible` additionally resumes compatible completed stages of interrupted personas. `--labels-file` validates complete matching label coverage for every persona before paid generation, then propagates the file to every per-persona pipeline. Supplied labels take precedence over whole-pipeline `--reuse`; combine them with `--resume-compatible` to reuse matching paid query/judge stages while recomputing metrics from the supplied labels. For reranked list, graph, and profile modes, pre/post-rerank memory-stage metrics are computed automatically after final metrics and saved at persona and dataset level; checkpoints are reused after interruptions. Pass `--skip-memory-stage-metrics` only when those additional semantic-judge calls are deliberately unwanted. Reused artifacts remain in place and are referenced with provenance.
- ↳ `/compare_memory_modes <dataset> <persona>`: wrap two consecutive full pipelines (modes `0` and `1`), then compare their outputs.
- ↳ `/sweep_archival_search_limit_cimemories <dataset> <persona> [memory]`: wrap repeated full pipelines over limits `5`, `10`, `20`, `50`, `100`, and `200`, caching completed limits and printing per-context and aggregate `C | L | A` matrices. Use `agent-search` (default), `list-search`, or `list-rerank`.

### Memory-export evaluation

- `/run_memory_export_evaluation <dataset> <persona>`: initialize separate list, graph, and profile memory stores for one persona, ask the memory-export prompt against each architecture, and save human-readable Markdown plus prompts, responses, histories, and retrieval metadata under `research_outputs`
- ↳ `/evaluate_memory_export_outputs <memory_export_output_dir>`: judge that export against the original persona facts and save per-architecture retrieved, partial, and omitted recall as JSON and Markdown.

### Reports and publication exports

- `/print_privacy_pipeline_cimemories <pipeline_output_dir>`: print a per-scenario report table for a previous CIMemories pipeline run, including correctly shared attributes and privacy violations
- `/pipeline_runs [--limit N] [--mode MODE|--modes A,B] [--dataset TEXT] [--persona N] [--labels SOURCE] [--best completion|leakage|ambiguous]`: browse recent single-persona CIMemories pipelines, newest first by the creation timestamp recorded in each pipeline, in a compact table with completion status, artifact-derived ground-truth provenance, and macro `C / L / A` metrics. `--labels gpt-5` performs exact model matching and therefore excludes GPT-5.6 labels; file/model substrings such as `qwen` are also accepted. With `--modes`, `--best` selects one best complete run per requested mode from a single strictly matched comparison cohort, using newest-first tie-breaking. Label provenance distinguishes model-generated artifacts such as GPT-5 from externally imported files such as the Qwen labels. Per-metric stars mark the best values across all cataloged runs within strictly matched comparison groups, so filtering or limiting the display cannot create a false winner. The displayed global `@N` references can replace long pipeline paths in inspection and reporting commands; `@latest` selects the newest run and `@mode:<id-or-name>` selects the newest complete run for a memory mode. Filters change which rows are displayed without changing their global `@N` references.
- `/cimemories_experiment_progress [--root DIR] [--dataset TEXT|all] [--model TEXT] [--details] [--gaps-only]`: recursively inventory reranked list, graph, and profile experiments across both standalone persona runs and nested dataset runs. Its compact default matrix shows post/pre persona coverage, direct retrieved-memory evaluation before message generation, saved iteration counts, measured mean generation seconds per post/pre response, and a recorded-token USD cost estimate for each model and architecture. The memory-stage column reports completed persona coverage plus the deterministic exact-match method or semantic judge model, and recognizes both legacy and judge-namespaced summaries. Add `--details` for every persona or `--gaps-only` for only the cells needing work; overall completion requires post, pre, and memory-stage coverage. It connects derived evaluations only to their selected source pipeline, and interrupted retries are not double-counted: the dashboard selects the strongest compatible artifact for each model/architecture/persona cell. Costs use provider rates centralized in the CLI, apply cached-input rates when saved telemetry supplies them, and carry `*` whenever unmetered backend or embedding work prevents invoice-level completeness. The default root is `research_outputs`, the default dataset filter is `cimemories_raw`, and new models appear automatically from manifest metadata; unknown model prices remain unpriced rather than silently inheriting another model's rate.
- `/cimemories_comparative_report [--root DIR] [--dataset TEXT|all] [--output DIR] [--inputs FILE] [--bootstrap-iterations N]`: build a paper-level HTML report across generation models, list/graph/profile memory, and pre/post-reranking stages. It automatically selects the strongest complete artifact for each model/architecture/persona cell, pairs all 49 contexts, reports hierarchical-bootstrap intervals, separates full-cohort results from cross-model matched-persona comparisons, and includes latency, recorded-token cost, and model-role provenance. The report directory also contains analysis-ready CSV/JSON files and a locked `comparative_report_inputs.json`; pass that file back with `--inputs` to regenerate from exactly the same source manifests. This command only reads existing experiment artifacts and makes no model calls. The older report builders remain available for single-run and single-model inspection.
- `/export_cimemories_study_artifacts [--root DIR] [--dataset TEXT|all] [--models A,B,C] [--output DIR] [--inputs FILE] [--bootstrap-iterations N] [--expected-personas N] [--require-complete]`: export the factorial paper bundle for the model × list/graph/profile × pre/post-reranking study while preserving all older exporters. It reports absolute operating points, paired reranking effects, architecture contrasts within each model/stage, model contrasts within each architecture/stage, and difference-in-differences for model-by-reranking and architecture-by-reranking interactions. Every pairwise test uses its exact shared persona/context cohort and records its own sample size; interim exports expose incomplete coverage, while `--require-complete` refuses to export until every selected model/architecture condition has the expected persona count (10 by default). Use comma-separated case-insensitive `--models` substrings to exclude unrelated smoke models; each substring must identify exactly one saved model. Outputs include a self-contained HTML report, generated SVG and LaTeX figures/tables for paper preparation, long-form CSVs, discarded-context sensitivity, persona and recipient/task-domain heterogeneity, memory-stage readiness, and a locked `study_artifact_inputs.json` for exact regeneration with `--inputs`. The memory-stage inventory and explicit attribute-flow schema provide a stable bridge for the planned initialized → retrieved → reranked → exposed analysis. This command is offline and makes no model calls.
- `/export_cimemories_integrated_paper_artifacts [--root DIR] [--dataset TEXT|all] [--models A,B,C] [--output DIR] [--inputs FILE] [--bootstrap-iterations N] [--expected-personas N] [--memory-persona N] [--memory-judge-model MODEL] [--first-post-repeat] [--require-complete] [--require-memory-complete]`: export the integrated paper bundle that preserves the complete factorial response analysis from `/export_cimemories_study_artifacts` and adds direct returned-memory plus memory-to-response analyses. The response layer retains architecture comparisons, model comparisons, paired pre/post-reranking effects, difference-in-differences, heterogeneity, efficiency, and provenance. Use `--first-post-repeat` for the matched-repeat sensitivity analysis: post-rerank response metrics are recomputed offline from the lowest saved `repeat_idx` in every context, so GPT-5.6-sol contributes one generation just like the other models; pre-rerank and direct-memory metrics are unchanged. The default still uses all saved repetitions for backward compatibility, and the selected policy is recorded in the lockfile, manifest, README, and HTML report. A compact architecture-efficiency table macro-averages model-persona pipeline cells and reports retrieval/reranking preparation time, generation time, deployed latency, exact reported tokens per query, and one-time initialization time; token means use the cross-architecture matched telemetry cohort, while raw pipeline-cell measurements and coverage remain available as CSV. The direct-memory layer defaults to persona 0 and the `gpt-6-sol` production semantic judge, uses deterministic exact matching for list memory, and exports pre/post availability, retention, paired architecture/model contrasts, reranking effects, and architecture/model difference-in-differences for necessary, private, and ambiguous attributes. Cross-stage artifacts join the exact model/architecture/persona/context cells to compare memory availability with response exposure without treating the relationship as causal. Outputs are organized under `response_analysis/`, `efficiency_analysis/`, `memory_analysis/`, and `cross_stage_analysis/`, with an integrated HTML report, direct-memory trajectory, representation-density and memory-to-response SVGs, generated LaTeX tables/snippets for paper preparation, context-level CSVs, and `integrated_artifact_inputs.json` locking every pipeline and memory-stage summary. During the offline export, `letta-chat` prints throttled progress bars for source loading, all response and direct-memory bootstrap estimates, figure rendering, cross-stage joining, and final artifact writing. Use `--require-memory-complete` to refuse partial nine-cell memory case studies. Regeneration with `--inputs` uses the exact locked sources. The command is offline and makes no model calls; all older exporters remain unchanged.

For anonymous artifact review, create a separate non-destructive release copy from
the integrated export's locked inputs. The exporter follows the selected pipeline,
post-response, pre-rerank, and memory-stage directories; includes a sanitized copy
of the paper report; rewrites absolute paths; removes provider, agent, conversation,
run, trace, credential, timestamp, and fingerprint metadata; and refuses unsupported
binary files or a destination that already exists. It retains the synthetic persona
facts because those are the scientific data being evaluated. Every export ends with
an automated leak scan recorded in `anonymization_report.json` and SHA-256
checksums for every exported file:

```bash
python3 scripts/export_anonymized_cimemories_artifacts.py \
  research_outputs/cimemories-integrated-paper-artifacts_20260929_000041 \
  --scope github \
  --output research_outputs/cimemories-anonymous-artifact
```

Use `--dry-run` first to validate all locked inputs and print the expected file count
and size without writing anything. `--scope github` keeps the complete aggregate
paper report, sanitized provenance, and one context-0 response/judgment sample per
pipeline; `--scope paper` keeps normalized response, judgment, labeling, metric,
and memory-summary evidence for every evaluated cell while dropping redundant
histories and individual API-call envelopes; `--scope full` retains the prior
lossless behavior and remains the default for backward compatibility.
Verify a completed bundle with
`python3 scripts/verify_anonymized_cimemories_artifact.py <artifact-directory>`.
- `/export_cimemories_ground_truth_artifacts --ground-truth-file FILE [--ground-truth-name NAME] [same integrated-export options]`: create a separate sensitivity bundle under a caller-supplied CIMemories ground-truth file while preserving the legacy Qwen-based exporter. The command reads the generic `labels_combined` field, strictly validates personas, scenarios, and attribute universes, and reuses saved generations, exposure judgments, retrieval/reranking results, and direct-memory attribute matches. It recomputes label-dependent response, returned-memory, and cross-stage metrics plus bootstrap intervals entirely offline; total disclosed-attribute counts remain unchanged by construction. The bundle records the absolute label path and SHA-256 checksum and adds `ground_truth_analysis/summary.json` plus `response_reranking_effect_comparison.csv` for original-versus-supplied-label point estimates. Use `--inputs` with an existing integrated lockfile to keep the evaluated pipeline cohort identical.
  After every table, the command prints a dynamic reference-to-mode map and a diagnostic example built from up to the first three displayed rows, such as `@15=list-rerank, @17=graph-rerank, @1=profile-locomo-rerank` followed by `/diagnose_privacy_pipeline_cimemories @15 @17 @1`. Generic reference syntax remains `/diagnose_privacy_pipeline_cimemories @1 @2 @3`.
- `/diagnose_privacy_pipeline_cimemories <pipeline_output_dir> <pipeline_output_dir> [...] [--context N ... | --top N]`: compare completed architectures using their saved artifacts. It verifies whether datasets, personas, models, repeats, and exact ground-truth paths match; reports whether labels were model-generated (for example GPT-5) or imported from an external labels file (for example Qwen); prints the full per-scenario `completion / leakage / ambiguous exposure` matrix with per-category winner markers and a macro-winner summary; automatically drills into the largest completion gaps; and aligns each necessary attribute with candidate memory, selected memory, exposure frequency, and a clearly marked heuristic failure stage. Use one or more `--context N` options for targeted drilldowns, or `--top N` to change the automatic count.
- `/print_memory_queries_cimemories <pipeline_output_dir>`: print every archival memory search query performed during a previous CIMemories pipeline run, the returned memories, and the necessary attributes from that scenario's context-labeling output; also saves Markdown and JSON reports next to the pipeline manifest
- `/compare_privacy_pipeline_cimemories <pipeline_output_dir> <pipeline_output_dir> [...]`: compare two CIMemories pipeline runs side by side, or three or more runs in a per-context `C | L | A` matrix with an aggregate summary
- `/latex_privacy_pipeline_cimemories <pipeline_output_dir> [...]`: validate one or more completed CIMemories pipeline runs and write a timestamped `booktabs` LaTeX table. One run produces recipient-wise completion, private-leak, and ambiguous-exposure rates plus an aggregate row. Multiple runs produce one recipient row with grouped `C | L | A` columns per architecture and an emphasized aggregate row at the bottom. For every recipient and the aggregate, highest completion and lowest private/ambiguous exposure are bold; ties are all bold. The command reads saved metrics only and makes no model calls. Include `\usepackage{booktabs}` in the paper preamble
- `/latex_efficiency_pipeline_cimemories <compact|detailed> <pipeline_output_dir> [...]`: validate completed runs and write a timestamped token/runtime `booktabs` table without making model calls. `compact` produces one architecture row with exact deployed tokens/query, generation and preparation time, deployed mean/p95 latency, initialization time, and experiment wall time. `detailed` produces recipient rows with grouped exact-token and deployed mean/p95 columns per architecture, followed by a strongly separated aggregate row. The lowest value in each compact column, detailed recipient metric, and detailed overall metric is bold; ties are all bold. Backend-internal initialization tokens remain excluded because they are not reported consistently
- `/export_privacy_paper_artifacts <dataset_run> [...] [--baseline MODE] [--exclude-discarded-contexts] [--output DIR]`: build an offline paper bundle from matching completed full-dataset runs. For the recommended paper-oriented complete-case analysis, `--exclude-discarded-contexts` removes persona-context cells that lack either necessary or inappropriate ground-truth attributes; the same labeling-derived exclusion mask must match across architectures, and its policy and counts are recorded in the manifest. Without the flag, the legacy all-cells aggregation is retained for reproducibility. The bundle contains effectiveness and efficiency `booktabs` tables, paired hierarchical-bootstrap intervals, aggregate/persona/context/delta CSVs, per-persona efficiency data, context and persona figures, privacy-utility and effectiveness-efficiency plots, and a provenance manifest. Efficiency output separates online query cost, one-time initialization, experiment wall time, and evaluation overhead; missing backend-internal token telemetry is marked unavailable rather than counted as zero. Publication-ready LaTeX tables and captioned figure snippets are placed in `includes/`, while dependency-free vector SVGs are placed in the sibling `figures/` directory; shared Overleaf preamble and all-figures include files are provided. Runs can also be selected automatically with `--dataset TEXT --labels SOURCE --modes A,B,...`; this chooses the newest matching complete dataset run for each mode and prints the exact selected paths before exporting.
- `/export_memory_snapshots <pipeline_path> [--export-full-graph] [--output DIR]`: export a completed full-dataset run, one-persona privacy pipeline, or query run as a read-only memory inspection bundle; indexed pipeline references such as `@1` are accepted where applicable. The bundle contains a browsable `index.html`, an architecture-flow SVG, initialization JSON for every persona, compact retrieval JSON for every scenario, a CSV index, and source-run provenance. List and profile exports additionally contain a self-contained, searchable, print-friendly `personas/persona_NNN/memory/full_memory.html` for each persona; these files need no web server or external assets and can be opened directly on Windows. A prepared retrieval snapshot is stored once per persona and scenario because every response repetition reuses it. Provider response envelopes and duplicate reranker prompts are omitted, while memory candidates, facts, provenance, scores, and selections are retained. List exports retain atomic statements and reranking records; profile exports retain the saved consolidated backend profile and its flattened retrieval candidates. Graph exports always retain logged facts, UUIDs, episode provenance, and reranking. For graph runs, optional `--export-full-graph` additionally performs read-only queries against the currently configured Neo4j database and writes each persona's complete live Graphiti group as JSON, GraphML, a static SVG, and a self-contained interactive `full_graph.html`, plus a compact selected-fact SVG for every scenario. The interactive view initially shows only semantic entity relationships; it supports search, node-neighborhood inspection, fact/property details, zooming, panning, and an optional episode/`MENTIONS` provenance layer. It opens directly on Windows without external assets. The option fails rather than silently reconstructing a graph when a recorded group no longer exists; embedding vectors are omitted from exports.

Every CIMemories query/pipeline run also captures the complete backend-observed memory immediately after initialization and before scenario retrieval. The query directory's `initialized_memory/` folder contains a manifest and machine-readable JSON plus a searchable offline HTML view for list/profile memory; graph memory additionally includes interactive HTML, SVG, and GraphML. Capture completeness is checked before paid response generation. Embeddings are recursively omitted by default; pass `--include-embeddings` to either `/run_privacy_pipeline_cimemories` or its dataset variant only when vector-level debugging is necessary.
- `/export_pre_rerank_candidates <pipeline_path> [--output DIR]`: export every saved fact available immediately before reranking for a completed list-rerank, graph-rerank, or profile-rerank run. It accepts dataset, persona-pipeline, query-run, and indexed `@N` references. The portable report contains a searchable HTML browser for every persona/scenario, exact per-context JSON, long-form CSV and JSONL files, selection markers, native graph/profile provenance, candidate-cap indicators, and an explicit completeness warning for older artifacts whose full candidate set cannot be reconstructed. For mode 19, the fair architecture-level scope is exported: all preserved persistent-profile facts plus every query-selected event, with component, reranker-input, and retention markers. Other modes report the reranker input. This is not the complete initialized memory or hidden backend-internal search state.
- `/generate_pre_rerank_responses <pipeline_path> [--pilot-contexts N] [--repeats N] [--generate-only] [--force]`: extend a completed reranked persona or dataset pipeline with a paired pre-rerank response condition without rerunning memory initialization, retrieval, graph construction, profile extraction, or reranking. It requires complete saved candidate sets, replays them through the recorded response model, and always writes the derived stage inside the source experiment as `pre_rerank_response_evaluation/`. Pilot and repeat options select the requested scope rather than creating separate stage directories: a later larger run reuses matching per-context/repeat generation and exposure checkpoints and computes only missing cells. By default it also runs exposure judging and per-context privacy metrics against the original imported labels; `--generate-only` stops after response generation. Generation or judge configuration mismatches stop before incompatible paid outputs can be mixed. The saved architecture-specific response prompt wrapper is preserved and only its memory body is replaced. `--force` deliberately replaces checkpoints in the requested scope and recomputes judgments.
- `/generate_post_rerank_responses <pipeline_path> [--pilot-contexts N] [--repeats N] [--generate-only] [--force]`: expand a completed post-reranking pipeline from a small repeat pilot to a larger repeat count. Existing source responses and matching exposure judgments are imported as checkpoints; later runs generate and judge only missing context/repeat cells. The expandable stage is stored inside the source experiment as `post_rerank_response_evaluation/`, and generation/judge fingerprints prevent outputs from different models or configurations from being mixed.
- `/compute_memory_stage_metrics <pipeline_path> [--judge-model MODEL] [--strategy monolithic|per-attribute|attribute-batch|exact-match|all] [--attribute-batch-size N] [--pilot-contexts N] [--force]`: measure which ground-truth necessary, private, and ambiguous attributes are available before and after reranking. The default `monolithic` strategy preserves the one-call-per-context evaluator. `per-attribute` makes one simpler call per original attribute; `attribute-batch` evaluates configurable small groups (default 4); and `all` runs those three judge strategies into separate checkpoint and summary names for comparison. The opt-in `exact-match` strategy is restricted to list memory, makes no judge/API calls, strips only surrounding whitespace, and aborts if original attributes are duplicated or any saved candidate cannot be mapped exactly. Targeted prompts serialize every memory fact once with a post-rerank retention flag, place the changing attribute block at the end for prefix caching, save resumable per-call artifacts, and aggregate back into the existing pre/post metric schema. `--pilot-contexts N` deterministically limits the run to the first N contexts across the supplied pipeline or dataset. The command accepts dataset runs, persona pipelines, and `/pipeline_runs` references such as `@3`; summaries contain macro and micro aggregates, exact judge-token usage when supplied by the provider, and both all-context and context-discarded-excluded views. Use `/compute_memory_stage_metrics --missing --architecture list|graph|profile [--model TEXT] [--dataset TEXT|all] [--root DIR]` to reuse the progress dashboard's duplicate-selection rule and process only selected pipelines without a complete memory-stage summary. Batch mode prints every selected pipeline before starting, continues after individual failures, and emits exact recovery commands; `--strategy exact-match` is guarded to list architecture. LLM-judged artifact paths include the judge model and a non-secret configuration fingerprint so multiple judges coexist; summaries record endpoint, API style, reasoning, token ceiling, timeout, and credential source but never the credential value. Dedicated `LETTA_MEMORY_STAGE_JUDGE_*` settings fall back independently to the generic judge configuration. Existing matching checkpoints are reused; use `--force` only to deliberately pay for recomputation.

To backfill `A` for saved OpenAI evaluations without making model calls, first review the dry run and then apply it:

```bash
python3 backfill_ambiguous_exposure.py
python3 backfill_ambiguous_exposure.py --apply --regenerate-tables --quiet-reports
```

Before changing anything, the apply command creates one compressed backup under `backfill_backups/`. The archive contains every JSON and Markdown file that will be replaced plus a manifest mapping each archived member to its original absolute path. Keeping the originals inside an archive outside `research_outputs` prevents report scanners from treating them as additional evaluations. Use `--backup-dir <directory>` to choose another location. The command then atomically adds ambiguous/unknown exposure fields to existing per-context metric JSON files and replaces recognized comparison Markdown files with regenerated `C | L | A` tables. Pass `--all-models` to include non-OpenAI pipeline metrics as well.

New privacy-pipeline runs also save architecture-efficiency measurements. Monotonic timings separate agent creation, one-time memory initialization, per-context retrieval/reranking, response generation, history retrieval, concurrent batch wall time, and evaluator overhead. Reports show mean and nearest-rank p95 deployed online latency; this adds memory preparation to each response even though repeated evaluation generations share one prepared context. Token accounting uses provider-reported usage when present and records exact-usage coverage. A versioned provider-independent visible-token estimate is retained as a size proxy, but is never presented as billing usage.

Every new pipeline manifest contains a versioned `pipeline_usage_ledger`, and `pipeline_token_usage.md` renders the same information as a per-component table. Components distinguish response generation, explicit reranking, Graphiti extraction/embeddings/internal reranking, Letta archival insertion/search embeddings, Memobase extraction/context construction, context labeling, and exposure judging. `incremental_exact_total` excludes compatible reused artifacts; `logical_exact_total` represents the complete logical workload. Missing backend telemetry is marked `unavailable` or `partial` and is never silently counted as zero. Graphiti OpenAI-compatible calls are captured before Graphiti discards their response usage. Memobase and Letta backend counters are used when their deployments expose them; otherwise their raw telemetry snapshots and the reason for missing coverage remain in the ledger. Compact efficiency LaTeX tables include logical pipeline tokens and component coverage, while recipient-wise tables retain deployed online tokens because shared initialization and evaluation work cannot be assigned reliably to one recipient.

Batch-oriented commands use bounded concurrency for independent requests. You can tune this with `LETTA_RESEARCH_ARCHIVAL_CONCURRENCY`, `LETTA_RESEARCH_QUERY_CONCURRENCY`, and `LETTA_RESEARCH_OPENAI_CONCURRENCY`.
Compatible per-response pipeline checkpoints default to `research_outputs/query-generation-checkpoints`; set `LETTA_QUERY_GENERATION_CHECKPOINT_DIR` only when a different persistent cache location is required.

### Memory choices

Use readable names in new commands. Run `/memory_modes` inside `letta-chat` for
the same quick reference. Numeric IDs remain accepted as **legacy aliases** so
old scripts and experiment commands continue to work.

| Family | Memory name | Legacy | Behavior |
|---|---|---:|---|
| List | `all` | `0` | Inject every archival memory. |
| List | `agent-search` | `1` | Let the Letta agent call archival search. |
| List | `list-search` | `2` | Search list memory in the CLI and inject the results. |
| List | `list-rerank` | `3` | Retrieve a candidate pool, rerank it, and inject the selection. |
| Graph | `graph` | `4` | Ingest raw Graphiti episodes and retrieve graph facts. |
| Graph | `graph-normalized` | `5` | Normalize and batch facts before Graphiti retrieval. |
| Graph | `graph-rerank` | `6` | Retrieve normalized graph candidates and contextually rerank them. |
| Attacker | `attacker-search` | `7` | Simulate web search over the isolated attacker RAG file. |
| Attacker | `attacker-inject` | `8` | Inject the isolated attacker RAG file directly into the prompt. |
| Profile | `profile` | `9` | Ingest raw statements and retrieve Memobase profile context. |
| Profile | `profile-normalized` | `10` | Ingest third-person normalized profile facts. |
| Profile | `profile-domain` | `11` | Ingest related facts in domain batches. |
| Profile | `profile-labeled` | `12` | Add stability and event labels to domain batches. |
| Profile | `profile-schema` | `13` | Add schema-guided keys, domains, events, and values. |
| Profile | `profile-generic` | `14` | Ingest generic normalized, domain-grouped profile documents. |
| Profile | `profile-json` | `15` | Flatten and rank extracted profile JSON instead of generic context. |
| Profile | `profile-locomo` | `16` | Use the LoCoMo-style profile configuration and retrieval settings. |
| Profile | `profile-locomo-rerank` | `17` | Flatten the complete LoCoMo profile into atomic facts and rerank up to 200 candidates down to 20. |
| Profile | `profile-locomo-context-rerank` | `18` | Ask Memobase for query-dependent LoCoMo context, split its returned bullets into atomic facts, then rerank up to 200 candidates down to 20. |
| Profile | `profile-locomo-events-rerank` | `19` | Preserve Memobase's complete current profile and rerank only its query-selected Past Events, retaining at most 20 events. |

For example:

```text
/run_privacy_pipeline_cimemories data.json 0 agent-search
/run_privacy_pipeline_cimemories data.json 0 graph-rerank
/run_privacy_pipeline_cimemories data.json 0 profile-json
/run_privacy_pipeline_cimemories data.json 0 profile-locomo-rerank
/run_privacy_pipeline_cimemories data.json 0 profile-locomo-context-rerank
/run_privacy_pipeline_cimemories data.json 0 profile-locomo-events-rerank
```

New manifests retain the stable numeric `memory_mode` field and also include
`memory_mode_name`, making results both backward-compatible and self-explanatory.

## Automation usage

See `examples/automation_example.py` for non-interactive usage. The key idea:
- Import `AgentClient`, `ConversationClient`, `MemoryClient`
- Create/resume agent id
- Create a conversation id for each experiment run
- Insert/search archival memory as needed

## Notes / Server Compatibility

Letta deployments can differ:
- Some servers require `agent_id` as a query param for `/v1/conversations` endpoints (this repo assumes that behavior).
- If conversations API is unavailable (404), the CLI automatically falls back to agent-thread mode.
