# CIMemories DeepSeek V4 Reproduction Guide

This guide reproduces the DeepSeek comparison against the existing
GPT-5.6-sol and GLM-5.3-Flash CIMemories experiments. It evaluates list,
graph, and profile memory before and after contextual reranking.

## Fixed experiment design

| Role | Configuration |
|---|---|
| Response model | `deepseek-ai/DeepSeek-V4-Flash-0731` through Together |
| List/graph contextual reranker | DeepSeek V4, `high` reasoning |
| Graphiti extraction and internal reranker | DeepSeek V4 through Together |
| Exposure judge | `gpt-5.2` through OpenAI, reasoning `none` |
| Direct returned-memory judge | `gpt-6-sol` through OpenAI; used only by `/compute_memory_stage_metrics` for graph and profile memory |
| List and graph embeddings | `text-embedding-3-large`, 3072 dimensions |
| Memobase embeddings | `text-embedding-3-small`, 1536 dimensions |
| Reranker candidates/output | 50 / 20 |
| Contexts per persona | 49 |
| Response repetitions | 1 |
| Personas | 10, indices 0 through 9 |
| Labels | `labels_qwen.json` |

The non-secret shell configuration is stored in
[`experiment_templates/cimemories_deepseek_v4.env`](../experiment_templates/cimemories_deepseek_v4.env).
The launcher is
[`scripts/start_cimemories_deepseek_v4.sh`](../scripts/start_cimemories_deepseek_v4.sh).

## 1. Verify the Together model

The model should be tested directly before the experiment. This command does
not print the API key:

```bash
curl -sS https://api.together.xyz/v1/chat/completions \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "messages": [
      {"role": "user", "content": "Reply with exactly: DEEPSEEK_OK"}
    ],
    "max_tokens": 512,
    "temperature": 0
  }' |
jq '{
  model,
  content: .choices[0].message.content,
  finish_reason: .choices[0].finish_reason,
  usage
}'
```

Expected visible content:

```text
DEEPSEEK_OK
```

## 2. Configure and restart Memobase

Memobase is a separate service and does not inherit the environment loaded by
the `letta-chat` launcher. Edit:

```text
${MEMOBASE_ROOT}/src/server/api/config.yaml
```

Set `MEMOBASE_ROOT` to the directory containing the pinned Memobase checkout.

Set its LLM fields to:

```yaml
llm_style: openai
llm_base_url: https://api.together.xyz/v1
llm_api_key: YOUR_TOGETHER_KEY

best_llm_model: deepseek-ai/DeepSeek-V4-Flash-0731
thinking_llm_model: deepseek-ai/DeepSeek-V4-Flash-0731
summary_llm_model: deepseek-ai/DeepSeek-V4-Flash-0731
```

Keep the established Memobase embedding configuration unchanged:

```yaml
enable_event_embedding: true
embedding_provider: openai
embedding_base_url: https://api.openai.com/v1
embedding_api_key: YOUR_OPENAI_KEY
embedding_model: text-embedding-3-small
embedding_dim: 1536
```

Do not change Memobase to 3072 dimensions: its existing database schema uses
1536-dimensional embeddings.

Apply `patches/memobase-v0.0.42-reasoning-models.patch` to the pinned Memobase
commit documented in `ARTIFACT.md`. The complete patch updates the startup
sanity check in `memobase_server/llms/__init__.py` and the provider adapter in
`memobase_server/llms/openai_model_llm.py`. It includes
`deepseek-ai/DeepSeek-V4-Flash-0731` in the Together reasoning-model branch,
which raises the completion ceiling from Memobase's 1024-token default to
16384 so internal reasoning cannot consume the entire response before visible
profile-extraction output is emitted.

Memobase source code is baked into the API image, while only `config.yaml` is
bind-mounted. Rebuild and recreate only the API service after a code change:

```bash
cd "$MEMOBASE_ROOT/src/server"
docker compose up -d --build memobase-server-api
docker compose logs --tail=80 memobase-server-api
```

A configuration-only YAML change can instead use
`docker compose restart memobase-server-api`.

Confirm that the log contains equivalents of:

```text
Embedding dimension matched: 1536
LLM sanity check passed
Start Memobase Server
```

## 3. Export credentials

In the shell that will launch the evaluation, export the credentials without
printing them:

```bash
export OPENAI_API_KEY="..."
export TOGETHER_API_KEY="..."
export MEMOBASE_API_KEY="secret"
```

The existing Neo4j credentials must also remain available, especially
`GRAPHITI_NEO4J_PASSWORD` if it is not supplied elsewhere.

The launcher refuses to start when `OPENAI_API_KEY` or `TOGETHER_API_KEY` is
empty. It deliberately never prints credential values. Memobase and Neo4j
credentials are validated by the backend check.

## 4. Launch the experiment environment

```bash
cd /path/to/anonymous-artifact-repository
scripts/start_cimemories_deepseek_v4.sh
```

This starts a fresh `letta-chat` process with the complete DeepSeek agent,
reranker, Graphiti, OpenAI judge, and embedding configuration. Graphiti is a
Python library loaded inside this process; only Neo4j is a separate background
service. Consequently, changing Graphiti models requires restarting
`letta-chat`, not Neo4j.

To configure a shell without launching `letta-chat`, use:

```bash
source experiment_templates/cimemories_deepseek_v4.env
```

## 5. Validate backend routing

Inside `letta-chat`, run:

```text
/check_local_backends
```

Verify the following routing:

| Component | Expected endpoint/model |
|---|---|
| Agent LLM | Together / DeepSeek V4 |
| Exposure judge | OpenAI / GPT-5.2 |
| Direct returned-memory judge | OpenAI / `gpt-6-sol` |
| Contextual reranker | Together / DeepSeek V4 |
| Graphiti LLM | Together / DeepSeek V4 |
| Graphiti embeddings | OpenAI / `text-embedding-3-large`, 3072 |
| Graphiti reranker | Together / DeepSeek V4 |
| Memobase | Reachable at `http://localhost:8019` |

Optionally inspect the sanitized Memobase configuration:

```text
/memobase_server_config
```

## 6. Calibrate the reranker

```text
/calibrate_reranker cimemories_raw.json 0 --pilot-contexts 10 --reasoning-efforts high --max-tokens 2048
```

Proceed with the configured 2048-token ceiling if all ten responses are valid.
If responses fail specifically because reasoning exhausts the output budget,
change `LETTA_RERANK_MAX_TOKENS` in the environment template to `4096`, restart
`letta-chat`, and repeat the calibration. Do not change the reasoning effort or
prompt mode without evidence from the calibration.

## 7. Run persona 0 as a reusable pilot

```text
/run_privacy_pipeline_cimemories cimemories_raw.json 0 list-rerank,graph-rerank,profile-locomo-events-rerank --repeats 1 --labels-file labels_qwen.json --skip-memory-stage-metrics --resume-compatible --with-pre-rerank
```

This pilot is part of the final evaluation rather than disposable work. It
generates both post-reranking and pre-reranking response evaluations.

Check its status with:

```text
/cimemories_experiment_progress --model DeepSeek --details
```

Persona 0 should be complete at both stages for list, graph, and profile.

## 8. Run personas 1 through 9

```text
/run_privacy_pipeline_cimemories cimemories_raw.json 1,2,3,4,5,6,7,8,9 list-rerank,graph-rerank,profile-locomo-events-rerank --repeats 1 --labels-file labels_qwen.json --skip-memory-stage-metrics --resume-compatible --with-pre-rerank
```

If a request times out, relaunch the DeepSeek environment and rerun the exact
same command. `--resume-compatible` reuses compatible completed pipeline work,
and the pre-reranking stage has its own checkpoints.

## 9. Verify completion

```text
/cimemories_experiment_progress --model DeepSeek --details
```

The intended final coverage is:

```text
DeepSeek list:    post 10/10, pre 10/10
DeepSeek graph:   post 10/10, pre 10/10
DeepSeek profile: post 10/10, pre 10/10
```

The progress command derives model identity and experimental settings from the
saved manifests. DeepSeek runs are stored separately from GPT-5.6-sol and GLM
runs because model and configuration fingerprints differ.

## Reproduction cautions

- Never replace `OPENAI_API_KEY` with the Together key. Role-specific Together
  variables route agent, reranker, and Graphiti calls independently.
- Never leave `GRAPHITI_INGEST_LIMIT` set for a full run. The launcher unsets
  it explicitly.
- Do not change embedding models or dimensions between personas.
- Do not reuse an already-running `letta-chat` process after changing the
  environment; relaunch it through the script.
- Do not restart Neo4j merely to change the Graphiti model.
- Memobase configuration changes require restarting `memobase-server-api`.
- Keep `labels_qwen.json`, retrieval limits, judge settings, and repetition
  count fixed for comparability with the earlier experiments.
