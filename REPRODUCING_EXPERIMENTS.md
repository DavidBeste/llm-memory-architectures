# Reproducing the CIMemories experiments

This is the authoritative, end-to-end reviewer guide for reproducing the
reported CIMemories model by memory-architecture by reranking experiment. It
separates the free verification path from paid experiment replication and
documents the fixed configuration, matched analysis cohort, and recovery steps
that are easy to miss when reading individual command descriptions.

## 1. Choose the reproduction level

### A. Verify the submitted evidence (recommended first)

This path is offline and makes no model calls:

```bash
python3 scripts/verify_anonymized_cimemories_artifact.py artifact/results
```

A valid compact bundle reports 90 pipelines: three response models, three
memory architectures, and ten personas. Open
`artifact/results/paper_report/REPORT.html` to inspect the results. The compact
GitHub bundle contains all aggregate tables and representative samples; use the
separate `paper`-scope release asset when recomputing aggregates from every raw
response is required.

### B. Re-run the experiments

This path calls hosted models and incurs provider charges. It also depends on
hosted model aliases that providers may update, so it reproduces the recorded
configuration and analysis procedure but cannot guarantee byte-identical text
from future API calls.

## 2. Install the client

The core client and offline verification support Python 3.10. Use Python 3.11
or newer for the complete experiment environment because the published
Memobase SDK used by live profile-memory runs requires Python 3.11+. From the
repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-artifact.txt
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py'
```

An offline-only reviewer may substitute `python3.10`; the environment marker
in `requirements-artifact.txt` then omits the Memobase SDK. The unit tests and
artifact verifier mock or avoid that external service boundary. Python 3.10
must not be used for live profile-memory reproduction.

Source archives such as ZIP downloads may not preserve executable permissions.
The commands below invoke repository shell scripts with `bash` so they work
without changing file modes. To use direct invocation instead, run
`chmod +x scripts/*.sh` once after extracting the archive.

## 3. Restore the fixed inputs

The released dataset contains synthetic personas. Copy the sanitized dataset
and fixed context labels to the conventional filenames used by the recorded
runs:

```bash
cp artifact/results/inputs/dataset.json cimemories_raw.json
cp artifact/results/inputs/context_labels.json labels_qwen.json
```

Do not regenerate the context labels during replication. The reported runs use
the released `labels_qwen.json` as fixed ground truth for all three response
models and all three memory architectures.

## 4. Export credentials without committing them

All three experiment families use OpenAI for embeddings and evaluation. GLM
and DeepSeek additionally use Together for response generation, reranking, and
Graphiti extraction.

```bash
export OPENAI_API_KEY="..."
export TOGETHER_API_KEY="..."
export MEMOBASE_API_KEY="secret"
export GRAPHITI_NEO4J_PASSWORD="password"
```

The repository templates contain no credentials. Never write rendered keys
back into a tracked template or paste them into an issue or artifact report.

## 5. Start the pinned Letta and Neo4j services

The default images in `docker-compose.yml` are pinned by digest. Export the
credentials before creating the Letta container because Docker Compose passes
them into the service:

```bash
docker compose up -d
docker compose ps
```

Expected local endpoints are Letta at `http://localhost:8283/v1` and Neo4j at
`bolt://localhost:7687`. The reported Neo4j user is `neo4j`; its password must
match `GRAPHITI_NEO4J_PASSWORD`.

## 6. Install and patch the pinned Memobase server

Memobase is a separate service and does not inherit the model environment of
`letta-chat`:

```bash
git clone https://github.com/memodb-io/memobase.git
export MEMOBASE_ROOT="$PWD/memobase"
git -C "$MEMOBASE_ROOT" checkout 358c16bbc6d687937d79bc2f984a11c3be8da901
git -C "$MEMOBASE_ROOT" apply \
  "$PWD/patches/memobase-v0.0.42-reasoning-models.patch"
cp "$MEMOBASE_ROOT/src/server/.env.example" \
  "$MEMOBASE_ROOT/src/server/.env"
```

The patch contains both tracked source changes used by the experiments:
`memobase_server/llms/__init__.py` increases the startup sanity-check budget
and requests a short visible answer, while
`memobase_server/llms/openai_model_llm.py` handles reasoning-model parameters,
budgets, and empty visible output. Local `config.yaml.*` backups and `.orig`
files are deliberately excluded.

The copied `.env` supplies the database, Redis, port, project, and access-token
settings required by Memobase's Docker Compose file. Its default
`ACCESS_TOKEN="secret"` matches the `MEMOBASE_API_KEY="secret"` exported above.

Before each model family, copy the corresponding sanitized configuration,
replace only its credential placeholders locally, and rebuild the API service:

| Response-model family | Memobase template | Memobase internal LLM |
|---|---|---|
| GPT-5.6-sol | `gpt-5.6-sol.config.yaml.template` | `gpt-4o-mini` |
| GLM-5.3-Flash | `glm-5.3-flash.config.yaml.template` | GLM-5.3-Flash |
| DeepSeek-V4-Flash | `deepseek-v4-flash.config.yaml.template` | DeepSeek-V4-Flash |

For example:

```bash
cp artifact_configs/memobase/deepseek-v4-flash.config.yaml.template \
  "$MEMOBASE_ROOT/src/server/api/config.yaml"
# Replace REPLACE_WITH_* placeholders in the untracked rendered file.
cd "$MEMOBASE_ROOT/src/server"
docker compose up -d --build memobase-server-api
docker compose logs --tail=80 memobase-server-api
cd -
```

Confirm that the Memobase log reports a 1536-dimensional
`text-embedding-3-small` embedding configuration, `LLM sanity check passed`,
and server startup. Do not reuse a Memobase server configured for another model
family.

## 7. Load one exact model environment

Use a fresh shell or relaunch `letta-chat` whenever changing model families:

```bash
bash scripts/start_cimemories_gpt_5_6_sol.sh
# or
bash scripts/start_cimemories_glm_5_3.sh
# or
bash scripts/start_cimemories_deepseek_v4.sh
```

Inside `letta-chat`, run:

```text
/check_local_backends
/memobase_server_config
```

The model-specific environment files are the source of truth for endpoints,
reasoning effort, structured-output mode, token ceilings, timeouts, retrieval
limits, and concurrency:

- `experiment_templates/cimemories_gpt_5_6_sol.env`
- `experiment_templates/cimemories_glm_5_3.env`
- `experiment_templates/cimemories_deepseek_v4.env`

In every family, list and graph retrieval use at most 50 reranker candidates
and 20 selected outputs. Profile mode preserves the persistent Memobase profile
and reranks only the query-selected Past Events, retaining at most 20 events.
`GRAPHITI_INGEST_LIMIT` must remain unset.

## 8. Generate post- and pre-rerank responses

The helper prints the exact commands without making network calls:

```bash
bash scripts/reproduce_cimemories_experiments.sh gpt responses
bash scripts/reproduce_cimemories_experiments.sh glm responses
bash scripts/reproduce_cimemories_experiments.sh deepseek responses
```

Add `--execute` only after the matching Memobase configuration is active and
`/check_local_backends` has passed:

```bash
bash scripts/reproduce_cimemories_experiments.sh deepseek responses --execute
```

The paper-facing comparison uses one post-rerank and one pre-rerank generated
response per model-architecture-persona-context cell. The helper preserves the
recorded pipeline-generation settings; the reporting step in Section 11
selects the matched single-generation cohort from the saved responses.

For GPT, the helper runs one ten-persona dataset pipeline for each
architecture. Record the three printed dataset output directories, then
generate one pre-rerank repetition from each:

```bash
bash scripts/reproduce_cimemories_experiments.sh gpt pre \
  --pipeline research_outputs/<GPT_LIST_DATASET_RUN> \
  --pipeline research_outputs/<GPT_GRAPH_DATASET_RUN> \
  --pipeline research_outputs/<GPT_PROFILE_DATASET_RUN> \
  --execute
```

For GLM and DeepSeek, the response command uses the original matrix form and
`--with-pre-rerank`, so both stages are produced together. Every command uses
`--resume-compatible`; after a timeout, rerun exactly the same command rather
than starting a differently configured experiment.

## 9. Check response-stage completeness

```bash
bash scripts/reproduce_cimemories_experiments.sh gpt status --execute
bash scripts/reproduce_cimemories_experiments.sh glm status --execute
bash scripts/reproduce_cimemories_experiments.sh deepseek status --execute
```

Expected coverage for every model and architecture is `post 10/10` and
`pre 10/10`. Repeat counts shown by the status command describe the responses
stored in each source pipeline; they do not determine the weight of a cell in
the paper-facing matched analysis. Each completed pipeline covers the same 49
scenarios, and the complete response cohort contains 90 distinct
model-architecture-persona pipeline cells. Run replication in a clean checkout
with an initially empty `research_outputs/` directory so automatic report
selection cannot pick up unrelated smoke tests or older retries.

## 10. Reproduce the direct-memory case study

The paper's direct-memory analysis uses persona 0 only, producing nine
model-architecture cells. Identify the selected persona-0 pipeline directory
for list, graph, and profile in each model family, then run:

```bash
bash scripts/reproduce_cimemories_experiments.sh gpt memory \
  --list-pipeline <GPT_PERSONA0_LIST_PIPELINE> \
  --graph-pipeline <GPT_PERSONA0_GRAPH_PIPELINE> \
  --profile-pipeline <GPT_PERSONA0_PROFILE_PIPELINE> \
  --execute
```

Repeat with `glm` and `deepseek`. The helper uses deterministic exact matching
for list memory and the fixed GPT-6-sol monolithic semantic judge for graph and
profile memory. The semantic-judge configuration is OpenAI Responses,
reasoning `none`, and 4096 maximum output tokens. Do not use `--force` when
resuming; successful context-level checkpoints are reusable.

The progress dashboard defines full memory-stage completion as all ten
personas. It can therefore display `1/10` for graph or profile even when the
paper's explicitly scoped persona-0 direct-memory case study is complete.

## 11. Regenerate the paper artifacts

This step is offline and makes no model calls:

```bash
bash scripts/reproduce_cimemories_experiments.sh all report --execute
```

The command requires a complete 3-model by 3-architecture by 10-persona
response matrix and all nine direct-memory cells. It uses
`--first-post-repeat` to select the lowest saved post-rerank repetition in every
model-architecture-persona-context cell, so each response condition contributes
one generation to the matched main comparison. Any additional stored responses
remain available for optional sensitivity analyses but are not treated as
additional independent units in that comparison. Confidence intervals use the
deterministic implementation's 5,000 hierarchical bootstrap resamples.

The generated `integrated_artifact_inputs.json` is the immutable provenance
lock for subsequent regeneration. Pass it back with `--inputs` when rebuilding
the same report rather than allowing automatic run selection.

## 12. Create and verify a release bundle

Substitute the generated integrated-report directory:

```bash
python3 scripts/export_anonymized_cimemories_artifacts.py \
  research_outputs/<INTEGRATED_REPORT_DIRECTORY> \
  --scope github \
  --output research_outputs/cimemories-anonymous-github

python3 scripts/verify_anonymized_cimemories_artifact.py \
  research_outputs/cimemories-anonymous-github
```

The exporter refuses to overwrite an existing directory, rewrites internal
paths to portable `artifact://` references, removes operational identifiers
and credentials, audits the output, and writes `SHA256SUMS`.

## 13. Recovery and interpretation rules

- A repeated command with `--resume-compatible` reuses only configuration-
  compatible initialization, retrieval, reranking, generation, and judgment
  checkpoints.
- Never use `--force` merely to recover from a timeout.
- Never change model, reasoning, embedding, reranking, or label settings inside
  an interrupted run.
- A server timeout does not imply that completed context checkpoints were lost;
  inspect the progress command and rerun the exact command.
- Profile-memory configuration changes require a Memobase API restart. Graphiti
  model changes require relaunching `letta-chat`, not restarting Neo4j.
- Provider-reported token accounting excludes backend operations for which the
  service exposed no telemetry; starred cost totals are therefore estimates,
  not invoices.
