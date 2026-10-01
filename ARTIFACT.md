# CIMemories artifact guide

This repository contains the implementation and experiment configuration for
the paper's model by memory-architecture by reranking evaluation. The companion
anonymous results bundle contains the locked 90-pipeline cohort: three response
models, three memory architectures, and ten synthetic personas.

For a linear fresh-machine-to-paper-results walkthrough, including every paid
run command and the asymmetric response-repeat policy, see
[`REPRODUCING_EXPERIMENTS.md`](REPRODUCING_EXPERIMENTS.md).

## Artifact claims and evidence tiers

The compact `github` bundle is intended for paper-result verification. It
contains every aggregate CSV/JSON table used by the report, sanitized
provenance for every selected pipeline, direct-memory summaries, and one
context-0/repeat-0 response sample per pipeline. It does not contain every raw
response repetition and therefore cannot independently recompute every paper
aggregate.

The optional `paper` bundle retains normalized responses, judgments, labels,
metrics, and memory summaries for every evaluated cell. It should be attached
as a release asset when full raw-evidence recomputation is required. The `full`
scope additionally preserves redundant histories and individual call envelopes
for archival use.

## Verify the submitted results

From the repository root:

```bash
python3 scripts/verify_anonymized_cimemories_artifact.py artifact/results
```

Success reports `"status": "passed"`, verifies every SHA-256 checksum, confirms
that the anonymization audit passed, checks the declared complete experiment
matrix, and validates compact-sample selection. Open
`artifact/results/paper_report/REPORT.html` for the human-readable report. The
analysis-ready values are under its `response_analysis/`, `memory_analysis/`,
`cross_stage_analysis/`, and `efficiency_analysis/` directories.

The exported figures and tables are provided as analysis and visualization
aids. They do not necessarily match the final presentation, layout, or
emphasis conventions used in the paper. The accompanying CSV and JSON files
contain the authoritative numerical results.

## Install and test the code

The core client and offline verification support Python 3.10. Full experiment
reproduction, including the published Memobase SDK used by profile memory,
requires Python 3.11 or newer. The reference direct dependencies are pinned in
`requirements-artifact.txt`; its environment marker skips the service-only
Memobase SDK on Python 3.10:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-artifact.txt
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py'
```

The test suite and report inspection are offline. End-to-end experiment
replication requires provider credentials and can incur API charges.

For an offline-only review on Python 3.10, use the same commands with
`python3.10 -m venv .venv`. The tests mock the external Memobase boundary and
do not require its SDK. Do not use that environment for live profile-memory
replication.

## External services

`docker-compose.yml` defaults to the exact locally inspected images used by the
artifact environment:

- Letta: `letta/letta@sha256:aa66c3eeee13d2dfc40c650d709b550237ee31bfc91942a52fa488a13fa8c102`
- Neo4j: `neo4j@sha256:4bae36aff76271e27fd6a6ed0835413f86a284cd179cfb1cb7d188f5f7533aca`

Set `LETTA_IMAGE` or `NEO4J_IMAGE` only when deliberately testing another
version. Memobase used upstream commit
`358c16bbc6d687937d79bc2f984a11c3be8da901` (`v0.0.42`) plus
`patches/memobase-v0.0.42-reasoning-models.patch`. The patch is the complete
two-file source diff used by the experiments: it adjusts the startup sanity
check for reasoning models and adds reasoning-model request/output handling.

Memobase-derived material is covered by Apache License 2.0; see
`THIRD_PARTY_NOTICES.md` and `licenses/memobase-APACHE-2.0.txt`. The remainder
of this repository retains its top-level license.

Apply the patch to a clean checkout:

```bash
git clone https://github.com/memodb-io/memobase.git
export MEMOBASE_ROOT="$PWD/memobase"
git -C "$MEMOBASE_ROOT" checkout 358c16bbc6d687937d79bc2f984a11c3be8da901
git -C "$MEMOBASE_ROOT" apply \
  "$PWD/patches/memobase-v0.0.42-reasoning-models.patch"
```

Choose the corresponding non-secret template from
`artifact_configs/memobase/`, copy it to
`$MEMOBASE_ROOT/src/server/api/config.yaml`, and replace credential placeholders
locally. Never commit the rendered configuration.

## Experiment configurations

The shell templates are:

- `experiment_templates/cimemories_gpt_5_6_sol.env`
- `experiment_templates/cimemories_glm_5_3.env`
- `experiment_templates/cimemories_deepseek_v4.env`

They keep response-exposure judging on GPT-5.2 and semantic direct-memory
judging on GPT-6-sol. List direct-memory matching is deterministic and makes no
judge call. Graph and profile use the semantic judge. The templates contain no
credentials; export `OPENAI_API_KEY` and, for Together runs,
`TOGETHER_API_KEY` before launching.

The sanitized dataset and fixed context labels are available as
`artifact/results/inputs/dataset.json` and
`artifact/results/inputs/context_labels.json`. For a paid replication, either
pass those paths explicitly or copy them to the conventional names expected by
the example commands.

## Build the anonymous repository safely

After committing the intended source changes in the private development
repository, construct a clean tree without its history or unrelated untracked
files:

```bash
scripts/prepare_anonymous_repository.sh \
  ../anonymous-cimemories-artifact \
  research_outputs/cimemories-anonymous-github
```

Inspect that directory before running `git init`. Do not copy the original
`.git`, `.env*`, token files, operational backups, paper drafts, or unrelated
datasets. Follow the submission venue's policy if the source was previously
published, because identical public code can remain searchable even after Git
metadata is removed.

The committed `.gitattributes` marks the unrelated CWEval/code-style example
datasets as `export-ignore`. They remain available in the development
repository but are automatically omitted from the anonymous CIMemories tree
created by this script.
