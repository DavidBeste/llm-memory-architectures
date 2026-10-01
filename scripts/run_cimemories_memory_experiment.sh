#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
config_file="${repo_dir}/experiment_templates/cimemories_retrieval50_rerank20.env"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_cimemories_memory_experiment.sh <list|graph|profile> [options]

Options:
  --dataset FILE       Dataset (default: cimemories_raw.json)
  --labels-file FILE   External labels (default: labels_qwen.json)
  --persona INDEX      Run only this zero-based persona instead of the full dataset
  --token-file FILE    OPENAI_API_KEY fallback file (default: openai_token.txt)
  --no-token-file      Require OPENAI_API_KEY to already be exported
  --allow-env-overrides
                       Allow existing non-secret environment settings to win
  --resume-compatible  Reuse only stages whose complete configuration matches
  --include-embeddings Include backend-exposed embeddings in initialization snapshots
  --dry-run            Print the non-secret configuration and slash command only
  -h, --help           Show this help

If OPENAI_API_KEY is unset, the launcher reads its first line from the token
file without printing it or placing it in process arguments.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

architecture="$1"
shift
dataset="cimemories_raw.json"
labels_file="labels_qwen.json"
persona_idx=""
token_file="openai_token.txt"
use_token_file=true
allow_env_overrides=false
resume_compatible=false
include_embeddings=false
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      [[ $# -ge 2 ]] || { printf '%s\n' 'Error: --dataset requires a file.' >&2; exit 2; }
      dataset="$2"
      shift 2
      ;;
    --labels-file)
      [[ $# -ge 2 ]] || { printf '%s\n' 'Error: --labels-file requires a file.' >&2; exit 2; }
      labels_file="$2"
      shift 2
      ;;
    --persona)
      [[ $# -ge 2 ]] || { printf '%s\n' 'Error: --persona requires a zero-based index.' >&2; exit 2; }
      [[ "$2" =~ ^[0-9]+$ ]] || { printf 'Error: --persona must be a non-negative integer: %s\n' "$2" >&2; exit 2; }
      persona_idx="$2"
      shift 2
      ;;
    --token-file)
      [[ $# -ge 2 ]] || { printf '%s\n' 'Error: --token-file requires a file.' >&2; exit 2; }
      token_file="$2"
      shift 2
      ;;
    --no-token-file)
      use_token_file=false
      shift
      ;;
    --allow-env-overrides)
      allow_env_overrides=true
      shift
      ;;
    --resume-compatible)
      resume_compatible=true
      shift
      ;;
    --include-embeddings)
      include_embeddings=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Error: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$architecture" in
  list) mode="list-rerank"; retrieval_label="letta_semantic_search" ;;
  graph) mode="graph-rerank"; retrieval_label="graphiti_advanced_search" ;;
  profile) mode="profile-locomo-events-rerank"; retrieval_label="memobase_profile_plus_query_selected_events" ;;
  *)
    printf 'Error: architecture must be list, graph, or profile; received: %s\n' "$architecture" >&2
    exit 2
    ;;
esac

cd "$repo_dir"
[[ -f "$dataset" ]] || { printf 'Error: dataset not found: %s\n' "$dataset" >&2; exit 2; }
[[ -f "$labels_file" ]] || { printf 'Error: labels file not found: %s\n' "$labels_file" >&2; exit 2; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ "$use_token_file" != true ]]; then
    printf '%s\n' 'Error: OPENAI_API_KEY is unset and --no-token-file was requested.' >&2
    exit 2
  fi
  [[ -f "$token_file" ]] || {
    printf 'Error: OPENAI_API_KEY is unset and token file was not found: %s\n' "$token_file" >&2
    exit 2
  }
  IFS= read -r OPENAI_API_KEY < "$token_file" || true
  OPENAI_API_KEY="${OPENAI_API_KEY%$'\r'}"
  [[ -n "$OPENAI_API_KEY" ]] || {
    printf 'Error: token file has no non-empty first line: %s\n' "$token_file" >&2
    exit 2
  }
  export OPENAI_API_KEY
fi

# This file intentionally contains only non-secret settings. It aliases the
# inherited OPENAI_API_KEY for Graphiti clients without displaying its value.
# shellcheck source=../experiment_templates/cimemories_retrieval50_rerank20.env
export CIMEMORIES_ALLOW_ENV_OVERRIDES="$allow_env_overrides"
source "$config_file"

printf -v dataset_arg '%q' "$dataset"
printf -v labels_arg '%q' "$labels_file"
if [[ -n "$persona_idx" ]]; then
  pipeline_command="/run_privacy_pipeline_cimemories ${dataset_arg} ${persona_idx} ${mode} --labels-file ${labels_arg}"
  experiment_scope="persona-${persona_idx}"
else
  pipeline_command="/run_privacy_pipeline_cimemories_dataset ${dataset_arg} ${mode} --labels-file ${labels_arg}"
  experiment_scope="full-dataset"
fi
if [[ "$resume_compatible" == true ]]; then
  pipeline_command+=" --resume-compatible"
fi
if [[ "$include_embeddings" == true ]]; then
  pipeline_command+=" --include-embeddings"
fi

environment_policy="strict-template"
if [[ "$allow_env_overrides" == true ]]; then
  environment_policy="inherited-overrides"
fi

printf '%s\n' "CIMemories experiment: ${architecture}"
printf '%s\n' "  mode=${mode}"
printf '%s\n' "  scope=${experiment_scope}"
printf '%s\n' "  retrieval_source=${retrieval_label}"
if [[ "$architecture" == "profile" ]]; then
  printf '%s\n' "  native_context_tokens<=${MEMOBASE_CONTEXT_MAX_TOKEN_SIZE}"
  printf '%s\n' "  persistent_profile=preserved"
  printf '%s\n' "  event_candidates=all query-selected events in native context"
  printf '%s\n' "  reranked_events<=${LETTA_RERANK_OUTPUT_LIMIT}"
else
  printf '%s\n' "  retrieval_candidates<=${LETTA_RERANK_CANDIDATE_LIMIT}"
  printf '%s\n' "  reranker_output<=${LETTA_RERANK_OUTPUT_LIMIT}"
fi
printf '%s\n' "  agent_model=${LETTA_AGENT_MODEL}"
printf '%s\n' "  reranker_model=${LETTA_RERANK_MODEL}"
if [[ "$architecture" != "profile" ]]; then
  printf '%s\n' "  embedding_model=${LETTA_EMBEDDING_MODEL}"
  printf '%s\n' "  embedding_dimension=${LETTA_EMBEDDING_DIM}"
fi
printf '%s\n' "  labels=${labels_file}"
printf '%s\n' "  environment_policy=${environment_policy}"
printf '%s\n' "  resume_compatible=${resume_compatible}"
printf '%s\n' "  snapshot_embeddings=${include_embeddings}"

if [[ "$dry_run" == true ]]; then
  printf '%s\n' "  command=${pipeline_command}"
  exit 0
fi

command -v letta-chat >/dev/null 2>&1 || {
  printf '%s\n' 'Error: letta-chat is not available in this environment. Activate the project environment first.' >&2
  exit 127
}

# Do not echo the environment: it may contain credentials inherited from the
# caller. EOF after this command exits the interactive CLI when the run ends.
printf '%s\n' "$pipeline_command" | letta-chat
