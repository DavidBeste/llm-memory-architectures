#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
config_file="${repo_dir}/experiment_templates/cimemories_gpt_5_6_sol.env"

for credential in OPENAI_API_KEY TOGETHER_API_KEY; do
  if [[ -z "${!credential:-}" ]]; then
    printf 'Error: %s must already be exported in this shell.\n' "$credential" >&2
    exit 2
  fi
done

# shellcheck source=../experiment_templates/cimemories_gpt_5_6_sol.env
source "$config_file"
command -v letta-chat >/dev/null 2>&1 || {
  printf '%s\n' 'Error: letta-chat is unavailable; activate the project environment first.' >&2
  exit 127
}

printf '%s\n' \
  'Starting letta-chat with the GPT-5.6-sol CIMemories configuration:' \
  "  agent/reranker/Graphiti: ${LETTA_AGENT_MODEL}" \
  "  exposure judge:          ${LETTA_JUDGE_MODEL}" \
  "  returned-memory judge:   ${LETTA_MEMORY_STAGE_JUDGE_MODEL}" \
  "  list/graph embeddings:   ${LETTA_EMBEDDING_MODEL} (${LETTA_EMBEDDING_DIM})" \
  "  Memobase endpoint:       ${MEMOBASE_PROJECT_URL}" \
  '  credentials:             present (values hidden)' \
  '' \
  'Note: this launcher does not edit or restart the separate Memobase server.'

cd "$repo_dir"
exec letta-chat "$@"
