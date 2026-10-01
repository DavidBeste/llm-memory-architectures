#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/reproduce_cimemories_experiments.sh MODEL STAGE [options]

MODEL:
  gpt | glm | deepseek | all

STAGE:
  responses   Generate the reported response stage (network calls; potentially costly)
  pre         Generate one pre-rerank repetition for --pipeline PATH entries
  memory      Evaluate persona-0 direct memory for three supplied pipelines
  check       Run /check_local_backends under the selected model environment
  status      Show detailed experiment progress for the selected model
  report      Export the locked, matched-repeat integrated paper artifacts (MODEL=all)

Options:
  --dataset FILE             Default: cimemories_raw.json
  --labels-file FILE         Default: labels_qwen.json
  --root DIR                 Default: research_outputs
  --pipeline PATH            Repeatable; required for STAGE=pre
  --list-pipeline PATH       Required for STAGE=memory
  --graph-pipeline PATH      Required for STAGE=memory
  --profile-pipeline PATH    Required for STAGE=memory
  --execute                  Execute the displayed commands; default is dry-run
  -h, --help

The script never reads credential files. Model launchers require credentials to
already be exported. Review the dry-run output before using --execute because
responses, pre, memory, and check can contact paid provider endpoints.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
model="$1"
stage="$2"
shift 2

dataset="cimemories_raw.json"
labels_file="labels_qwen.json"
output_root="research_outputs"
execute=false
pipelines=()
list_pipeline=""
graph_pipeline=""
profile_pipeline=""

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
    --root)
      [[ $# -ge 2 ]] || { printf '%s\n' 'Error: --root requires a directory.' >&2; exit 2; }
      output_root="$2"
      shift 2
      ;;
    --pipeline)
      [[ $# -ge 2 ]] || { printf '%s\n' 'Error: --pipeline requires a path.' >&2; exit 2; }
      pipelines+=("$2")
      shift 2
      ;;
    --list-pipeline)
      [[ $# -ge 2 ]] || { printf '%s\n' 'Error: --list-pipeline requires a path.' >&2; exit 2; }
      list_pipeline="$2"
      shift 2
      ;;
    --graph-pipeline)
      [[ $# -ge 2 ]] || { printf '%s\n' 'Error: --graph-pipeline requires a path.' >&2; exit 2; }
      graph_pipeline="$2"
      shift 2
      ;;
    --profile-pipeline)
      [[ $# -ge 2 ]] || { printf '%s\n' 'Error: --profile-pipeline requires a path.' >&2; exit 2; }
      profile_pipeline="$2"
      shift 2
      ;;
    --execute)
      execute=true
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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "$repo_dir"

case "$model" in
  gpt)
    launcher="scripts/start_cimemories_gpt_5_6_sol.sh"
    model_filter="gpt-5.6-sol"
    ;;
  glm)
    launcher="scripts/start_cimemories_glm_5_3.sh"
    model_filter="zai-org/GLM-5.3-Flash"
    ;;
  deepseek)
    launcher="scripts/start_cimemories_deepseek_v4.sh"
    model_filter="deepseek-ai/DeepSeek-V4-Flash-0731"
    ;;
  all)
    launcher=""
    model_filter=""
    ;;
  *)
    printf 'Error: MODEL must be gpt, glm, deepseek, or all; received: %s\n' "$model" >&2
    exit 2
    ;;
esac

printf -v dataset_arg '%q' "$dataset"
printf -v labels_arg '%q' "$labels_file"
printf -v root_arg '%q' "$output_root"
commands=()

case "$stage" in
  responses)
    [[ "$model" != all ]] || { printf '%s\n' 'Error: responses requires one model.' >&2; exit 2; }
    if [[ "$model" == gpt ]]; then
      for architecture in list-rerank graph-rerank profile-locomo-events-rerank; do
        commands+=("/run_privacy_pipeline_cimemories_dataset ${dataset_arg} ${architecture} --repeats 10 --labels-file ${labels_arg} --skip-memory-stage-metrics --resume-compatible")
      done
    else
      commands+=("/run_privacy_pipeline_cimemories ${dataset_arg} 0,1,2,3,4,5,6,7,8,9 list-rerank,graph-rerank,profile-locomo-events-rerank --repeats 1 --labels-file ${labels_arg} --skip-memory-stage-metrics --resume-compatible --with-pre-rerank")
    fi
    ;;
  pre)
    [[ "$model" != all ]] || { printf '%s\n' 'Error: pre requires one model.' >&2; exit 2; }
    [[ ${#pipelines[@]} -gt 0 ]] || { printf '%s\n' 'Error: pre requires at least one --pipeline PATH.' >&2; exit 2; }
    for pipeline in "${pipelines[@]}"; do
      printf -v pipeline_arg '%q' "$pipeline"
      commands+=("/generate_pre_rerank_responses ${pipeline_arg} --repeats 1")
    done
    ;;
  memory)
    [[ "$model" != all ]] || { printf '%s\n' 'Error: memory requires one model.' >&2; exit 2; }
    [[ -n "$list_pipeline" && -n "$graph_pipeline" && -n "$profile_pipeline" ]] || {
      printf '%s\n' 'Error: memory requires --list-pipeline, --graph-pipeline, and --profile-pipeline.' >&2
      exit 2
    }
    printf -v list_arg '%q' "$list_pipeline"
    printf -v graph_arg '%q' "$graph_pipeline"
    printf -v profile_arg '%q' "$profile_pipeline"
    commands+=("/compute_memory_stage_metrics ${list_arg} --strategy exact-match")
    commands+=("/compute_memory_stage_metrics ${graph_arg} --strategy monolithic")
    commands+=("/compute_memory_stage_metrics ${profile_arg} --strategy monolithic")
    ;;
  check)
    [[ "$model" != all ]] || { printf '%s\n' 'Error: check requires one model.' >&2; exit 2; }
    commands+=("/check_local_backends")
    ;;
  status)
    [[ "$model" != all ]] || { printf '%s\n' 'Error: status requires one model.' >&2; exit 2; }
    printf -v model_arg '%q' "$model_filter"
    commands+=("/cimemories_experiment_progress --root ${root_arg} --dataset cimemories_raw --model ${model_arg} --details")
    ;;
  report)
    [[ "$model" == all ]] || { printf '%s\n' 'Error: report requires MODEL=all.' >&2; exit 2; }
    commands+=("/export_cimemories_integrated_paper_artifacts --root ${root_arg} --dataset cimemories_raw --models gpt-5.6-sol,GLM-5.3-Flash,DeepSeek-V4-Flash --bootstrap-iterations 5000 --expected-personas 10 --memory-persona 0 --memory-judge-model gpt-6-sol --first-post-repeat --require-complete --require-memory-complete")
    ;;
  *)
    printf 'Error: unknown STAGE: %s\n' "$stage" >&2
    usage >&2
    exit 2
    ;;
esac

printf 'CIMemories reproduction plan: model=%s stage=%s mode=%s\n' \
  "$model" "$stage" "$([[ "$execute" == true ]] && printf execute || printf dry-run)"
for command in "${commands[@]}"; do
  printf '  %s\n' "$command"
done

if [[ "$execute" != true ]]; then
  printf '%s\n' 'Dry-run only. Re-run with --execute after reviewing the commands and active service configuration.'
  exit 0
fi

if [[ "$stage" == responses ]]; then
  [[ -f "$dataset" ]] || { printf 'Error: dataset not found: %s\n' "$dataset" >&2; exit 2; }
  [[ -f "$labels_file" ]] || { printf 'Error: labels file not found: %s\n' "$labels_file" >&2; exit 2; }
fi
if [[ "$stage" == pre ]]; then
  for pipeline in "${pipelines[@]}"; do
    [[ -e "$pipeline" ]] || { printf 'Error: pipeline not found: %s\n' "$pipeline" >&2; exit 2; }
  done
fi
if [[ "$stage" == memory ]]; then
  for pipeline in "$list_pipeline" "$graph_pipeline" "$profile_pipeline"; do
    [[ -e "$pipeline" ]] || { printf 'Error: pipeline not found: %s\n' "$pipeline" >&2; exit 2; }
  done
fi

command -v letta-chat >/dev/null 2>&1 || {
  printf '%s\n' 'Error: letta-chat is unavailable; activate the project environment first.' >&2
  exit 127
}

if [[ -n "$launcher" && "$stage" != status ]]; then
  printf '%s\n' "Executing through ${launcher}. Provider calls may incur charges."
  printf '%s\n' "${commands[@]}" | "$launcher"
else
  printf '%s\n' "${commands[@]}" | letta-chat
fi
