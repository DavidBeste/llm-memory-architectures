#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/prepare_anonymous_repository.sh OUTPUT_DIR RESULTS_DIR

Create a new anonymous-review tree from the committed source snapshot and a
completed anonymized CIMemories results bundle. OUTPUT_DIR must not exist.
The original .git directory and all untracked files are excluded by design.
EOF
}

if [[ $# -ne 2 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  [[ $# -eq 1 ]] && exit 0
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
output_dir="$1"
results_dir="$2"

[[ ! -e "$output_dir" ]] || {
  printf 'Error: output already exists: %s\n' "$output_dir" >&2
  exit 2
}
[[ -d "$results_dir" ]] || {
  printf 'Error: results directory does not exist: %s\n' "$results_dir" >&2
  exit 2
}
results_dir="$(cd -- "$results_dir" && pwd)"
[[ -f "$results_dir/artifact_manifest.json" ]] || {
  printf 'Error: not an anonymized artifact: %s\n' "$results_dir" >&2
  exit 2
}
[[ -f "$results_dir/SHA256SUMS" ]] || {
  printf 'Error: artifact has no SHA256SUMS: %s\n' "$results_dir" >&2
  exit 2
}

(cd -- "$repo_dir" && python3 scripts/verify_anonymized_cimemories_artifact.py "$results_dir")

mkdir -p "$output_dir"
git -C "$repo_dir" archive --format=tar HEAD | tar -x -C "$output_dir"
mkdir -p "$output_dir/artifact"
cp -a "$results_dir" "$output_dir/artifact/results"

printf 'Anonymous source tree: %s\n' "$(cd "$output_dir" && pwd)"
printf '%s\n' 'The directory has no Git history. Review it before initializing a new repository.'
