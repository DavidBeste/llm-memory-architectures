#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for architecture in list graph profile; do
  "${script_dir}/run_cimemories_memory_experiment.sh" "${architecture}" \
    --dataset cimemories_raw.json \
    --labels-file labels_qwen.json
done
