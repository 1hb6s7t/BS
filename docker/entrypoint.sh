#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-/workspace/bs}"

if [[ $# -eq 0 || "${1:-}" == "gpu-demo" ]]; then
  if [[ "${1:-}" == "gpu-demo" ]]; then
    shift
  fi
  exec python3 /usr/local/bin/bs-gpu-demo-runner.py "$@"
fi

exec "$@"
