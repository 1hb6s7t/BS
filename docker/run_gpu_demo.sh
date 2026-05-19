#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

docker compose -f docker-compose.gpu.yml up --build --abort-on-container-exit

echo
echo "Docker GPU demo output:"
echo "  output/docker_gpu_demo/docker_gpu_evidence.png"
echo "  output/docker_gpu_demo/docker_gpu_demo_report.md"
echo "  output/docker_gpu_demo/docker_gpu_demo_export.zip"
