$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

docker compose -f docker-compose.gpu.yml up --build --abort-on-container-exit

Write-Host ""
Write-Host "Docker GPU demo output:"
Write-Host "  output/docker_gpu_demo/docker_gpu_evidence.png"
Write-Host "  output/docker_gpu_demo/docker_gpu_demo_report.md"
Write-Host "  output/docker_gpu_demo/docker_gpu_demo_export.zip"
