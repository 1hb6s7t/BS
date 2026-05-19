param(
    [string]$OutputDir = "output/defense_gpu_visual_live",
    [string]$InputImage = "CRA/test/images/1.png",
    [string]$MaskImage = "CRA/test/masks/1.png",
    [string]$CraCkpt = "ckpt/generator_epoch11_batch56358.ckpt",
    [string]$SrganCkpt = "ckpt/pretrained_generator_epoch100000.ckpt"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "============================================================"
Write-Host "CRA + SRGAN real GPU visual demo"
Write-Host "Project:    $ProjectRoot"
Write-Host "OutputDir:  $OutputDir"
Write-Host "Input:      $InputImage"
Write-Host "Mask:       $MaskImage"
Write-Host "CRA ckpt:   $CraCkpt"
Write-Host "SRGAN ckpt: $SrganCkpt"
Write-Host "============================================================"

docker compose -f docker-compose.gpu.yml run --rm `
  -e CRSR_SRGAN_REPAIR_BLEND=1.0 `
  -e CRSR_SRGAN_REPAIR_SMOOTH=1.0 `
  -e CRSR_SRGAN_REPAIR_SMOOTH_SIGMA=8.0 `
  -e CRSR_SRGAN_GLOBAL_DENOISE=0.22 `
  -e CRSR_SRGAN_GLOBAL_DENOISE_SIGMA=0.75 `
  gpu-demo gpu-demo `
  --input_image $InputImage `
  --mask_image $MaskImage `
  --cra_ckpt $CraCkpt `
  --srgan_ckpt $SrganCkpt `
  --scale 4 `
  --output_dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "Docker GPU demo failed with exit code $LASTEXITCODE"
}

$Python = "python"
if (Test-Path ".\.venv-ms\Scripts\python.exe") {
    $Python = ".\.venv-ms\Scripts\python.exe"
}

$ReportPath = & $Python ".\tools\generate_gpu_visual_report.py" `
  --output_dir $OutputDir `
  --input_image $InputImage `
  --mask_image $MaskImage

if ($LASTEXITCODE -ne 0) {
    throw "Visual report generation failed with exit code $LASTEXITCODE"
}

$ReportFullPath = Resolve-Path $ReportPath.Trim()
Write-Host "============================================================"
Write-Host "Visual report: $ReportFullPath"
Write-Host "Output folder: $(Resolve-Path $OutputDir)"
Write-Host "============================================================"

Start-Process $ReportFullPath
Start-Process (Resolve-Path $OutputDir)
