$ErrorActionPreference = 'Stop'

Set-Location -Path "$PSScriptRoot"

Write-Host "[1/3] Training suspicious classifier..." -ForegroundColor Cyan
conda run -n efficientnet_env --no-capture-output python train_gdc_suspicious.py

Write-Host "[2/3] Training risk classifier..." -ForegroundColor Cyan
conda run -n efficientnet_env --no-capture-output python train_gdc_risk.py

Write-Host "[3/3] Running two-stage inference on data/gdc.jpg..." -ForegroundColor Cyan
conda run -n efficientnet_env --no-capture-output python predict_gdc_pipeline.py --image data/gdc.jpg

Write-Host "Done. Check gdc_suspicious_results and gdc_risk_results for metrics and plots." -ForegroundColor Green
