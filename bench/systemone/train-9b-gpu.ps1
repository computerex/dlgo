$ErrorActionPreference = 'Stop'
$env:DLGO_SSM_BATCH = '1'
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& .\systemone-train-gpu.exe -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' `
  -train systemone-train/data/generated/splits-gp2/train-aug.jsonl `
  -dev systemone-train/data/generated/splits-gp2/dev.jsonl `
  -calibration systemone-train/data/generated/splits-gp2/calibration.jsonl `
  -test systemone-train/data/generated/splits-gp2/test.jsonl `
  -device gpu -feature-cache systemone-train/data/generated/gp2-features-cache-9b-gpu.json `
  -output systemone-train/artifacts/qwen35-9b-q3-readout-v1-gpu.json `
  -gguf-output systemone-train/artifacts/qwen35-9b-q3-readout-v1-gpu.gguf `
  -report systemone-train/artifacts/qwen35-9b-v1-gpu-report.json `
  -epochs 100 -lr 0.02 -ctx 512
$code = $LASTEXITCODE
$sw.Stop()
Write-Output ("TRAIN_EXIT=" + $code + " TRAIN_HOURS=" + [math]::Round($sw.Elapsed.TotalHours,2))
if ($code -ne 0) { throw '9B training failed' }
