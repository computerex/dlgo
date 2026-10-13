$ErrorActionPreference = 'Stop'
# Union retrain after aug3 caches are complete on both devices.
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& .\systemone-train-gpu.exe -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' `
  -train systemone-train/data/generated/splits-gp2/train-aug3.jsonl `
  -dev systemone-train/data/generated/splits-gp2/dev.jsonl `
  -calibration systemone-train/data/generated/splits-gp2/calibration.jsonl `
  -test systemone-train/data/generated/splits-gp2/test.jsonl `
  -device cpu `
  -feature-cache systemone-train/data/generated/gp2-features-cache-9b-cpu.json `
  -union-cache systemone-train/data/generated/gp2-features-cache-9b-gpu.json `
  -output systemone-train/artifacts/qwen35-9b-q3-readout-v4-uni.json `
  -gguf-output systemone-train/artifacts/qwen35-9b-q3-readout-v4-uni.gguf `
  -report systemone-train/artifacts/qwen35-9b-v4-uni-report.json `
  -epochs 100 -lr 0.02 -ctx 512
$code = $LASTEXITCODE
$sw.Stop()
Write-Output ('UNION_TRAIN_EXIT=' + $code + ' MINUTES=' + [math]::Round($sw.Elapsed.TotalMinutes,1))
if ($code -ne 0) { throw '9B v4 union training failed' }
