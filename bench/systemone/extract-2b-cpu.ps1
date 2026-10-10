$ErrorActionPreference = 'Stop'
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& .\systemone-train.exe -model 'C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf' `
  -train systemone-train/data/generated/splits-gp2/train-aug.jsonl `
  -dev systemone-train/data/generated/splits-gp2/dev.jsonl `
  -calibration systemone-train/data/generated/splits-gp2/calibration.jsonl `
  -test systemone-train/data/generated/splits-gp2/test.jsonl `
  -device cpu `
  -feature-cache systemone-train/data/generated/gp2-features-cache-2b-cpu-fresh.json `
  -output systemone-train/artifacts/qwen38-2b-q4-readout-cpu-tmp.json `
  -report systemone-train/artifacts/qwen38-2b-cpu-tmp-report.json `
  -epochs 5 -lr 0.02 -ctx 512
$code = $LASTEXITCODE
$sw.Stop()
Write-Output ("CPU_EXTRACT_EXIT=" + $code + " HOURS=" + [math]::Round($sw.Elapsed.TotalHours,2))
if ($code -ne 0) { throw '2B CPU extraction failed' }
