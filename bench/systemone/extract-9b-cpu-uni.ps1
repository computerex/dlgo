$ErrorActionPreference = 'Stop'
# Full 9B CPU feature extraction for the device-agnostic (union) head.
# Sequential, bounded memory, checkpoints every 250 rows, ETA every 15s.
$env:DLGO_EXTRACT_PROGRESS_SECS = '15'
$env:DLGO_EXTRACT_CHECKPOINT   = '250'

$sw = [System.Diagnostics.Stopwatch]::StartNew()

& .\systemone-train.exe `
  -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' `
  -train systemone-train/data/generated/splits-gp2/train-aug.jsonl `
  -dev systemone-train/data/generated/splits-gp2/dev.jsonl `
  -calibration systemone-train/data/generated/splits-gp2/calibration.jsonl `
  -test systemone-train/data/generated/splits-gp2/test.jsonl `
  -device cpu `
  -feature-cache systemone-train/data/generated/gp2-features-cache-9b-cpu.json `
  -output systemone-train/artifacts/qwen35-9b-q3-readout-cpu-tmp.json `
  -report systemone-train/artifacts/qwen35-9b-cpu-tmp-report.json `
  -epochs 5 -lr 0.02 -ctx 512

$code = $LASTEXITCODE
$sw.Stop()
Write-Output ("CPU_EXTRACT_EXIT=" + $code + " HOURS=" + [math]::Round($sw.Elapsed.TotalHours, 2))
if ($code -ne 0) { throw '9B CPU extraction failed' }
