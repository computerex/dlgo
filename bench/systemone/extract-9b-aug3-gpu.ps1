$ErrorActionPreference = 'Stop'
# Fill only the new train-aug3 feature rows. Stop the server first: one GPU job.
$env:DLGO_EXTRACT_PROGRESS_SECS = '15'
$env:DLGO_EXTRACT_CHECKPOINT = '50'
& .\systemone-train-gpu.exe -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' `
  -train systemone-train/data/generated/splits-gp2/train-aug3.jsonl `
  -dev systemone-train/data/generated/splits-gp2/dev.jsonl `
  -calibration systemone-train/data/generated/splits-gp2/calibration.jsonl `
  -test systemone-train/data/generated/splits-gp2/test.jsonl `
  -device gpu `
  -feature-cache systemone-train/data/generated/gp2-features-cache-9b-gpu.json `
  -output systemone-train/artifacts/qwen35-9b-aug3-gpu-tmp.json `
  -epochs 1 -ctx 512
if ($LASTEXITCODE -ne 0) { throw '9B aug3 GPU extraction failed' }
