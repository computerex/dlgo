$ErrorActionPreference = 'Stop'
# Union mode must run with the GPU-capable trainer: it resolves the real
# Vulkan device-name suffix for GPU-cache key lookups (the CPU-only trainer's
# stub suffix would not match the gpu cache).
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& .\systemone-train-gpu.exe -model 'C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf' `
  -train systemone-train/data/generated/splits-gp2/train-aug.jsonl `
  -dev systemone-train/data/generated/splits-gp2/dev.jsonl `
  -calibration systemone-train/data/generated/splits-gp2/calibration.jsonl `
  -test systemone-train/data/generated/splits-gp2/test.jsonl `
  -device cpu `
  -feature-cache systemone-train/data/generated/gp2-features-cache-2b-cpu-fresh.json `
  -union-cache systemone-train/data/generated/gp2-features-cache-2b-gpu-backbone.json `
  -output systemone-train/artifacts/qwen38-2b-q4-readout-v2-uni.json `
  -gguf-output systemone-train/artifacts/qwen38-2b-q4-readout-v2-uni.gguf `
  -report systemone-train/artifacts/qwen38-2b-v2-uni-report.json `
  -epochs 100 -lr 0.02 -ctx 512
$code = $LASTEXITCODE
$sw.Stop()
Write-Output ("UNION_TRAIN_EXIT=" + $code + " HOURS=" + [math]::Round($sw.Elapsed.TotalHours,2))
if ($code -ne 0) { throw '2B union training failed' }
