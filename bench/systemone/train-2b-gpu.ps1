$ErrorActionPreference = 'Stop'
$env:DLGO_SSM_BATCH = '1'
& .\systemone-train-gpu.exe -model 'C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf' -train systemone-train/data/generated/splits-gp2/train-aug.jsonl -dev systemone-train/data/generated/splits-gp2/dev.jsonl -calibration systemone-train/data/generated/splits-gp2/calibration.jsonl -test systemone-train/data/generated/splits-gp2/test.jsonl -device gpu -feature-cache systemone-train/data/generated/gp2-features-cache-2b-gpu-backbone.json -output systemone-train/artifacts/qwen38-2b-q4-readout-v1-gpu.json -gguf-output systemone-train/artifacts/qwen38-2b-q4-readout-v1-gpu.gguf -report systemone-train/artifacts/qwen38-2b-v1-gpu-report.json -epochs 100 -lr 0.02 -ctx 512
if ($LASTEXITCODE -ne 0) { throw "GPU training failed: $LASTEXITCODE" }
