$ErrorActionPreference = 'Stop'
$go = 'C:\Program Files\Go\bin\go.exe'
& $go build -tags vulkan -ldflags '-linkmode internal' -o systemone-train-gpu.exe ./cmd/systemone-train
if ($LASTEXITCODE -ne 0) { throw 'trainer build failed' }
Write-Output 'TRAINER_BUILD=0'
$env:DLGO_SSM_BATCH = '1'
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& .\systemone-train-gpu.exe -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' `
  -train systemone-train/data/seed_train.jsonl -device gpu -epochs 1 -ctx 512 `
  -output .harness_output/smoke-9b-readout.json
$code = $LASTEXITCODE
$sw.Stop()
Write-Output ("SMOKE_EXIT=" + $code + " SMOKE_SECONDS=" + [math]::Round($sw.Elapsed.TotalSeconds,1))
if ($code -ne 0) { throw 'smoke failed' }
