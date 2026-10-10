$ErrorActionPreference = 'Stop'
$go = 'C:\Program Files\Go\bin\go.exe'
# CPU trainer (no vulkan tag): used for -device cpu feature extraction.
# Requires -linkmode internal: the external linker produces an invalid PE here.
& $go build -ldflags '-linkmode internal' -o systemone-train.exe ./cmd/systemone-train
if ($LASTEXITCODE -ne 0) { throw 'CPU trainer build failed' }
Write-Output "CPU_TRAIN_BUILD=$LASTEXITCODE"
# GPU trainer (vulkan): used for -device gpu extraction and union training.
& $go build -tags vulkan -ldflags '-linkmode internal' -o systemone-train-gpu.exe ./cmd/systemone-train
if ($LASTEXITCODE -ne 0) { throw 'GPU trainer build failed' }
Write-Output "GPU_TRAIN_BUILD=$LASTEXITCODE"
