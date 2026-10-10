$go = 'C:\Program Files\Go\bin\go.exe'
& $go build -tags vulkan -ldflags '-linkmode internal' -o systemone-train-gpu.exe ./cmd/systemone-train
Write-Output "TRAIN_GPU_BUILD=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { throw 'GPU trainer build failed' }
& $go build -tags vulkan -ldflags '-linkmode internal' -o bench/systemone/eval-2b.exe ./cmd/systemone-eval
if ($LASTEXITCODE -ne 0) { throw 'GPU evaluator build failed' }
& $go build -tags vulkan -ldflags '-linkmode internal' -o bench/systemone/server-2b.exe ./cmd/dlgo-server
Write-Output "SERVER_BUILD=$LASTEXITCODE"
