$ErrorActionPreference = 'Stop'
$go = 'C:\Program Files\Go\bin\go.exe'
& $go test -tags 'cgo vulkan' -ldflags '-linkmode internal' ./gpu -run '^TestGemmQ3KRecoveredTimingGPU$' -count=1 -v 2>&1 | Out-File -Encoding utf8 bench\systemone\recovered-timing.log
if ($LASTEXITCODE -ne 0) { throw 'timing test failed' }
& $go vet -tags 'cgo vulkan' ./gpu
if ($LASTEXITCODE -ne 0) { throw 'vet failed' }
& $go build -tags 'cgo vulkan' -ldflags '-linkmode internal' -o bench/systemone/server-9b.exe ./cmd/dlgo-server
if ($LASTEXITCODE -ne 0) { throw 'server build failed' }
Write-Output 'RECOVERED_GATE_BUILD_PASS'
