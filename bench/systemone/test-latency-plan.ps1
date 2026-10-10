$ErrorActionPreference = 'Stop'
$go = 'C:\Program Files\Go\bin\go.exe'
& 'C:\Program Files\Go\bin\gofmt.exe' -w gpu/forward.go gpu/ssm_batch.go
& $go test -a -tags 'cgo vulkan' -ldflags '-linkmode internal' ./gpu -run '^(TestSSMBatchT32ParityGPU|TestGemmA3KTiledGPU)$' -count=1 -v
if ($LASTEXITCODE -ne 0) { throw 'GPU numerical tests failed' }
& $go vet -tags 'cgo vulkan' ./gpu ./cmd/dlgo-server
if ($LASTEXITCODE -ne 0) { throw 'GPU vet failed' }
& $go build -tags 'cgo vulkan' -ldflags '-linkmode internal' -o bench/systemone/server-9b.exe ./cmd/dlgo-server
if ($LASTEXITCODE -ne 0) { throw 'Server build failed' }
Write-Output 'LATENCY_PLAN_TEST_BUILD_PASS'
