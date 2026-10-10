$ErrorActionPreference = 'Stop'
$go = 'C:\Program Files\Go\bin\go.exe'
& $go test -tags 'cgo vulkan' -ldflags '-linkmode internal' ./gpu -run '^(TestAttentionBarrierParityGPU|TestSSMBatchT32ParityGPU)$' -count=1 -v
if ($LASTEXITCODE -ne 0) { throw 'Attention/T32 tests failed' }
& $go vet -tags 'cgo vulkan' ./gpu
if ($LASTEXITCODE -ne 0) { throw 'GPU vet failed' }
