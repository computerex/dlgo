$ErrorActionPreference = 'Continue'
$go = 'C:\Program Files\Go\bin\go.exe'
$env:DLGO_SSM_BATCH = '1'
& $go test -tags 'cgo vulkan' -ldflags '-linkmode internal' ./gpu -run '^(TestGemmQ3KRecoveredGPU|TestSSMBatchT32ParityGPU|TestAttentionBarrierParityGPU)$' -count=1 -v 2>&1 | Out-File -Encoding utf8 bench\systemone\quick-tests.log
Write-Output "GO_TEST_EXIT=$LASTEXITCODE"
