$ErrorActionPreference = 'Stop'
$go = 'C:\Program Files\Go\bin\go.exe'
& 'C:\Program Files\Go\bin\gofmt.exe' -w gpu/attention_kv4_test.go gpu/gpu_cgo.go
& $go test -tags 'cgo vulkan' -ldflags '-linkmode internal' ./gpu -run '^TestAttentionKV4ParityGPU$' -count=1 -v 2>&1 | Out-File -Encoding utf8 bench\systemone\kv4-test.log
Write-Output "KV4_TEST_EXIT=$LASTEXITCODE"
