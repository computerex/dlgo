# System One playground launcher.
# Starts the GPU decision server with the built frontend on :8080 and opens a
# browser. Models can be switched from inside the UI (left panel) at runtime;
# the launcher only picks the initially-resident backbone.
#
# Usage:
#   .\bench\systemone\playground.ps1                      # 0.8B (fast start)
#   .\bench\systemone\playground.ps1 -Size 9b             # 9B (slow start)
#   .\bench\systemone\playground.ps1 -Size 2b -NoBrowser
param(
    [ValidateSet('08b', '2b', '9b')]
    [string]$Size = '08b',
    [int]$Port = 8080,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$env:DLGO_SSM_BATCH = '1'   # batched SSM prefill (bit-identical, 2-3x)
$env:DLGO_GEMM_Q3K = '1'    # K1 Q3_K GEMM (validated: 0 decision changes, 2.7x on 9B)

$models = @{
    '08b' = @{ Path = 'C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf'; Id = 'qwen35-08b' }
    '2b'  = @{ Path = 'C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf';  Id = 'qwen38-2b' }
    '9b'  = @{ Path = 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf';  Id = 'qwen35-9b' }
}
$m = $models[$Size]
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$dist = Join-Path $root 'frontend\dist'
if (-not (Test-Path (Join-Path $dist 'index.html'))) {
    throw 'frontend not built - run: cd frontend; npm install; npm run build'
}

Write-Host "System One playground -> http://localhost:$Port  (backbone: $($m.Id))"
if (-not $NoBrowser) {
    Start-Process "http://localhost:$Port"
}
& (Join-Path $PSScriptRoot 'server-9b.exe') -host 127.0.0.1 -port $Port -gpu -ctx 8192 -model $m.Path -id $m.Id -frontend $dist
