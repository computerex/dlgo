$ErrorActionPreference = 'Stop'
$go = 'C:\Program Files\Go\bin\go.exe'
& $go build -tags vulkan -ldflags '-linkmode internal' -o bench/systemone/server-9b.exe ./cmd/dlgo-server
if ($LASTEXITCODE -ne 0) { throw 'server build failed' }
& $go build -tags vulkan -ldflags '-linkmode internal' -o bench/systemone/eval-9b.exe ./cmd/systemone-eval
if ($LASTEXITCODE -ne 0) { throw 'eval build failed' }
Write-Output '9B_TOOLS_BUILD=0'
