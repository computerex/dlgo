$ErrorActionPreference = 'Stop'
Wait-Process -Id 19920 -ErrorAction SilentlyContinue
# Repeat the gate here so a failed prior process cannot lead to an unchecked benchmark.
& powershell -NoProfile -ExecutionPolicy Bypass -File bench/systemone/test-attention-plan.ps1
if ($LASTEXITCODE -ne 0) { throw 'Attention parity gate failed' }
& 'C:\Program Files\Go\bin\go.exe' build -tags 'cgo vulkan' -ldflags '-linkmode internal' -o bench/systemone/server-9b.exe ./cmd/dlgo-server
if ($LASTEXITCODE -ne 0) { throw 'Attention server build failed' }
& python bench/systemone/run_latency_plan.py --attention-only --runs 30
if ($LASTEXITCODE -ne 0) { throw 'Attention HTTP benchmark failed' }
& python bench/systemone/compare_latency_plan.py bench/systemone/latency-plan-attention-off.json bench/systemone/latency-plan-attention-on.json --output bench/systemone/latency-plan-attention-comparison.json
if ($LASTEXITCODE -ne 0) { throw 'Attention matched parity failed' }
