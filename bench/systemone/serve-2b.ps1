$ErrorActionPreference = 'Stop'
$env:DLGO_SSM_BATCH = '1'
& .\bench\systemone\server-2b.exe -host 127.0.0.1 -port 8097 -gpu -ctx 16384 -model 'C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf' -id qwen38-2b
if ($LASTEXITCODE -ne 0) { throw "2B server failed: $LASTEXITCODE" }
