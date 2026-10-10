$ErrorActionPreference = 'Stop'
$env:DLGO_SSM_BATCH = '1'
& .\bench\systemone\server-9b.exe -host 127.0.0.1 -port 8097 -gpu -ctx 16384 -model 'C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf' -id qwen35-08b
if ($LASTEXITCODE -ne 0) { throw "0.8B server failed: $LASTEXITCODE" }
