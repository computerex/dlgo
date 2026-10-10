$ErrorActionPreference = 'Stop'
$env:DLGO_SSM_BATCH = '1'
$env:DLGO_GEMM_Q3K = '1'
# Deleted experimental coopmat path must not be enabled.
Remove-Item Env:DLGO_GEMM_Q3K_COOPMAT -ErrorAction SilentlyContinue
& .\bench\systemone\server-9b.exe -host 127.0.0.1 -port 8097 -gpu -ctx 16384 -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' -id qwen35-9b
if ($LASTEXITCODE -ne 0) { throw '9B server failed' }
