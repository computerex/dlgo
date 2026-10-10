$ErrorActionPreference = 'Stop'
$env:DLGO_SSM_BATCH = '1'
$env:DLGO_GEMM_Q3K = '1'
$env:DLGO_SSM_TIME = '1'
# 10.8K tokens spans roughly 22 chunks; count prefix and initial suffix work.
$env:DLGO_SSM_TIME_CHUNKS = '24'
& .\bench\systemone\server-9b.exe -host 127.0.0.1 -port 8097 -gpu -ctx 16384 -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' -id qwen35-9b
if ($LASTEXITCODE -ne 0) { throw '9B profiling server failed' }
